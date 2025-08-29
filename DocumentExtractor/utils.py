import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import shutil

import faiss
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader

try:
    from .config import (
        VECTOR_STORE_DIR,
        EMBEDDING_MODEL,
        GENERATION_MODEL,
        GEMINI_API_KEY,
        EMBEDDING_DIMENSION,
    )
except Exception:
    # Standalone execution fallback
    from config import (
        VECTOR_STORE_DIR,
        EMBEDDING_MODEL,
        GENERATION_MODEL,
        GEMINI_API_KEY,
        EMBEDDING_DIMENSION,
    )


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MemoryManager:
    """Document-only FAISS + LlamaIndex manager with accuracy-focused retrieval."""

    def __init__(self) -> None:
        self.vector_store_path = Path(VECTOR_STORE_DIR)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)

        # Configure LlamaIndex
        Settings.embed_model = GoogleGenAIEmbedding(model_name=EMBEDDING_MODEL, api_key=GEMINI_API_KEY)
        Settings.llm = GoogleGenAI(model_name=GENERATION_MODEL, api_key=GEMINI_API_KEY)
        Settings.chunk_size = 800
        Settings.chunk_overlap = 120

        self.vector_store = None
        self.doc_store = None
        self.index_store = None
        self.storage_context = None
        self.index = None

        self._load_or_initialize()

    def _load_or_initialize(self) -> None:
        try:
            if any(self.vector_store_path.iterdir()):
                self.storage_context = StorageContext.from_defaults(persist_dir=str(self.vector_store_path))
                self.index = load_index_from_storage(self.storage_context, embed_model=Settings.embed_model)
                self.vector_store = self.storage_context.vector_store
                self.doc_store = self.storage_context.docstore
                self.index_store = getattr(self.storage_context, 'index_store', SimpleIndexStore())
                logging.info("Loaded existing index")
            else:
                self._init_new()
        except Exception as e:
            logging.warning(f"Failed to load index, creating new. Reason: {e}")
            self._init_new()

    def _init_new(self) -> None:
        self.vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatIP(EMBEDDING_DIMENSION))
        # Normalize embeddings recommended for IP; LlamaIndex Google embeddings are normalized.
        self.doc_store = SimpleDocumentStore()
        self.index_store = SimpleIndexStore()
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            docstore=self.doc_store,
            index_store=self.index_store,
        )
        self.index = VectorStoreIndex([], storage_context=self.storage_context, embed_model=Settings.embed_model)
        self.persist()
        logging.info("Initialized new index")

    def persist(self) -> None:
        try:
            self.storage_context.persist(persist_dir=str(self.vector_store_path))
        except Exception as e:
            logging.error(f"Failed to persist index: {e}")

    def clear_all(self) -> None:
        """Delete all stored vectors and re-initialize an empty index."""
        try:
            if self.vector_store_path.exists():
                shutil.rmtree(self.vector_store_path)
        except Exception as e:
            logging.error(f"Failed to clear vector store: {e}")
        self._init_new()

    def add_documents(self, file_paths: List[Path]) -> bool:
        splitter = SentenceSplitter(chunk_size=Settings.chunk_size, chunk_overlap=Settings.chunk_overlap)
        new_nodes = []

        for file_path in file_paths:
            try:
                docs = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()
                if not docs:
                    logging.warning(f"No content loaded from {file_path}")
                    continue
                nodes = splitter.get_nodes_from_documents(docs, show_progress=False)
                for node in nodes:
                    node.metadata["file_name"] = file_path.name
                    node.metadata["source_type"] = "uploaded_doc"
                    new_nodes.append(node)
            except Exception as e:
                logging.error(f"Failed processing {file_path.name}: {e}")

        if not new_nodes:
            logging.warning("No nodes to insert")
            return False

        # Insert and persist
        self.index.insert_nodes(new_nodes)
        self.persist()
        return True

    def _get_node_id(self, node_with_score: Any) -> str:
        try:
            if hasattr(node_with_score, 'node') and hasattr(node_with_score.node, 'node_id'):
                return node_with_score.node.node_id
            if hasattr(node_with_score, 'node') and hasattr(node_with_score.node, 'id_'):
                return node_with_score.node.id_
            if hasattr(node_with_score, 'id_'):
                return node_with_score.id_
        except Exception:
            pass
        # Fallback to hash of text
        text = ''
        try:
            text = node_with_score.node.text if hasattr(node_with_score, 'node') else node_with_score.text
        except Exception:
            pass
        return f"auto_{abs(hash(text))}"

    def _rrf_merge(self, lists: List[List[Any]], k: int) -> List[Any]:
        # Reciprocal Rank Fusion merge for robustness across retrievers
        scores: Dict[str, float] = {}
        ref: Dict[str, Any] = {}
        c = 60  # RRF constant
        for results in lists:
            for rank, item in enumerate(results, start=1):
                nid = self._get_node_id(item)
                ref[nid] = item
                scores[nid] = scores.get(nid, 0.0) + 1.0 / (c + rank)
        ranked_ids = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
        return [ref[i] for i in ranked_ids[:k]]

    def compress_nodes_to_context(self, nodes: List[Any], max_chars: int = 12000) -> Tuple[str, List[str]]:
        pieces = []
        sources: List[str] = []
        total = 0
        for n in nodes:
            try:
                text = n.node.text if hasattr(n, 'node') else n.text
                fname = n.node.metadata.get('file_name', 'unknown') if hasattr(n, 'node') else n.metadata.get('file_name', 'unknown')
            except Exception:
                text, fname = '', 'unknown'
            snippet = f"[Source: {fname}]\n{text}"
            length = len(snippet)
            if total + length > max_chars:
                # truncate last snippet to fit budget
                remaining = max(0, max_chars - total)
                if remaining > 0:
                    pieces.append(snippet[:remaining])
                    sources.append(fname)
                    total += remaining
                break
            pieces.append(snippet)
            sources.append(fname)
            total += length
        return ("\n\n---\n\n".join(pieces), list(dict.fromkeys(sources)))

    def compute_confidence(self, nodes: List[Any]) -> float:
        scores = []
        for n in nodes:
            s = getattr(n, 'score', None)
            if s is not None:
                scores.append(float(s))
        if not scores:
            return 0.5
        # clip to [0,1]
        mean_score = sum(max(0.0, min(1.0, s)) for s in scores) / len(scores)
        return round(mean_score, 3)

    def retrieve(self, query: str, top_k: int = 12, rerank_k: int = 8, min_relevance: float = 0.35) -> List[Any]:
        # Guard: no index or empty store → no retrieval
        if self.index is None or self.vector_store is None:
            return []
        try:
            ntotal = getattr(self.vector_store, 'client', None).ntotal  # type: ignore[attr-defined]
        except Exception:
            ntotal = 0
        if not ntotal or ntotal <= 0:
            return []

        # Two retrievers: default and MMR, then RRF-merge
        retriever_default = self.index.as_retriever(
            similarity_top_k=top_k,
            vector_store_query_mode="default",
        )
        retriever_mmr = self.index.as_retriever(
            similarity_top_k=top_k,
            vector_store_query_mode="mmr",
        )
        try:
            results_default = retriever_default.retrieve(query)
        except Exception:
            results_default = []
        try:
            results_mmr = retriever_mmr.retrieve(query)
        except Exception:
            results_mmr = []
        merged = self._rrf_merge([results_default, results_mmr], k=top_k)

        # Filter by score threshold
        filtered = []
        for node in merged:
            score = getattr(node, 'score', None)
            if score is None or score >= min_relevance:
                filtered.append(node)

        # Final rerank by score
        filtered.sort(key=lambda n: getattr(n, 'score', 0.0), reverse=True)
        return filtered[:rerank_k]

    def get_status(self) -> Dict[str, Any]:
        vector_count = self.vector_store.client.ntotal if self.vector_store else 0
        doc_count = len(self.doc_store.docs) if self.doc_store else 0
        return {"index_size": vector_count, "doc_count": doc_count, "has_data": vector_count > 0 or doc_count > 0}


