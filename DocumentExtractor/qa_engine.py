import logging
from typing import Dict, Any, List

from llama_index.core.llms import ChatMessage
from llama_index.llms.google_genai import GoogleGenAI

try:
    from .utils import MemoryManager
    from .config import GENERATION_MODEL, GEMINI_API_KEY
except Exception:
    from utils import MemoryManager
    from config import GENERATION_MODEL, GEMINI_API_KEY


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


SYSTEM_PROMPT = (
    "You are a precise assistant that answers ONLY using the provided context. "
    "If the answer is not fully supported by the context, respond: 'No info in docs about the asked question.' "
    "Output STRICT HTML only (no markdown). Use <h4>, <p>, <ul>, <li>. Keep paragraphs short. "
    "Do not insert extra blank lines or empty elements. No inline styles/scripts."
)


class QAEngine:
    def __init__(self, memory_manager: MemoryManager | None = None) -> None:
        self.memory = memory_manager or MemoryManager()
        self.llm = GoogleGenAI(model_name=GENERATION_MODEL, api_key=GEMINI_API_KEY)

    def _build_context(self, nodes: List[Any]) -> str:
        return self.memory.compress_nodes_to_context(nodes)[0]

    def _generate(self, question: str, context: str) -> str:
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Return ONLY valid HTML fragment following these rules:\n"
            "- Use <h4> for section headings\n"
            "- Use <p> for short paragraphs (1-3 lines)\n"
            "- Use <ul><li> for bullet lists; no blank lines between <li>\n"
            "- Do not include <html>, <body>, styles, or scripts\n"
            "- No content beyond what is supported by the context\n"
            "- Do NOT add any file name citations like [filename] or similar"
        )
        messages = [ChatMessage(role="user", content=prompt)]
        resp = self.llm.chat(messages)
        if hasattr(resp, "content"):
            return resp.content
        if hasattr(resp, "message") and hasattr(resp.message, "content"):
            return resp.message.content
        return str(resp)

    def _validate_answer(self, question: str, context: str, draft: str) -> str:
        validation_prompt = (
            "Validate the following answer STRICTLY against the context. "
            "Remove any claims not supported by the context. If little is supported, reply: 'No info in docs about the asked question.'\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\nDraft Answer:\n{draft}\n\n"
            "Return ONLY the corrected final answer as a clean HTML fragment (<h4>, <p>, <ul>, <li>), no extra whitespace. "
            "Do NOT add any file name citations like [filename] or similar."
        )
        messages = [ChatMessage(role="user", content=validation_prompt)]
        resp = self.llm.chat(messages)
        if hasattr(resp, "content"):
            return resp.content
        if hasattr(resp, "message") and hasattr(resp.message, "content"):
            return resp.message.content
        return str(resp)

    def answer(self, query: str) -> Dict[str, Any]:
        nodes = self.memory.retrieve(query, top_k=14, rerank_k=8, min_relevance=0.35)
        if not nodes:
            status = self.memory.get_status()
            if not status.get("has_data", False):
                dynamic_msg = "No documents uploaded yet. Please upload PDF/DOCX and try again."
                nearest_sources: list[str] = []
            else:
                # Get nearest documents (low threshold) to craft a helpful dynamic message
                near_nodes = self.memory.retrieve(query, top_k=5, rerank_k=3, min_relevance=0.0)
                seen: set[str] = set()
                nearest_sources = []
                for n in near_nodes:
                    try:
                        fname = n.node.metadata.get("file_name", "unknown") if hasattr(n, 'node') else n.metadata.get("file_name", "unknown")
                    except Exception:
                        fname = "unknown"
                    if fname and fname not in seen:
                        seen.add(fname)
                        nearest_sources.append(fname)
                qshort = (query[:70] + "…") if len(query) > 70 else query
                if nearest_sources:
                    dynamic_msg = f"No info in docs about the asked question: '{qshort}'. Closest documents: {', '.join(nearest_sources)}."
                else:
                    dynamic_msg = f"No info in docs about the asked question: '{qshort}'."

            return {
                "query": query,
                "answer": dynamic_msg,
                "source": "no_documents",
                "document_nodes": [],
                "confidence": 0.0,
                "sources": nearest_sources,
            }
        context, sources = self.memory.compress_nodes_to_context(nodes)
        draft = self._generate(query, context)
        answer = self._validate_answer(query, context, draft)
        confidence = self.memory.compute_confidence(nodes)
        return {
            "query": query,
            "answer": answer,
            "source": "document_rag",
            "document_nodes": nodes,
            "confidence": confidence,
            "sources": sources,
        }


