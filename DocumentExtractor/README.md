## DocumentExtractor (Standalone)

A document-only RAG app. Upload PDF/DOCX and ask questions. Built for higher accuracy using hybrid retrieval (default + MMR with RRF fusion), grounded prompting, and a strict validation pass.

### Features
- Auto-upload on file selection (no extra button)
- Persistent success banner (shows uploaded filenames)
- Chat enabled only after indexing completes
- Answers only from uploaded documents
- If found in docs, shows a single document reference line
- Dynamic fallback when no answer in docs:
  - If no docs: “No documents uploaded yet. Please upload PDF/DOCX and try again.”
  - If docs exist: “No info in docs about the asked question: '<your query…>'. Closest documents: <doc names>.”
- Clear Index button fully resets the store and uploader in one click

### Quick Start

1) Create a virtual environment (Windows PowerShell example):
```powershell
python -m venv .venv
.\n+venv\Scripts\Activate.ps1
```

2) Install dependencies inside the `DocumentExtractor/` folder:
```bash
pip install -r requirements.txt
```

3) Add your API key (create `.env` in `DocumentExtractor/` or set env var). The app loads `.env` from this folder:
```env
GEMINI_API_KEY=your_key_here
```

4) Run the app from the `DocumentExtractor/` folder:
```bash
streamlit run app.py
```

### How It Works
- Embeddings/LLM: Google Gemini
- Vector DB: FAISS (inner product)
- Retrieval: Default + MMR, fused via RRF, score thresholding, final rerank
- Context: Compressed with source tags, budgeted character length
- Validation: Second-pass LLM strips unsupported claims

### UI Flow
1) Select PDF/DOCX files – upload starts automatically.
2) See “Uploaded: <filenames>. Indexed successfully. Search is now enabled.” (persists above chat)
3) Ask questions – the input is available only after indexing finishes.
4) Clear Chat resets the conversation; Clear Uploaded Index wipes the vector store and resets uploader state immediately.

### Configuration (edit `config.py`)
- `EMBEDDING_MODEL`: "models/embedding-001"
- `GENERATION_MODEL`: "gemini-2.5-pro"
- `EMBEDDING_DIMENSION`: 768
- Paths are scoped under `DocumentExtractor/` (`uploads/`, `vector_store/`)

### Tuning Retrieval (edit `utils.py`)
- `Settings.chunk_size = 800`, `Settings.chunk_overlap = 120`
- `retrieve(query, top_k=14, rerank_k=8, min_relevance=0.35)`
- `compress_nodes_to_context(max_chars=12000)`

### Troubleshooting
- GEMINI_API_KEY missing: ensure `.env` contains the key or that the env var is set. The app will error on start if it’s missing.
- Ran from the wrong folder: run `streamlit run app.py` from inside the `DocumentExtractor/` directory (or use package-relative imports as configured).
- PDFs not extracting text: scanned PDFs may require OCR; PyMuPDF extracts embedded text only.
- “No info in docs …” but relevant doc exists: lower `min_relevance` in `retrieve`, increase `top_k`, or re-upload to ensure text was extracted.
- Clear index didn’t reset uploader: this version forces a rerun and bumps the uploader key; try again if needed.

### Project Structure
```
DocumentExtractor/
├── app.py                # Streamlit UI (auto-upload, chat, clear index)
├── qa_engine.py          # Grounded prompting + validation pass
├── utils.py              # FAISS + LlamaIndex, hybrid retrieval, compression
├── runtime_docs.py       # Upload handling and ingestion
├── config.py             # Paths and model settings; reads GEMINI_API_KEY
├── requirements.txt      # Dependencies for standalone usage
└── README.md             # This file
```

### Security & Privacy
- Documents are processed locally.
- Keys are read from `.env` or environment variables.

### Notes
- Only PDF and DOCX are supported out of the box.
- Document reference shown is a single best-fit filename to keep UI clean.



