# Offline Persian Company AI Assistant

This is a local, offline interface for Persian conversations with Ollama/Gemma, approved organizational knowledge, SQLite chat history, and document retrieval (RAG). It binds to `127.0.0.1` by default and uses no external API.

## Run

The application requires Python 3.10+ and a local Ollama service. `pypdf` is optional and only needed for text-based PDF extraction. From Anaconda Prompt on Windows:

```powershell
python ollama_persian_rag.py
```

Open `http://127.0.0.1:8765/`. Optional environment variables:

- `OLLAMA_URL` — defaults to `http://127.0.0.1:11434`
- `OLLAMA_UI_PORT` — defaults to `8765`
- `OLLAMA_EMBED_MODEL` — an explicitly selected local embedding model only

Gemma models are selected from locally available Ollama models. The application does not download, change, or re-index embedding models automatically.

## Data safety and operations

The live database is `ollama_company_data/company_memory.sqlite3`. Never delete, reset, or commit it or its WAL/SHM companions. Before an upgrade, stop the application and copy the complete data directory. Imported documents are stored as extracted chunks; the original uploaded file is not retained.

At startup, the application applies only additive migrations and records schema version `1` with `PRAGMA user_version`. Existing tables and records are retained. `backup_database()` creates a consistent SQLite backup through SQLite's backup API; by default it writes a new timestamped database under `ollama_company_data/backups/`. It never overwrites an existing backup.

`GET /api/health` is a local diagnostic endpoint. It reports local Ollama connectivity, embedding-model availability, SQLite connectivity, schema version, document/chunk counts, integrity-check result, and foreign-key violation count. The embedding object also reports model presence, vector dimension when available, a non-sensitive error category, and whether lexical fallback is active. Operational logs are local and structured in `ollama_company_data/assistant.log`; they intentionally exclude request bodies, document content, prompts, embeddings, and credentials.

`GET /api/rag-monitoring` is a read-only local monitoring endpoint. Its `health` object preserves the `/api/health` diagnostics, while its `index` object reports document and chunk totals, vector coverage, embedding-model distribution, vector dimensions, invalid vectors, model incompatibility, and configured-model dimension mismatches. It never returns document text, vectors, prompts, or credentials. The Persian RTL interface exposes the same data through **پایش RAG و Embedding** and only refreshes when opened or when its manual refresh button is selected.

The source event of `/api/chat-stream` may additionally include `retrieval_diagnostics`. These diagnostics are request-scoped: they describe the retrieval that created that source event, including actual mode (`lexical` or `hybrid`), embedding attempt/result, fallback category, candidate/selected chunk counts, citation identifiers, and elapsed milliseconds. They do not claim semantic retrieval merely because the embedding health check succeeded.

Supported document formats are TXT, MD, CSV, JSON, LOG, DOCX, XLSX, and text-based PDF. DOCX/XLSX archive member counts and uncompressed sizes are limited. Scanned PDFs need a separately approved local OCR solution.

## Tests

```powershell
python -m unittest discover -s tests -v
python -m py_compile ollama_persian_rag.py
```

Tests use synthetic files and temporary SQLite databases only. They do not contact Ollama or access the live database.

## Architecture direction

The entry point remains `ollama_persian_rag.py` for compatibility. SQLite responsibilities are isolated in `app/db/`; `app/config.py` centralizes the established environment configuration; `app/services/ollama_client.py` owns local Ollama HTTP requests; chat services coordinate bounded prompts and streaming; document services preserve local formats and atomic persistence; retrieval/citation services preserve Persian ranking and evidence identifiers; and `rag_monitoring.py` calculates read-only coverage statistics. The next incremental boundaries are documented in [ROADMAP.md](ROADMAP.md): HTTP handlers, static UI assets, and focused tests. Preserve API and database compatibility while extracting one boundary at a time.

## Prompt budgeting

Prompt construction reserves at least 256 tokens, or one quarter of `num_ctx`, for model output. Without an approved tokenizer dependency, it estimates two characters per input token. System instructions and the current user turn are allocated first, followed by retrieved evidence and newest conversation history. Any shortened section receives a Persian truncation marker where capacity permits and is recorded by identifier-only diagnostics; source contents are never logged. This is deterministic but remains an estimate, so Ollama still enforces the final token limit.
