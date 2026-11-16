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

Supported document formats are TXT, MD, CSV, JSON, LOG, DOCX, XLSX, and text-based PDF. DOCX/XLSX archive member counts and uncompressed sizes are limited. Scanned PDFs need a separately approved local OCR solution.

## Tests

```powershell
python -m unittest discover -s tests -v
python -m py_compile ollama_persian_rag.py
```

Tests use synthetic files and temporary SQLite databases only. They do not contact Ollama or access the live database.

## Architecture direction

The entry point remains `ollama_persian_rag.py` for compatibility. SQLite responsibilities are isolated in `app/db/`; `app/config.py` centralizes the established environment configuration; and `app/services/ollama_client.py` owns local Ollama HTTP requests and streaming validation. The next incremental boundaries are documented in [ROADMAP.md](ROADMAP.md): chat/RAG/ingestion services, HTTP handlers, static UI assets, and focused tests. Preserve API and database compatibility while extracting one boundary at a time.
