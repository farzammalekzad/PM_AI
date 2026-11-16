# Architecture Assessment and Roadmap

## Current architecture and end-to-end flow

`ollama_persian_rag.py` remains the compatibility entry point and combines a standard-library HTTP server and inline HTML/JavaScript. SQLite connection configuration, migrations, backups, integrity diagnostics, and persistence operations are isolated in `app/db/`. Established environment settings are in `app/config.py`; local Ollama HTTP communication and NDJSON validation are in `app/services/ollama_client.py`; prompt construction and chat-stream persistence are in `app/services/prompt_builder.py` and `chat_service.py`; supported local document extraction and ingestion are in `document_extractor.py` and `document_ingestion.py`; Persian retrieval and citation formatting are in `retrieval_service.py` and `citation_service.py`; and read-only index coverage monitoring is in `rag_monitoring.py`.

1. The Persian RTL UI sends a request to `/api/chat-stream`. The server retrieves approved knowledge and document chunks, loads prior messages, and streams local Ollama `/api/chat` NDJSON to the browser. User and assistant messages are stored only after a completed stream.
2. Approved manual knowledge is stored in `knowledge` and retrieved through normalized lexical matching.
3. A document arrives as base64, is extracted from TXT/DOCX/XLSX/PDF, chunked, hashed, and stored in `documents` and `doc_chunks`. If `OLLAMA_EMBED_MODEL` is explicitly configured, chunk embeddings come from the local Ollama endpoint; otherwise lexical retrieval remains active.
4. Retrieval combines lexical scoring with cosine similarity only for vectors produced by the configured embedding model. Prompt citations use `K#` and `D#-C#` identifiers.
5. The local `/api/rag-monitoring` endpoint combines existing health diagnostics with SQLite-derived embedding coverage. Per-request retrieval diagnostics are sent only with the corresponding chat source event and displayed in the Persian RTL monitoring view.

Current strengths include localhost-only binding, parameterized SQL, baseline CSP, `textContent` UI insertion, streaming, size limits, citations, Persian `lang="fa" dir="rtl"`, and `unicode-bidi: plaintext` messages.

## Stabilization completed in this phase

- Every application SQLite connection uses a busy timeout and foreign-key enforcement.
- Schema initialization is backward-compatible and idempotent. `PRAGMA user_version=1` records the existing schema through an additive migration; no tables, rows, or vectors are rewritten.
- `database_status()` runs `integrity_check`, `foreign_key_check`, and reports schema/document/chunk counts.
- `backup_database()` uses SQLite's backup API and refuses to overwrite backups.
- `GET /api/health` reports non-sensitive local Ollama, embedding, and SQLite diagnostics.
- Rotating structured local logs record operational metadata only, never prompts, documents, embeddings, request bodies, or credentials.
- Synthetic tests cover legacy-database migration, backup restoration, failed Ollama connectivity, interrupted streams, invalid uploads, Persian normalization, archive limits, retrieval, and foreign keys.

## Prioritized remaining issues

| Priority | Issue | Recommended action |
|---|---|---|
| P0 | Code, HTML, SQLite, and Ollama logic remain in one file. | Extract compatibility-preserving modules one boundary at a time. |
| P0 | Interrupted streams no longer create a new chat, but partial-output audit state is not retained. | Add an additive request/message status migration and explicit local audit policy. |
| P1 | Prompt budgeting is character-based and `MAX_CONTEXT` is not enforced end-to-end. | Add a tested prompt builder with an explicit budget. |
| P1 | Embeddings are generated synchronously during ingestion. | Add local job state, bounded retry/cancellation, and model/dimension diagnostics without automatic re-indexing. |
| P1 | Lexical retrieval scans all records and uses substring matching. | Evaluate Persian tokenization and local FTS behind an additive migration. |
| P1 | PDF, DOCX, and XLSX extraction does not retain complete page/table/source spans. | Introduce a tested extractor interface and richer citations. |
| P1 | Document deletion is permanent once confirmed in the UI. | Design archive/restore and audit behavior before changing the endpoint. |
| P2 | The inline UI has limited accessibility, keyboard behavior, and error states. | Extract static assets while retaining Persian RTL and API contracts. |
| P2 | There is no authentication, role model, or retention policy. | Establish a deployment threat model before adding local access controls. |

## Target structure (future, not a migration instruction)

```text
app/
  config.py
  db/{connection.py,migrations.py,repositories.py}
  services/{chat.py,ollama.py,retrieval.py,ingestion.py,extraction.py}
  web/{server.py,handlers.py,static/}
  domain/{models.py,errors.py}
tests/{test_retrieval.py,test_ingestion.py,test_api.py,test_migrations.py}
scripts/{backup.py,verify_db.py}
```

Create compatibility wrappers around existing functions first, then move one endpoint or service at a time. The existing schema (`knowledge`, `chats`, `messages`, `documents`, and `doc_chunks`) is a data contract and must not be dropped or recreated.

## Phased plan

1. **Stabilize:** completed baseline migration, health diagnostics, safe backup helper, integrity checks, structured local logging, and regression coverage.
2. **Modularize:** database, configuration, Ollama-client, prompt-builder, chat-service, document-extractor, document-ingestion, retrieval, and citation boundaries are extracted. Next extract HTTP handlers and static UI assets without changing streaming or schema contracts.
3. **RAG diagnostics:** completed read-only embedding health, index coverage, model/dimension mismatch, and request-scoped fallback diagnostics with a Persian RTL dashboard. Next, record vector version through an approved additive migration, provide explicit opt-in re-indexing, create a Persian quality set, and preserve citation spans.
4. **Document lifecycle:** add ingestion jobs, failure state, preview, archive/restore, and testable extractor interfaces.
5. **Read-only SQL Server:** only after explicit approval; use external credentials, network allowlists, query templates, timeouts, row limits, and auditing. Never execute free-form SQL from a prompt.
6. **UI and operations:** improve Persian RTL accessibility, establish access control appropriate to the threat model, and add offline Windows packaging, backup, and retention operations.

## Remaining risks and decisions

- Exact approved Ollama/Gemma and embedding versions, as well as hardware capacity, remain unspecified. Automatic model changes and re-indexing remain prohibited.
- The first migration sets a version marker after ensuring the historical schema exists. Before any schema-changing migration, run and review `PRAGMA foreign_key_check` against a backed-up live database.
- The current CSP permits inline script/style because the UI remains embedded. Replace it with hashes or nonces when static assets are extracted.
- Localhost without authentication is appropriate only for a single-user workstation. Shared-workstation use requires an explicit threat model and authorization design.
- Scanned PDFs, advanced spreadsheets, and rich Word content are not fully extracted.
- Citations identify retrieved evidence but do not prove that a model response is grounded in it; a Persian groundedness evaluation set is still needed.
- Index monitoring reads the current records but does not validate vector normalization or repair invalid/mismatched vectors; remediation must remain explicit and opt-in.
