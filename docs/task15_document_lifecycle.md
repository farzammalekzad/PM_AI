# Task 15: Document Lifecycle Management

This work uses only temporary SQLite databases and synthetic documents in tests. It
does not inspect, migrate manually, re-index, delete, or reset the production data
directory.

## Previous behavior

Documents were stored in `documents` with an integer ID, original filename, raw-upload
SHA-256 digest, extracted character count, chunk count, embedding model, and creation
time. `doc_chunks` stored the document ID, chunk number, extracted text, and vector.
The existing upload path already rejected an identical raw-upload digest, regardless
of filename. The documents endpoint listed only a small metadata subset; the UI had a
single delete button with browser-side confirmation. Deletion removed chunks and then
the document, and no replacement or metadata update path existed.

## Implemented lifecycle behavior

- `GET /api/documents` now returns document ID, name, SHA-256, character/chunk counts,
  embedding model, number of vector-bearing chunks, creation/update times, and status.
- `POST /api/document-metadata` changes only the sanitized document name. It never
  changes extracted content, hash, chunks, or vectors.
- Exact duplicate upload continues to use a stable SHA-256 of raw upload bytes and
  returns the existing document identifier in the Persian validation message. A same
  filename with different bytes remains a separate document.
- `POST /api/document-replace` requires the selected document ID and `confirmed:true`.
  Extraction, chunking, hash calculation, and requested embeddings all complete before
  the final transaction. The final transaction preserves the document ID, replaces all
  chunks/vectors, and rejects a digest already owned by a different document. Any
  extraction or embedding failure leaves the original document untouched.
- `POST /api/document-delete-confirmed` requires explicit confirmation. The older
  `/api/document-delete` endpoint remains for local API compatibility; the UI uses the
  confirmed endpoint and shows an explicit warning.

## Migration

Migration `2` is additive and idempotent. It adds `documents.status` (default
`active`) and `documents.updated_at`, filling the latter from `created_at` for existing
records. It changes no chunk text, vectors, IDs, hashes, or constraints. Schema version
is now `2`.

## Interface

The existing RTL document panel now shows metadata, vector availability, status, and
the SHA-256 value using `textContent`. It provides rename, explicit replacement
selection/confirmation, and a clearer deletion warning. It adds no external assets,
frameworks, or CSP exceptions.

## Limitations and deferred work

Duplicate detection is exact raw-byte matching only. It intentionally does not claim
near-duplicate detection for files with different formatting or byte encodings. There
is no bulk deletion, archive/restore, document version history, or automatic re-index
of unrelated records. Those require future policy and review.
