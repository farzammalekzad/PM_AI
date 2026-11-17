# Task 16: Provenance, Precise Citations, and Typography

## Previous limitations

Chunks previously preserved only a stable document ID/chunk number, text, and vector.
The source event exposed `D#-C# (filename)` but no page, heading, worksheet, or row
location. The prompt could therefore cite the chunk identifier but could not communicate
precise source location. Legacy chunks have no inferred location.

## Implemented provenance design

Migration 3 adds optional `doc_chunks.page_start`, `page_end`, `heading`, `worksheet`,
`row_start`, and `row_end`. It is additive and idempotent; old identifiers, text,
vectors, and retrieval formulas are unchanged. Null/empty values mean source location
is unknown rather than guessed.

New PDF extraction produces one source segment per actual text-extractable page. New
chunks receive the minimum/maximum actual intersected page, including chunks that span
pages. DOCX extraction tracks the most recent reliable Word heading style; it does not
claim physical Word pages. XLSX extraction records worksheet display name and actual
row number ranges. Plain text remains location-unknown.

Retrieval records carry this metadata. Existing `doc_sources` remains unchanged for
compatibility; the additive NDJSON field `doc_source_details` supplies identifier,
document name, and optional formatted Persian location. Prompt evidence also includes
the available location next to the existing `D#-C#` identifier. The RTL source display
uses only `textContent`.

Replacement deletes and inserts source metadata in the same transaction as chunks and
vectors, so old provenance cannot remain attached to a replacement chunk.

## Vazir font status

No Vazir WOFF2/WOFF/TTF/OTF asset exists in the repository, and no approved font asset
was supplied. Per the offline and licensing requirements, this task does not download,
copy, or substitute a font. The current Tahoma/Segoe UI stack and restrictive static
allowlist are intentionally unchanged. To complete this portion, provide an approved,
properly licensed Vazir WOFF2 file for `app/web/static/fonts/`; then add an explicit
allowlist entry, `font/woff2` MIME type, `@font-face` with `font-display: swap`,
same-origin `font-src`, and focused static/CSS tests.

## Legacy deletion endpoint security finding

The legacy `/api/document-delete` endpoint can be called directly without the UI's
explicit confirmation and therefore bypasses Task 15's UI confirmation. It remains for
local API compatibility. The smallest backward-compatible mitigation is to deprecate
that endpoint in documentation and introduce/retain `document-delete-confirmed` for all
UI use; removing or changing the legacy endpoint's contract requires explicit client
approval. Localhost-only deployment limits but does not eliminate this risk.

## Deferred capabilities

No OCR, Word physical page calculation, document preview/download, near-duplicate
analysis, font download, automatic re-indexing, or production-data operation is added.
