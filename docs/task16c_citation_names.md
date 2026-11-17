# Task 16C: Human-Readable RAG Citation Sources

## Root cause

Task 16 added `doc_source_details`, but the browser rendered every retrieved source as
soon as the initial NDJSON source event arrived. It could not know whether Gemma would
actually emit `[D#-C#]`, so names were not reliably associated with final citations and
the source block appeared above the answer.

## Implemented flow

Retrieval resolves chunk records through the existing `doc_chunks.document_id` to
`documents.id` relationship. Citation formatting preserves existing `D#-C#` syntax.
After a completed response, `ChatSession` emits an additive terminal
`cited_doc_sources` event. It contains only source details whose identifiers both occur
in the generated answer and belong to the retrieved document set. Each entry includes
the current document name and an optional persisted source location.

The frontend renders this list below the assistant bubble using `textContent`. Duplicate
citations appear once in first-occurrence order. Unknown, deleted, or non-retrieved
identifiers are omitted rather than assigned a guessed name. Renames are reflected
because the name originates from the retrieved record for the current request. Legacy
chunks remain compatible and show no location when provenance is unknown.

The terminal event is additive; existing source events, citation identifiers, NDJSON
content events, retrieval ranking, and prompts are unchanged. Citation membership does
not prove an answer is factually grounded.
