# Task 16F: Reliable Document-Grounded Answer Mode

## Confirmed cause

Document retrieval, evidence construction, and `D#-C#` source metadata already ran for document-related requests. The former single system instruction asked the model to cite sources but did not make citations a response-mode requirement. When Gemma omitted an identifier, the terminal `cited_doc_sources` event correctly returned an empty list, but the interface had no integrity status and therefore made an uncited response look complete. This is generation behavior and missing post-generation detection, not proof of a retrieval or citation-recognition failure.

## Additive answer modes

`answer_mode` is an additive `/api/chat-stream` request field. The Persian interface explicitly sends and defaults to `grounded`. Task 16G later isolated General mode and retained the pre-mode retrieval behavior for callers that omit the field; clients that require a specific mode must send it explicitly.

- `grounded`: retrieved company documents are the authoritative evidence for document claims. The prompt requires exact retrieved `D#-C#` identifiers. If no document is retrieved for a substantive question, the chat service returns a deterministic Persian evidence-insufficient abstention without calling the model. Exact short greetings remain ordinary conversations.
- `hybrid`: document-backed statements require document identifiers; general knowledge must be described as general rather than presented as company-document evidence.
- `general`: normal conversation is allowed. Real citations may still be used, but they are not required.

The browser selector defaults to the Persian Grounded label. The selected value is sent only with the existing local streaming request and does not change retrieval ranking, thresholds, documents, embeddings, schema, or model selection.

## Citation integrity

The existing initial source event remains the retrieved-source metadata. The existing terminal `cited_doc_sources` event remains the subset of identifiers actually emitted by the model and known to the retrieved set. A new additive terminal `citation_integrity` event reports only citation membership status and count. It deliberately does not claim factual correctness or attach uncited sources to a response.

Grounded answers with retrieved documents but no valid `D#-C#` citation receive `missing_citation`. Answers without retrieved documents receive `insufficient_evidence`. The UI shows a visible Persian warning for both cases; valid citation status explicitly states that automatic citation checking does not prove the response facts are correct.

## Scope and limits

Tests use synthetic documents and mocked model streams. They verify modes, default compatibility, source instruction independence, abstention, citation membership, UI event handling, and safe DOM rendering. They do not establish that a real model always follows the prompt when retrieved documents exist; uncited output is instead surfaced visibly for review. No real-model evaluation is part of the standard suite.
