# Task 16G: Answer Mode Isolation

## Confirmed root cause

Task 16F sent the selected `answer_mode` to the server, but `ChatService.start()` always called organizational knowledge and document retrieval before mode handling. `PromptBuilder` always started with the organizational system prompt and evidence header, and it always received retrieved sources. It also replayed every recent assistant message, including prior document-grounded answers. As a result, General mode was only a wording variation over organizational context.

## Isolation behavior

| Mode | Organizational retrieval | Evidence sent to Ollama | Document source events |
| --- | --- | --- | --- |
| General | No | No | No |
| Grounded | Yes | Retrieved evidence | Yes, for valid emitted citations |
| Hybrid | Yes | Retrieved evidence | Yes, for valid emitted citations |

General uses a separate non-organizational system prompt and never calls either knowledge or document retrieval. It sends no organizational evidence header, `K#`/`D#-C#` evidence items, retrieval diagnostics, or `cited_doc_sources` terminal event. Its terminal citation-integrity status is retained only as a non-sensitive mode diagnostic.

Grounded and Hybrid preserve the existing retrieval pipeline, ranking, thresholds, identifiers, prompt budgeting, and citation validation. Grounded retains deterministic abstention when no document is retrieved for a substantive question. Hybrid requires document citations for claims attributed to company documents and prompts the model to identify general knowledge as general.

## Conversation history

Schema version 4 adds the additive `message_modes` table. New user/assistant message pairs are tagged with their selected mode while their original message text and existing identifiers remain unchanged. A General-mode request includes only previously tagged General messages. This prevents new Grounded/Hybrid content—including uploaded attachment text stored in user turns—from becoming General context.

Legacy messages have no mode metadata and remain readable in chat history, but are omitted from General-mode prompt history because their origin cannot be determined safely. Grounded and Hybrid retain the established recent-history behavior. This is a deliberate privacy and correctness trade-off; no historical content is rewritten or deleted.

## Diagnostics and limits

The additive source event now carries `mode_diagnostics`: selected mode, whether organizational retrieval ran, the number of retrieved organizational sources, and whether organizational evidence was sent. The RTL monitoring panel displays these values and the final citation status without exposing document contents.

The tests use mocked Ollama streams and synthetic documents. They prove context routing and citation membership, not factual correctness or universal model compliance when document evidence is present.
