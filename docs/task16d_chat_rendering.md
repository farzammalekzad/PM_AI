# Task 16D: Persian RTL Chat Rendering

Assistant responses are stored and streamed as their original model text. Markdown is a browser-only presentation transformation performed after the terminal `done` event, so no API field, chat-history value, or NDJSON event changes.

During streaming, response fragments remain plain text through `textContent` for prompt visibility and to tolerate incomplete Markdown. Once complete, the local renderer safely builds paragraphs, headings, ordered and unordered lists, bold and italic text, inline code, fenced code blocks, and line breaks with `createElement`, `createTextNode`, and `textContent`. It does not parse HTML or support Markdown links, so model-provided HTML and unsafe link syntax stay literal text.

`[D#-C#]` identifiers are isolated LTR, non-wrapping citation spans. The existing terminal `cited_doc_sources` event remains the authoritative source list: it is emitted only for identifiers both generated in the answer and retrieved for that turn. The UI displays each such source once with its current document name and only persisted provenance locations.

The source section continues to use the local Vazir font through inherited application typography. Inline and fenced code, citations, and Thinking output retain monospaced fonts where appropriate. No font, network, CSP, static-file, database, retrieval, prompt, or model behavior changes are part of this task.

The browser-renderer contract test uses synthetic content only and verifies safe DOM construction, Markdown structure, RTL/LTR citation presentation, duplicate source suppression, known location display, and literal treatment of HTML-like model output. The test runs with locally installed Node.js when available; the Python suite also verifies the static safety and CSP/font contracts.
