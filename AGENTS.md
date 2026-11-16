# Contributor Guide

## Non-negotiable rules

- Keep the application offline. Its default host is `127.0.0.1` only.
- Preserve Gemma/Ollama compatibility, Persian RTL behavior, and existing SQLite data.
- Never delete, reset, overwrite, or commit `ollama_company_data/`, SQLite files, WAL files, or SHM files.
- Do not add dependencies, Docker, external APIs, embedding models, or SQL Server connectivity without an explicit decision.
- Use parameterized SQL only. Any future SQL Server access must be read-only, least-privilege, and restricted to an allowlist of query templates.
- Preserve existing HTTP API contracts and the current SQLite tables unless an additive, versioned migration is reviewed and tested.

## Language and localization policy

- All agent communication, progress reports, technical documentation, code comments, commit messages, test descriptions, and operational logs must be in English.
- Persian is the primary application UI language. Do not translate existing Persian UI labels, organizational knowledge, database content, or user-facing messages unless explicitly requested.
- Preserve `dir="rtl"` for the page, `unicode-bidi: plaintext` for messages, and `dir="auto"` support for Latin text and identifiers.

## Change process

1. Inspect Git status and existing migrations before changing code.
2. Keep changes small and reviewable. Schema changes must be additive, idempotent, versioned with `PRAGMA user_version`, and non-destructive.
3. Add synthetic tests for new behavior. Never use organizational data in tests.
4. Run `python -m unittest discover -s tests -v` and `python -m py_compile ollama_persian_rag.py`.
5. Before each independent commit, run `git diff --check` and inspect `git status --short`.

## Technical requirements

- Every SQLite connection must enable `foreign_keys=ON` and a busy timeout.
- Validate and limit network input, base64, ZIP/OOXML archives, text sizes, and Ollama failures.
- Insert user-controlled UI text with `textContent`; never use `innerHTML` for it.
- Do not log prompts, document contents, embeddings, credentials, authorization values, or raw request bodies.
