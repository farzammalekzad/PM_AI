# Task 12: Synthetic Persian RAG Failure Analysis

This analysis is synthetic-only and does not read the company database or documents.

## Retrieval findings

For `چه کسی از دیرکرد فعالیت‌ها مطلع می‌شود؟`, the expected source is `D1-C2`.
The lexical tokenizer treats the Persian question mark as part of `شود؟` because its
character class includes the Persian Unicode block. The remaining meaningful exact
match is `فعالیت`, which occurs in `D1-C1`; `دیرکرد` does not equal the source term
`تاخیر`. Consequently lexical mode returns only `D1-C1` (lexical score `0.2`) and
excludes `D1-C2` (score `0.0`). In the deterministic mocked embedding diagnostic,
semantic mode returns `D1-C2` with cosine `1.0`, and hybrid ranks `D1-C2` first
(hybrid `1.8`) ahead of `D1-C1` (hybrid `0.1`). These are mocked-embedding diagnostic
results, not real-model measurements.

For `رمز عبور سامانه مالی چیست؟`, lexical and hybrid return `D5-C1` because of
`سامانه` and `D7-C1` because of `مالی`; each has lexical score `0.2`. Neither is
password evidence. This is a retrieval false positive, but retrieval alone does not
make either source valid answer evidence. The existing system prompt requires an
explicit insufficient-evidence response and prohibits fabricated citations.

## Grounded-answer workflow

`scripts/evaluate_persian_answers.py` evaluates four synthetic cases: direct answer,
paraphrase, partial evidence, and unanswerable password. It deterministically checks
whether citations use retrieved identifiers, whether expected abstention markers are
present, and whether required Persian phrases remain present. It creates a separate
human-review record for evidence support and unsupported factual claims. Automatic
similarity or citation validity is never treated as factual-grounding proof.

Use the following commands:

```sh
python3 scripts/diagnose_persian_rag_failures.py
python3 scripts/evaluate_persian_answers.py --json reports/task12_mocked_answers.json
python3 scripts/evaluate_persian_answers.py --real-ollama --ollama-url http://127.0.0.1:11434 --chat-model gemma4:e4b-it-qat --json reports/task12_real_answers.json
```

## Proposed corrective actions requiring approval

1. Review lexical token boundaries so Persian punctuation is not retained as part of a term.
2. Evaluate a Persian synonym/normalization strategy for `دیرکرد` and `تاخیر` on a reviewed quality set.
3. Consider a prompt clarification that retrieved material can be topically related but insufficient; do not alter it without answer-quality evidence.

No retrieval formula, threshold, embedding model, schema, API contract, or user interface was changed.
