# Task 13: Persian Retrieval Normalization and Safe Abstention

All evaluation data is synthetic. No company document, SQLite record, embedding, or
index was accessed or modified.

## Confirmed root cause and tokenizer change

The old token pattern included the whole Arabic/Persian Unicode block, including the
Persian question mark. Therefore `می‌شود؟` yielded `شود؟`, which could not match
`شود` in a chunk. The production tokenizer now recognizes word characters only,
normalizes the established Arabic/Persian `ي`/`ى`/`ك` variants, removes tatweel and
diacritics, and treats zero-width spacing characters as boundaries. It changes neither
stored text nor document identifiers.

## Measured synthetic retrieval results

The pre-change mocked-embedding baseline had lexical Recall@3/Recall@5/MRR@5/
Precision@5 of `0.963/0.963/0.926/0.168`, with one unanswerable false positive.
After the tokenizer change they are `1.000/1.000/0.938/0.174`, also with one
unanswerable false positive. The lexical failure count drops from two to one: the
remaining failure is the password question. Timings are recorded in generated reports
but are not compared as performance claims because this small local benchmark is noisy.

The opt-in synonym experiment uses only these reviewed groups:

- `دیرکرد`, `تاخیر`, `تأخیر`
- `صورتجلسه`, `صورت‌جلسه`
- `تایید`, `تأیید`

It is evaluation-only and does not call production retrieval. Its measured lexical
Recall@3/Recall@5/MRR@5/Precision@5 is `1.000/1.000/0.957/0.174`, with one
unanswerable false positive. The MRR gain requires review before any production use.

## Unanswerable questions and safe abstention

For `رمز عبور سامانه مالی چیست؟`, `D5-C1` is retrieved through the weak lexical term
`سامانه` and `D7-C1` through `مالی`. These are not password evidence. The current
prompt instructs Gemma to say that evidence is insufficient and not fabricate
citations. Deterministic evaluation can validate that a citation identifier was
retrieved, but factual support and unsupported claims remain explicit human-review
questions. A possible future safeguard is a prompt-level clarification that topical
retrieval is not sufficient evidence; it must be evaluated and approved before any
production prompt modification.

## Commands

```sh
python3 scripts/evaluate_persian_retrieval.py --json reports/task13_after_tokenization.json --markdown reports/task13_after_tokenization.md
python3 scripts/evaluate_persian_synonyms.py
python3 scripts/evaluate_persian_answers.py --real-ollama --ollama-url http://127.0.0.1:11434 --chat-model gemma4:e4b-it-qat --json reports/task13_real_answers.json
```

The real-model commands remain explicit. If the local service is unavailable, their
reports use the exact error category and do not substitute mocked results.
