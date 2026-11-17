# Synthetic Persian Retrieval Evaluation

Questions: 31; documents: 14.

| Mode | Status | Recall@3 | Recall@5 | MRR@5 | Precision@5 | Mean latency (ms) | Failures |
|---|---|---:|---:|---:|---:|---:|---:|
| lexical | evaluated | 0.963 | 0.963 | 0.926 | 0.168 | 0.259 | 2 |
| semantic | unavailable (embedding_model_unavailable) | — | — | — | — | — | — |
| hybrid | unavailable (embedding_model_unavailable) | — | — | — | — | — | — |

## Failure analysis

### lexical
- `relevant_chunk_not_retrieved` — چه کسی از دیرکرد فعالیت‌ها مطلع می‌شود؟
- `unanswerable_question_returned_sources` — رمز عبور سامانه مالی چیست؟
