# Synthetic Persian Retrieval Evaluation

Questions: 31; documents: 14.

| Mode | Status | Recall@3 | Recall@5 | MRR@5 | Precision@5 | Mean latency (ms) | Failures |
|---|---|---:|---:|---:|---:|---:|---:|
| lexical | evaluated | 0.963 | 0.963 | 0.926 | 0.168 | 0.238 | 2 |
| semantic | evaluated | 1.000 | 1.000 | 0.944 | 0.174 | 0.094 | 0 |
| hybrid | evaluated | 1.000 | 1.000 | 0.981 | 0.174 | 0.316 | 1 |

## Failure analysis

### lexical
- `relevant_chunk_not_retrieved` — چه کسی از دیرکرد فعالیت‌ها مطلع می‌شود؟
- `unanswerable_question_returned_sources` — رمز عبور سامانه مالی چیست؟

### semantic
No failures at the evaluated cutoff.

### hybrid
- `unanswerable_question_returned_sources` — رمز عبور سامانه مالی چیست؟
