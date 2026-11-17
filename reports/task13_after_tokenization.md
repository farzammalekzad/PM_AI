# Synthetic Persian Retrieval Evaluation

Questions: 31; documents: 14.

| Mode | Status | Recall@3 | Recall@5 | MRR@5 | Precision@5 | Mean latency (ms) | Failures |
|---|---|---:|---:|---:|---:|---:|---:|
| lexical | evaluated | 1.000 | 1.000 | 0.938 | 0.174 | 0.285 | 1 |
| semantic | evaluated | 1.000 | 1.000 | 0.944 | 0.174 | 0.092 | 0 |
| hybrid | evaluated | 1.000 | 1.000 | 0.981 | 0.174 | 0.547 | 1 |

## Failure analysis

### lexical
- `unanswerable_question_returned_sources` — رمز عبور سامانه مالی چیست؟

### semantic
No failures at the evaluated cutoff.

### hybrid
- `unanswerable_question_returned_sources` — رمز عبور سامانه مالی چیست؟
