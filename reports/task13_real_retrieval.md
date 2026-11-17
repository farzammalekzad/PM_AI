# Synthetic Persian Retrieval Evaluation

Questions: 31; documents: 14.

| Mode | Status | Recall@3 | Recall@5 | MRR@5 | Precision@5 | Mean latency (ms) | Failures |
|---|---|---:|---:|---:|---:|---:|---:|
| lexical | evaluated | 1.000 | 1.000 | 0.938 | 0.174 | 0.252 | 1 |
| semantic | unavailable (embedding_model_unavailable) | — | — | — | — | — | — |
| hybrid | unavailable (embedding_model_unavailable) | — | — | — | — | — | — |

## Failure analysis

### lexical
- `unanswerable_question_returned_sources` — رمز عبور سامانه مالی چیست؟
