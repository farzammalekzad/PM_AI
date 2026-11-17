# Task 14: Real Gemma Grounded-Answer Evaluation

This workflow evaluates only synthetic Persian documents and questions. It does not
read the company database, re-index documents, alter production rankings, or modify
the production system prompt.

## Experiment configuration

`scripts/evaluate_gemma4_grounding.py` has three distinct experiments over the same
four synthetic cases (direct answer, paraphrase, partial evidence, and unanswerable
password):

1. **Direct:** sends the question without retrieved documents or the production prompt.
2. **Oracle context:** builds the existing production prompt with the exact labeled
   synthetic source.
3. **End-to-end RAG:** materializes vectors only in memory for the synthetic corpus
   using the explicitly named local embedding model, runs the existing retriever, then
   sends the selected records through the existing prompt builder.

The required models are `qwen3-embedding-local:latest` and `gemma4:e4b-it-qat` at
`http://127.0.0.1:11434`. If either model is unavailable, all real experiment modes
are recorded as unavailable. The program never generates mocked substitute answers.

## Measurements and review boundary

For each mode, the report includes retrieval success, citation validity, abstention
rate, unanswerable false-answer rate, required Persian wording, and mean response
latency. Citation validity means only that cited source identifiers were in the passed
evidence. Human fields for answer correctness, groundedness, and unsupported facts
remain unfilled until a reviewer assesses each answer; automatic text or citation
checks do not establish factual correctness.

The password case specifically detects topical retrieval (`سامانه` or `مالی`) without
mistaking it for evidence. A suitable response abstains, adds no fabricated citation,
and does not state a password.

## Latest local attempt

The configured local endpoint returned `URLError`. Therefore direct Gemma, oracle
context, and end-to-end RAG have no real answers or metrics in
`reports/task14_real_gemma_grounding.json`. This is an availability result, not a
quality result.

## Command

```sh
python3 scripts/evaluate_gemma4_grounding.py \
  --ollama-url http://127.0.0.1:11434 \
  --embedding-model qwen3-embedding-local:latest \
  --chat-model gemma4:e4b-it-qat \
  --json reports/task14_real_gemma_grounding.json
```
