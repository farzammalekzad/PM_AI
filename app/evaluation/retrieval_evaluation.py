"""Reproducible, read-only retrieval quality evaluation utilities."""
import json
import math
import statistics
import time

from app.services.citation_service import document_identifier
from app.services.retrieval_service import (
    RetrievalService, cosine_similarity, normalize_persian, tokenize_persian,
)


def metric_summary(results):
    """Calculate macro retrieval metrics; unanswerable cases are excluded from recall/MRR."""
    answerable = [result for result in results if result['relevant_ids']]

    def recall_at(limit):
        return sum(len(set(result['retrieved_ids'][:limit]) & set(result['relevant_ids'])) /
                   len(result['relevant_ids']) for result in answerable) / max(1, len(answerable))

    def reciprocal_rank_at_five(result):
        relevant = set(result['relevant_ids'])
        for rank, identifier in enumerate(result['retrieved_ids'][:5], 1):
            if identifier in relevant:
                return 1 / rank
        return 0.0

    precision_at_five = sum(
        len(set(result['retrieved_ids'][:5]) & set(result['relevant_ids'])) / 5
        for result in results) / max(1, len(results))
    latencies = [result['latency_ms'] for result in results]
    unanswerable_false_positives = sum(
        1 for result in results if not result['relevant_ids'] and result['retrieved_ids'])
    return {
        'recall_at_3': recall_at(3),
        'recall_at_5': recall_at(5),
        'mrr_at_5': sum(reciprocal_rank_at_five(item) for item in answerable) / max(1, len(answerable)),
        'precision_at_5': precision_at_five,
        'mean_latency_ms': statistics.fmean(latencies) if latencies else 0.0,
        'p95_latency_ms': sorted(latencies)[max(0, math.ceil(len(latencies) * .95) - 1)] if latencies else 0.0,
        'evaluated_questions': len(results),
        'answerable_questions': len(answerable),
        'unanswerable_questions': len(results) - len(answerable),
        'unanswerable_false_positives': unanswerable_false_positives,
    }


def failure_analysis(case, retrieved):
    relevant = set(case['relevant_ids'])
    missing = sorted(relevant - set(retrieved))
    if not relevant:
        return None if not retrieved else 'unanswerable_question_returned_sources'
    if not missing:
        return None
    return 'relevant_chunk_ranked_below_top_5' if relevant & set(retrieved) else 'relevant_chunk_not_retrieved'


def retrieval_diagnostics(records, question, embedder=None, embedding_model=''):
    """Expose production-compatible candidate scores for one synthetic query.

    This is an evaluator-only trace.  It deliberately mirrors, rather than alters,
    ``RetrievalService.retrieve_documents`` so reports can explain exclusions.
    """
    words = tokenize_persian(question)
    query_vector = embedder(question, embedding_model) if embedder and embedding_model else None
    candidates = []
    for position, record in enumerate(records):
        body = normalize_persian(record['content'])
        title = normalize_persian(record['name'])
        title_hits = sorted(word for word in words if word in title)
        body_hits = sorted(word for word in words if word in body)
        hits = 2 * len(title_hits) + len(body_hits)
        lexical = hits / max(1, len(words))
        semantic = None
        if (query_vector is not None and record['embed_model'] == embedding_model
                and record['vector']):
            vector = json.loads(record['vector'])
            if len(vector) == len(query_vector):
                semantic = cosine_similarity(query_vector, vector)
        if semantic is None:
            included = bool(hits)
            hybrid = lexical if included else None
            exclusion = None if included else 'no_lexical_hit_without_semantic_vector'
        else:
            included = semantic >= 0.38 or bool(hits)
            hybrid = 1.8 * semantic + 0.5 * min(lexical, 3) if included else None
            exclusion = None if included else 'semantic_below_0.38_and_no_lexical_hit'
        candidates.append({
            'identifier': document_identifier(record), 'position': position,
            'lexical_score': round(lexical, 7),
            'cosine_similarity': None if semantic is None else round(semantic, 7),
            'hybrid_score': None if hybrid is None else round(hybrid, 7),
            'title_hits': title_hits, 'body_hits': body_hits,
            'included': included, 'exclusion_reason': exclusion,
        })
    ranked = sorted((item for item in candidates if item['included']),
                    key=lambda item: (-(item['hybrid_score'] or 0), item['position']))
    ranks = {item['identifier']: index for index, item in enumerate(ranked, 1)}
    for item in candidates:
        item['rank'] = ranks.get(item['identifier'])
        if item['included'] and item['rank'] > 5:
            item['exclusion_reason'] = 'ranked_below_top_5'
    return {
        'query_terms': sorted(words),
        'threshold': {'semantic_minimum_without_lexical_hit': 0.38},
        'top_five': [item['identifier'] for item in ranked[:5]],
        'candidates': candidates,
    }


class InMemoryRepository:
    """Minimal read-only repository adapter for synthetic records."""

    def __init__(self, records):
        self.records = records

    def document_records(self):
        return self.records


def lexical_retriever(records):
    service = RetrievalService(lambda: InMemoryRepository(records), None, '')
    return lambda question: service.retrieve_documents(question, limit=5).records


def hybrid_retriever(records, embedder, embedding_model):
    service = RetrievalService(lambda: InMemoryRepository(records), embedder, embedding_model)
    return lambda question: service.retrieve_documents(question, limit=5).records


def materialize_embeddings(records, embedder, embedding_model):
    """Build synthetic evaluation records with vectors from one explicit model."""
    embedded = []
    for record in records:
        vector = embedder(record['content'], embedding_model)
        if not isinstance(vector, list) or not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError('invalid_evaluation_embedding')
        embedded.append(dict(record, embed_model=embedding_model, vector=json.dumps(vector)))
    return embedded


def semantic_retriever(records, embedder, embedding_model):
    """Evaluate semantic-only ranking with the production cosine threshold, not a fallback."""
    def retrieve(question):
        query_vector = embedder(question, embedding_model)
        ranked = []
        for position, record in enumerate(records):
            if record['embed_model'] != embedding_model or not record['vector']:
                continue
            vector = json.loads(record['vector'])
            if len(vector) != len(query_vector):
                continue
            score = cosine_similarity(query_vector, vector)
            if score >= 0.38:
                ranked.append((score, position, record))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [record for _, _, record in ranked[:5]]
    return retrieve


def evaluate_mode(name, cases, retrieve=None, unavailable_reason=None):
    """Return an explicit unavailable result or metrics from actual retrieval calls."""
    if retrieve is None:
        return {'mode': name, 'status': 'unavailable', 'reason': unavailable_reason or 'not_configured',
                'metrics': None, 'cases': [], 'failures': []}
    results = []
    failures = []
    for case in cases:
        started = time.perf_counter()
        records = retrieve(case['question'])
        latency = (time.perf_counter() - started) * 1000
        retrieved = [document_identifier(record) for record in records]
        item = {
            'question': case['question'], 'category': case['category'],
            'relevant_ids': case['relevant_ids'], 'retrieved_ids': retrieved,
            'latency_ms': round(latency, 4),
        }
        analysis = failure_analysis(case, retrieved)
        if analysis:
            failures.append({**item, 'reason': analysis})
        results.append(item)
    return {'mode': name, 'status': 'evaluated', 'reason': None,
            'metrics': metric_summary(results), 'cases': results, 'failures': failures}


def evaluate_dataset(records, cases, embedder=None, embedding_model=''):
    """Evaluate lexical always; semantic/hybrid only when explicitly available."""
    report = {
        'dataset': {'documents': len(records), 'questions': len(cases),
                    'synthetic': True, 'embedding_model': embedding_model or None},
        'modes': [evaluate_mode('lexical', cases, lexical_retriever(records))],
    }
    if not embedder or not embedding_model:
        reason = 'embedding_model_unavailable'
        report['modes'].append(evaluate_mode('semantic', cases, unavailable_reason=reason))
        report['modes'].append(evaluate_mode('hybrid', cases, unavailable_reason=reason))
        return report
    report['modes'].append(evaluate_mode(
        'semantic', cases, semantic_retriever(records, embedder, embedding_model)))
    report['modes'].append(evaluate_mode(
        'hybrid', cases, hybrid_retriever(records, embedder, embedding_model)))
    return report


def markdown_summary(report):
    """Render a concise operator-facing report without document contents."""
    lines = ['# Synthetic Persian Retrieval Evaluation', '',
             f"Questions: {report['dataset']['questions']}; documents: {report['dataset']['documents']}.", '',
             '| Mode | Status | Recall@3 | Recall@5 | MRR@5 | Precision@5 | Mean latency (ms) | Failures |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for mode in report['modes']:
        if mode['status'] != 'evaluated':
            lines.append(f"| {mode['mode']} | unavailable ({mode['reason']}) | — | — | — | — | — | — |")
            continue
        metrics = mode['metrics']
        lines.append(
            f"| {mode['mode']} | evaluated | {metrics['recall_at_3']:.3f} | "
            f"{metrics['recall_at_5']:.3f} | {metrics['mrr_at_5']:.3f} | "
            f"{metrics['precision_at_5']:.3f} | {metrics['mean_latency_ms']:.3f} | "
            f"{len(mode['failures'])} |")
    lines.extend(['', '## Failure analysis', ''])
    for mode in report['modes']:
        if mode['status'] == 'evaluated':
            lines.append(f"### {mode['mode']}")
            if not mode['failures']:
                lines.append('No failures at the evaluated cutoff.')
            else:
                for failure in mode['failures']:
                    lines.append(f"- `{failure['reason']}` — {failure['question']}")
            lines.append('')
    return '\n'.join(lines).rstrip()
