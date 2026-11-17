"""Synthetic-only Gemma experiment harness with deterministic and human review fields."""
import time

from app.evaluation.answer_evaluation import deterministic_checks
from app.evaluation.retrieval_evaluation import InMemoryRepository, materialize_embeddings
from app.services.citation_service import document_identifier
from app.services.prompt_builder import PromptBuilder
from app.services.retrieval_service import RetrievalService


DIRECT_MESSAGES = 'direct'
ORACLE_CONTEXT = 'oracle_context'
END_TO_END_RAG = 'end_to_end_rag'


def _stream_answer(client, model, messages, context_window):
    started = time.perf_counter()
    parts = []
    for event in client.stream_chat(model, messages, context_window):
        message = event.get('message') or {}
        content = message.get('content', '') if isinstance(message, dict) else ''
        if isinstance(content, str):
            parts.append(content)
    return ''.join(parts), round((time.perf_counter() - started) * 1000, 4)


def _documents(records, identifiers):
    wanted = set(identifiers)
    return [record for record in records if document_identifier(record) in wanted]


def _record(case, mode, answer, latency_ms, retrieved_ids, retrieval_success):
    checks = deterministic_checks(dict(case, retrieved_ids=retrieved_ids), answer)
    checks['citation_applicable'] = mode != DIRECT_MESSAGES
    return {
        'id': case['id'], 'question': case['question'], 'mode': mode, 'answer': answer,
        'latency_ms': latency_ms, 'retrieved_ids': retrieved_ids,
        'retrieval_success': retrieval_success, 'deterministic': checks,
    }


def run_gemma_experiments(client, model, cases, records, context_window=8192, embedding_model='',
                          evaluation_kind='real_local_gemma'):
    """Run direct, oracle, and lexical-RAG prompts through one explicit local client."""
    prompt_builder = PromptBuilder()
    if embedding_model:
        records = materialize_embeddings(records, client.embed, embedding_model)
        retriever = RetrievalService(lambda: InMemoryRepository(records), client.embed, embedding_model)
        retrieval_mode = 'hybrid_real_local_embedding'
    else:
        retriever = RetrievalService(lambda: InMemoryRepository(records), None, '')
        retrieval_mode = 'lexical'
    results = []
    for case in cases:
        expected_ids = list(case.get('relevant_ids', case.get('retrieved_ids', [])))
        answer, latency = _stream_answer(client, model, [{'role': 'user', 'content': case['question']}],
                                         context_window)
        results.append(_record(case, DIRECT_MESSAGES, answer, latency, [], None))

        oracle_ids = expected_ids
        oracle_prompt = prompt_builder.build(case['question'], [], _documents(records, oracle_ids), [],
                                             context_window)
        answer, latency = _stream_answer(client, model, oracle_prompt.messages, context_window)
        results.append(_record(case, ORACLE_CONTEXT, answer, latency, oracle_ids,
                               bool(oracle_ids) or not expected_ids))

        retrieval = retriever.retrieve_documents(case['question'], limit=5)
        retrieved_ids = [document_identifier(record) for record in retrieval.records]
        retrieval_success = (set(expected_ids) <= set(retrieved_ids)
                             if expected_ids else not retrieved_ids)
        rag_prompt = prompt_builder.build(case['question'], [], retrieval.records, [], context_window)
        answer, latency = _stream_answer(client, model, rag_prompt.messages, context_window)
        results.append(_record(case, END_TO_END_RAG, answer, latency, retrieved_ids,
                               retrieval_success))
    return grounding_report(results, model, embedding_model, retrieval_mode, evaluation_kind)


def grounding_report(results, model, embedding_model='', retrieval_mode='lexical',
                     evaluation_kind='real_local_gemma'):
    """Aggregate deterministic checks without representing them as factual verification."""
    experiments = []
    for mode in (DIRECT_MESSAGES, ORACLE_CONTEXT, END_TO_END_RAG):
        items = [item for item in results if item['mode'] == mode]
        answerable = [item for item in items if not item['deterministic']['expected_abstention']]
        unanswerable = [item for item in items if item['deterministic']['expected_abstention']]
        citation_items = [item for item in items if item['deterministic']['citation_applicable']]
        retrieval_items = [item for item in items if item['retrieval_success'] is not None]
        experiments.append({
            'mode': mode,
            'status': 'evaluated',
            'metrics': {
                'cases': len(items),
                'retrieval_success_rate': (sum(item['retrieval_success'] for item in retrieval_items) /
                                           max(1, len(retrieval_items))),
                'citation_validity_rate': (None if not citation_items else
                                            sum(item['deterministic']['citations_refer_to_retrieved_sources']
                                                for item in citation_items) / len(citation_items)),
                'abstention_rate': (sum(item['deterministic']['abstained'] for item in items) /
                                    max(1, len(items))),
                'unanswerable_false_answer_rate': (sum(not item['deterministic']['abstained']
                                                        for item in unanswerable) /
                                                  max(1, len(unanswerable))),
                'required_persian_wording_rate': (sum(
                    item['deterministic']['required_persian_wording_present'] for item in items) /
                    max(1, len(items))),
                'mean_response_latency_ms': (sum(item['latency_ms'] for item in items) /
                                             max(1, len(items))),
                'answerable_cases': len(answerable),
                'human_reviewed_grounding_cases': 0,
                'human_reviewed_answer_correctness_rate': None,
                'human_reviewed_factual_grounding_rate': None,
            },
            'cases': items,
        })
    return {
        'evaluation_kind': evaluation_kind, 'synthetic': True, 'chat_model': model,
        'embedding_model': embedding_model or None, 'retrieval_mode': retrieval_mode,
        'automatic_claim': ('Retrieval membership, citation syntax, abstention markers, and latency are '
                            'deterministic. Answer correctness and factual grounding require human review.'),
        'experiments': experiments,
    }


def unavailable_report(model, error_category):
    """Make local-model blocking state explicit without emitting substitute answers."""
    return {
        'evaluation_kind': 'real_local_gemma_unavailable', 'synthetic': True, 'chat_model': model,
        'error': error_category,
        'automatic_claim': 'No mocked or substitute answers were generated.',
        'experiments': [
            {'mode': mode, 'status': 'unavailable', 'metrics': None, 'cases': []}
            for mode in (DIRECT_MESSAGES, ORACLE_CONTEXT, END_TO_END_RAG)
        ],
    }
