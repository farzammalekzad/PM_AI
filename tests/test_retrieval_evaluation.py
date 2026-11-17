"""Synthetic tests for offline Persian retrieval evaluation metrics."""
import unittest

from app.evaluation.persian_retrieval_dataset import SYNTHETIC_CASES, SYNTHETIC_DOCUMENTS, synthetic_embedder
from app.evaluation.retrieval_evaluation import (
    evaluate_dataset, failure_analysis, markdown_summary, materialize_embeddings, metric_summary,
    retrieval_diagnostics,
)


class RetrievalEvaluationTests(unittest.TestCase):
    def test_metric_definitions_are_mathematically_stable(self):
        results = [
            {'relevant_ids': ['D1-C1', 'D1-C2'], 'retrieved_ids': ['D1-C1'], 'latency_ms': 1.0},
            {'relevant_ids': ['D2-C1'], 'retrieved_ids': ['D9-C1', 'D2-C1'], 'latency_ms': 3.0},
            {'relevant_ids': [], 'retrieved_ids': [], 'latency_ms': 2.0},
        ]
        metrics = metric_summary(results)
        self.assertEqual(metrics['recall_at_3'], 0.75)
        self.assertEqual(metrics['recall_at_5'], 0.75)
        self.assertEqual(metrics['mrr_at_5'], 0.75)
        self.assertAlmostEqual(metrics['precision_at_5'], 2 / 15)
        self.assertEqual(metrics['mean_latency_ms'], 2.0)
        self.assertEqual(metrics['p95_latency_ms'], 3.0)
        self.assertEqual(metrics['unanswerable_false_positives'], 0)

    def test_synthetic_dataset_has_required_scope_and_valid_labels(self):
        self.assertGreaterEqual(len(SYNTHETIC_CASES), 30)
        categories = {item['category'] for item in SYNTHETIC_CASES}
        self.assertTrue({'project_management', 'organizational_procedures', 'meeting_minutes',
                         'contracts', 'project_delays', 'progress_reporting',
                         'persian_arabic_variant', 'semantic_paraphrase', 'unanswerable'} <= categories)
        identifiers = {f"D{item['document_id']}-C{item['chunk_no']}"
                       for item in SYNTHETIC_DOCUMENTS}
        for item in SYNTHETIC_CASES:
            self.assertTrue(set(item['relevant_ids']) <= identifiers)

    def test_unavailable_modes_are_never_substituted_with_lexical_results(self):
        report = evaluate_dataset(SYNTHETIC_DOCUMENTS, SYNTHETIC_CASES)
        lexical, semantic, hybrid = report['modes']
        self.assertEqual(lexical['status'], 'evaluated')
        self.assertEqual(semantic['status'], 'unavailable')
        self.assertEqual(hybrid['status'], 'unavailable')
        self.assertEqual(semantic['reason'], 'embedding_model_unavailable')
        self.assertEqual(semantic['cases'], [])
        self.assertIn('unavailable (embedding_model_unavailable)', markdown_summary(report))

    def test_mocked_embeddings_evaluate_semantic_and_hybrid_without_changing_retrieval(self):
        report = evaluate_dataset(
            SYNTHETIC_DOCUMENTS, SYNTHETIC_CASES, synthetic_embedder,
            'synthetic-persian-evaluation')
        modes = {item['mode']: item for item in report['modes']}
        self.assertEqual(modes['semantic']['status'], 'evaluated')
        self.assertEqual(modes['hybrid']['status'], 'evaluated')
        self.assertEqual(modes['semantic']['metrics']['evaluated_questions'], len(SYNTHETIC_CASES))
        self.assertEqual(modes['semantic']['metrics']['recall_at_5'], 1.0)
        self.assertEqual(modes['hybrid']['metrics']['recall_at_5'], 1.0)
        self.assertEqual(failure_analysis({'relevant_ids': ['D1-C1']}, []),
                         'relevant_chunk_not_retrieved')
        self.assertEqual(failure_analysis({'relevant_ids': []}, ['D1-C1']),
                         'unanswerable_question_returned_sources')

    def test_materialized_embeddings_replace_synthetic_metadata_and_dimensions(self):
        records = materialize_embeddings(
            SYNTHETIC_DOCUMENTS[:1], lambda text, model: [1.0, 2.0], 'real-local-model')
        self.assertEqual(records[0]['embed_model'], 'real-local-model')
        self.assertEqual(records[0]['vector'], '[1.0, 2.0]')
        with self.assertRaises(ValueError):
            materialize_embeddings(SYNTHETIC_DOCUMENTS[:1], lambda text, model: [float('nan')], 'model')

    def test_failure_diagnostics_explain_expected_delay_chunk_exclusion(self):
        trace = retrieval_diagnostics(
            SYNTHETIC_DOCUMENTS, 'چه کسی از دیرکرد فعالیت‌ها مطلع می‌شود؟')
        expected = next(item for item in trace['candidates'] if item['identifier'] == 'D1-C2')
        lexical_match = next(item for item in trace['candidates'] if item['identifier'] == 'D1-C1')
        self.assertEqual(trace['top_five'], ['D1-C1'])
        self.assertEqual(expected['lexical_score'], 0.0)
        self.assertEqual(expected['exclusion_reason'], 'no_lexical_hit_without_semantic_vector')
        self.assertEqual(lexical_match['body_hits'], ['فعالیت'])


if __name__ == '__main__':
    unittest.main()
