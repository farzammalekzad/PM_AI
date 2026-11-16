"""Synthetic regression tests for retrieval, diagnostics, and citation formatting."""
import json
import unittest

from app.services.citation_service import evidence_items, source_metadata
from app.services.retrieval_service import RetrievalService, cosine_similarity
from app.services.retrieval_service import normalize_persian, tokenize_persian


class FakeRepository:
    def __init__(self, knowledge=None, documents=None):
        self.knowledge = knowledge or []
        self.documents = documents or []

    def approved_knowledge(self):
        return self.knowledge

    def document_records(self):
        return self.documents


class RetrievalServiceTests(unittest.TestCase):
    def test_persian_normalization_tokenization_and_knowledge_ordering(self):
        self.assertEqual(normalize_persian(' كِتاب\u200c ي '), 'کتاب ی')
        self.assertIn('بودجه', tokenize_persian('بودجه شرکت چگونه است'))
        repository = FakeRepository(knowledge=[
            {'id': 2, 'title': 'بودجه', 'category': 'عمومی', 'content': 'جزئیات', 'source': ''},
            {'id': 1, 'title': 'بودجه', 'category': 'عمومی', 'content': 'جزئیات', 'source': ''},
            {'id': 3, 'title': 'نامرتبط', 'category': 'عمومی', 'content': 'بودجه', 'source': ''},
        ])
        service = RetrievalService(lambda: repository, None, '')
        self.assertEqual([item['id'] for item in service.search_knowledge('بودجه')], [1, 2, 3])

    def test_semantic_hybrid_ranking_and_top_k(self):
        records = [
            {'document_id': 1, 'chunk_no': 1, 'content': 'نامرتبط', 'vector': json.dumps([1.0, 0.0]),
             'name': 'one.txt', 'embed_model': 'synthetic'},
            {'document_id': 2, 'chunk_no': 1, 'content': 'بودجه', 'vector': json.dumps([0.8, 0.6]),
             'name': 'budget.txt', 'embed_model': 'synthetic'},
            {'document_id': 3, 'chunk_no': 1, 'content': 'نامرتبط', 'vector': json.dumps([0.1, 0.0]),
             'name': 'three.txt', 'embed_model': 'synthetic'},
        ]
        service = RetrievalService(lambda: FakeRepository(documents=records),
                                   lambda query, model: [1.0, 0.0], 'synthetic')
        result = service.retrieve_documents('بودجه', limit=2)
        self.assertEqual([row['document_id'] for row in result.records], [2, 1])
        self.assertEqual(result.diagnostics.mode, 'hybrid')
        self.assertTrue(result.diagnostics.semantic_attempted)
        self.assertTrue(result.diagnostics.embedding_succeeded)
        self.assertEqual(result.diagnostics.selected_source_identifiers, ('D2-C1', 'D1-C1'))
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.8, 0.6]), 0.8)

    def test_embedding_failure_and_model_mismatch_keep_lexical_fallback(self):
        records = [
            {'document_id': 1, 'chunk_no': 1, 'content': 'بودجه سالانه', 'vector': json.dumps([1.0]),
             'name': 'budget.txt', 'embed_model': 'synthetic'},
        ]
        failing = RetrievalService(lambda: FakeRepository(documents=records),
                                   lambda query, model: (_ for _ in ()).throw(ValueError('offline')), 'synthetic')
        result = failing.retrieve_documents('بودجه')
        self.assertEqual([row['document_id'] for row in result.records], [1])
        self.assertTrue(result.diagnostics.lexical_fallback)
        self.assertEqual(result.diagnostics.errors, ('ValueError',))

        calls = []
        mismatched = RetrievalService(
            lambda: FakeRepository(documents=[dict(records[0], embed_model='other-model')]),
            lambda query, model: calls.append((query, model)), 'synthetic')
        result = mismatched.retrieve_documents('بودجه')
        self.assertEqual([row['document_id'] for row in result.records], [1])
        self.assertFalse(result.diagnostics.semantic_attempted)
        self.assertFalse(result.diagnostics.lexical_fallback)
        self.assertEqual(calls, [])

    def test_low_semantic_match_without_lexical_hit_is_excluded(self):
        records = [
            {'document_id': 1, 'chunk_no': 1, 'content': 'نامرتبط', 'vector': json.dumps([0.2]),
             'name': 'unrelated.txt', 'embed_model': 'synthetic'},
        ]
        service = RetrievalService(lambda: FakeRepository(documents=records),
                                   lambda query, model: [1.0], 'synthetic')
        self.assertEqual(service.retrieve_documents('بودجه').records, [])

    def test_citation_evidence_and_metadata_ordering(self):
        facts = [{'id': 4, 'title': 'سیاست', 'category': 'منابع انسانی',
                  'source': 'مصوبه', 'content': 'متن دانش'}]
        documents = [{'document_id': 8, 'chunk_no': 3, 'name': 'policy.txt', 'content': 'متن سند'}]
        items = evidence_items(facts, documents)
        self.assertEqual([identifier for identifier, _ in items], ['K4', 'D8-C3'])
        self.assertIn('[K4]', items[0][1])
        self.assertIn('[D8-C3]', items[1][1])
        self.assertEqual(source_metadata(facts, documents),
                         {'source_ids': [4], 'doc_sources': ['D8-C3 (policy.txt)']})


if __name__ == '__main__':
    unittest.main()
