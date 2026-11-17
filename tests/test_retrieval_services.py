"""Synthetic regression tests for retrieval, diagnostics, and citation formatting."""
import json
import unittest

from app.services.citation_service import cited_document_sources, evidence_items, source_location, source_metadata
from app.services.retrieval_service import RetrievalService, cosine_similarity
from app.services.retrieval_service import normalize_persian, persian_tokens, tokenize_persian


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

    def test_persian_punctuation_and_variants_preserve_token_boundaries(self):
        self.assertEqual(tokenize_persian('می‌شود؟'), {'شود'})
        self.assertEqual(tokenize_persian('مي شود؟'), {'شود'})
        self.assertEqual(normalize_persian('كـتاب ي ى'), 'کتاب ی ی')
        self.assertNotIn('شود؟', persian_tokens('می‌شود؟'))
        records = [
            {'document_id': 1, 'chunk_no': 1, 'content': 'فعالیت باید ثبت شود.',
             'vector': None, 'name': 'plan.txt', 'embed_model': ''},
        ]
        result = RetrievalService(lambda: FakeRepository(documents=records), None, '').retrieve_documents(
            'چه فعالیتی باید ثبت شود؟')
        self.assertEqual([item['document_id'] for item in result.records], [1])

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
        self.assertEqual(result.diagnostics.mode, 'lexical')
        self.assertEqual(result.diagnostics.fallback_reason, 'embedding_error')
        self.assertGreaterEqual(result.diagnostics.duration_ms, 0)

        calls = []
        mismatched = RetrievalService(
            lambda: FakeRepository(documents=[dict(records[0], embed_model='other-model')]),
            lambda query, model: calls.append((query, model)), 'synthetic')
        result = mismatched.retrieve_documents('بودجه')
        self.assertEqual([row['document_id'] for row in result.records], [1])
        self.assertFalse(result.diagnostics.semantic_attempted)
        self.assertFalse(result.diagnostics.lexical_fallback)
        self.assertEqual(result.diagnostics.mode, 'lexical')
        self.assertEqual(result.diagnostics.fallback_reason, 'no_compatible_vectors')
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
        documents = [{'document_id': 8, 'chunk_no': 3, 'name': 'policy.txt', 'content': 'متن سند',
                      'page_start': 2, 'page_end': 3}]
        items = evidence_items(facts, documents)
        self.assertEqual([identifier for identifier, _ in items], ['K4', 'D8-C3'])
        self.assertIn('[K4]', items[0][1])
        self.assertIn('[D8-C3]', items[1][1])
        self.assertIn('صفحه 2 تا 3', items[1][1])
        metadata = source_metadata(facts, documents)
        self.assertEqual(metadata['source_ids'], [4])
        self.assertEqual(metadata['doc_sources'], ['D8-C3 (policy.txt)'])
        self.assertEqual(metadata['doc_source_details'][0]['location'], 'صفحه 2 تا 3')
        self.assertEqual(source_location({'document_id': 1, 'chunk_no': 1}), '')

    def test_cited_document_sources_only_return_known_retrieved_documents_once(self):
        documents = [
            {'document_id': 4, 'chunk_no': 1, 'name': 'renamed.pdf', 'page_start': 5, 'page_end': 5},
            {'document_id': 5, 'chunk_no': 2, 'name': 'budget.xlsx', 'worksheet': 'بودجه',
             'row_start': 7, 'row_end': 9},
            {'document_id': 6, 'chunk_no': 1, 'name': 'legacy.txt'},
        ]
        sources = cited_document_sources(
            'پاسخ [D4-C1] و [D5-C2] و دوباره [D4-C1] و ناشناخته [D9-C9]', documents)
        self.assertEqual([item['identifier'] for item in sources], ['D4-C1', 'D5-C2'])
        self.assertEqual(sources[0]['name'], 'renamed.pdf')
        self.assertEqual(sources[0]['location'], 'صفحه 5')
        self.assertEqual(sources[1]['location'], 'کاربرگ بودجه، ردیف 7 تا 9')
        self.assertEqual(cited_document_sources('ارجاع قدیمی [D6-C1]', documents)[0]['location'], '')


if __name__ == '__main__':
    unittest.main()
