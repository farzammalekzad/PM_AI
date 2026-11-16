"""Synthetic tests for read-only RAG index monitoring statistics."""
import json
import tempfile
import unittest
from pathlib import Path

from app.db.connection import connect
from app.db.migrations import initialize_database
from app.db.repositories import Repositories
from app.services.rag_monitoring import RagMonitoringService


class RagMonitoringTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        path = Path(self.temp_dir.name) / 'synthetic.sqlite3'
        self.connect = lambda: connect(path)
        initialize_database(self.connect)
        self.repository = Repositories(self.connect)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_mixed_models_missing_vectors_and_invalid_vectors_are_counted(self):
        self.repository.create_document_with_chunks(
            'compatible.txt', 'a' * 64, 10,
            [(1, 'one', json.dumps([0.6, 0.8])), (2, 'two', '')], 'model-a', 'now')
        self.repository.create_document_with_chunks(
            'lexical.txt', 'b' * 64, 10, [(1, 'three', '')], '', 'now')
        self.repository.create_document_with_chunks(
            'other.txt', 'c' * 64, 10, [(1, 'four', 'not-json')], 'model-b', 'now')
        statistics = RagMonitoringService(lambda: self.repository, 'model-a').index_statistics(3)
        self.assertEqual(statistics['total_documents'], 3)
        self.assertEqual(statistics['total_chunks'], 4)
        self.assertEqual(statistics['documents_with_embeddings'], 2)
        self.assertEqual(statistics['documents_without_embeddings'], 1)
        self.assertEqual(statistics['chunks_with_vectors'], 2)
        self.assertEqual(statistics['chunks_without_vectors'], 2)
        self.assertEqual(statistics['compatible_vector_chunks'], 1)
        self.assertEqual(statistics['incompatible_vector_chunks'], 1)
        self.assertEqual(statistics['invalid_vector_chunks'], 1)
        self.assertEqual(statistics['dimension_mismatch_chunks'], 1)
        self.assertEqual(statistics['embedding_model_distribution'],
                         {'': 1, 'model-a': 1, 'model-b': 1})
        self.assertEqual(statistics['vector_dimension_distribution'], {'2': 1})

    def test_empty_index_reports_zero_coverage(self):
        statistics = RagMonitoringService(lambda: self.repository, '').index_statistics()
        self.assertEqual(statistics['total_documents'], 0)
        self.assertEqual(statistics['total_chunks'], 0)
        self.assertEqual(statistics['chunks_with_vectors'], 0)


if __name__ == '__main__':
    unittest.main()
