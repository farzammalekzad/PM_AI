"""Read-only offline RAG index monitoring based on existing SQLite records."""
import json
from collections import Counter


class RagMonitoringService:
    """Calculate non-sensitive embedding coverage without touching index content."""

    def __init__(self, repository_factory, embedding_model):
        self.repository_factory = repository_factory
        self.embedding_model = embedding_model

    def index_statistics(self, expected_dimension=None):
        records = self.repository_factory().index_records()
        document_models = {}
        chunks_with_vectors = 0
        chunks_without_vectors = 0
        compatible_vector_chunks = 0
        incompatible_vector_chunks = 0
        invalid_vector_chunks = 0
        dimension_mismatch_chunks = 0
        dimensions = Counter()
        for record in records:
            document_models[record['document_id']] = record['embed_model']
            if record['chunk_id'] is None:
                continue
            vector = record['vector'] or ''
            if not vector:
                chunks_without_vectors += 1
                continue
            chunks_with_vectors += 1
            if self.embedding_model and record['embed_model'] == self.embedding_model:
                compatible_vector_chunks += 1
            else:
                incompatible_vector_chunks += 1
            try:
                parsed = json.loads(vector)
                if not isinstance(parsed, list):
                    raise ValueError('not a vector')
                dimensions[str(len(parsed))] += 1
                if (expected_dimension is not None and self.embedding_model and
                        record['embed_model'] == self.embedding_model and
                        len(parsed) != expected_dimension):
                    dimension_mismatch_chunks += 1
            except (TypeError, ValueError, json.JSONDecodeError):
                invalid_vector_chunks += 1
        model_distribution = Counter(document_models.values())
        embedded_documents = sum(1 for model in document_models.values() if model)
        total_documents = len(document_models)
        return {
            'total_documents': total_documents,
            'total_chunks': chunks_with_vectors + chunks_without_vectors,
            'documents_with_embeddings': embedded_documents,
            'documents_without_embeddings': total_documents - embedded_documents,
            'chunks_with_vectors': chunks_with_vectors,
            'chunks_without_vectors': chunks_without_vectors,
            'embedding_model_distribution': dict(sorted(model_distribution.items())),
            'configured_embedding_model': self.embedding_model,
            'compatible_vector_chunks': compatible_vector_chunks,
            'incompatible_vector_chunks': incompatible_vector_chunks,
            'invalid_vector_chunks': invalid_vector_chunks,
            'dimension_mismatch_chunks': dimension_mismatch_chunks,
            'vector_dimension_distribution': dict(sorted(dimensions.items())),
        }
