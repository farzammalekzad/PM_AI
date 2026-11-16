"""Compatibility-preserving lexical and optional semantic retrieval services."""
import json
import re
import time
from dataclasses import dataclass

from app.services.citation_service import document_identifier


PERSIAN_STOP = {
    'این', 'آن', 'های', 'برای', 'است', 'هست', 'باشد', 'شده', 'درباره', 'چگونه', 'چه', 'چرا',
    'شرکت', 'لطفا', 'لطفاً', 'بگو', 'کن', 'کنید', 'را', 'از', 'در', 'به', 'با', 'یا', 'و', 'که',
    'یک', 'آیا', 'اطلاعات',
}
KNOWLEDGE_STOP = {
    'برای', 'چگونه', 'درباره', 'میخواهم', 'میشود', 'چیست', 'کنید', 'لطفا', 'اینکه', 'همه',
    'اطلاعات', 'شرکت', 'دارد',
}


def normalize_persian(text):
    """Keep the established Arabic/Persian normalization used by lexical retrieval."""
    normalized = str(text).lower().replace('ي', 'ی').replace('ك', 'ک').replace('‌', ' ')
    normalized = re.sub(r'[\u064b-\u065f\u0670]', '', normalized)
    return re.sub(r'\s+', ' ', normalized).strip()


def tokenize_persian(text):
    return set(token for token in re.findall(r'[\w\u0600-\u06ff]{3,}', normalize_persian(text))
               if token not in PERSIAN_STOP)


def cosine_similarity(left, right):
    """The stored and query vectors are normalized by the existing Ollama client."""
    return sum(first * second for first, second in zip(left, right))


@dataclass(frozen=True)
class RetrievalDiagnostics:
    embedding_model: str
    semantic_attempted: bool
    embedding_succeeded: bool
    lexical_fallback: bool
    candidate_chunks: int
    selected_chunks: int
    selected_source_identifiers: tuple
    mode: str
    fallback_reason: str
    duration_ms: int
    errors: tuple


@dataclass(frozen=True)
class DocumentRetrievalResult:
    records: list
    diagnostics: RetrievalDiagnostics


class RetrievalService:
    """Retrieve existing knowledge and chunk records without changing their ranking."""

    def __init__(self, repository_factory, embedder, embedding_model):
        self.repository_factory = repository_factory
        self.embedder = embedder
        self.embedding_model = embedding_model

    def search_knowledge(self, question, limit=5):
        terms = [term for term in re.findall(r'[\w\u0600-\u06ff]{3,}', normalize_persian(question))
                 if term not in KNOWLEDGE_STOP]
        terms = list(dict.fromkeys(terms))[:24]
        rows = self.repository_factory().approved_knowledge()
        scored = []
        for row in rows:
            title = normalize_persian(row['title'])
            content = normalize_persian(row['content'])
            category = normalize_persian(row['category'])
            score = sum((5 if term in title else 0) + (2 if term in category else 0) +
                        (1 if term in content else 0) for term in terms)
            if score:
                scored.append((score, row))
        scored.sort(key=lambda value: (-value[0], value[1]['id']))
        return [dict(row) for _, row in scored[:limit]]

    def retrieve_documents(self, query, limit=5):
        started = time.monotonic()
        records = self.repository_factory().document_records()
        words = tokenize_persian(query)
        model = self.embedding_model
        compatible_vectors = model and any(
            record['embed_model'] == model and record['vector'] for record in records)
        query_vector = None
        semantic_attempted = bool(compatible_vectors)
        embedding_succeeded = False
        errors = []
        if compatible_vectors:
            try:
                query_vector = self.embedder(query, model)
                embedding_succeeded = True
            except Exception as error:
                errors.append(type(error).__name__)
        ranked = []
        for record in records:
            body = normalize_persian(record['content'])
            title = normalize_persian(record['name'])
            hits = (sum(2 if word in title else 0 for word in words) +
                    sum(1 for word in words if word in body))
            lexical = hits / max(1, len(words))
            semantic = None
            if query_vector is not None and record['embed_model'] == model and record['vector']:
                try:
                    vector = json.loads(record['vector'])
                    if len(vector) == len(query_vector):
                        semantic = cosine_similarity(query_vector, vector)
                except (ValueError, TypeError) as error:
                    errors.append(type(error).__name__)
            if semantic is None:
                if not hits:
                    continue
                score = lexical
            else:
                if semantic < 0.38 and not hits:
                    continue
                score = 1.8 * semantic + 0.5 * min(lexical, 3)
            ranked.append((score, dict(record)))
        ranked.sort(key=lambda value: -value[0])
        selected = [record for _, record in ranked[:limit]]
        mode = 'hybrid' if query_vector is not None else 'lexical'
        if not model:
            fallback_reason = 'embedding_not_configured'
        elif semantic_attempted and query_vector is None:
            fallback_reason = 'embedding_error'
        elif not semantic_attempted:
            fallback_reason = 'no_compatible_vectors'
        else:
            fallback_reason = None
        diagnostics = RetrievalDiagnostics(
            embedding_model=model,
            semantic_attempted=semantic_attempted,
            embedding_succeeded=embedding_succeeded,
            lexical_fallback=semantic_attempted and query_vector is None,
            candidate_chunks=len(records),
            selected_chunks=len(selected),
            selected_source_identifiers=tuple(
                document_identifier(record) for record in selected),
            mode=mode,
            fallback_reason=fallback_reason,
            duration_ms=int((time.monotonic() - started) * 1000),
            errors=tuple(dict.fromkeys(errors)),
        )
        return DocumentRetrievalResult(selected, diagnostics)
