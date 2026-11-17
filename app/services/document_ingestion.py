"""Validation, chunking, embeddings, and atomic persistence for uploaded documents."""
import hashlib
import json
from datetime import datetime
from pathlib import Path


def chunk_document(text, chunk_chars=1300, overlap=180):
    """Use the established character chunks and overlap for retrieval compatibility."""
    text = text.strip()
    if not text:
        return []
    chunks = []
    position = 0
    while position < len(text):
        end = min(position + chunk_chars, len(text))
        if end < len(text):
            pivot = max(text.rfind('\n', position + chunk_chars // 2, end),
                        text.rfind(' ', position + chunk_chars // 2, end))
            if pivot > position + chunk_chars // 2:
                end = pivot + 1
        chunks.append(text[position:end].strip())
        if end >= len(text):
            break
        position = max(position + 1, end - overlap)
    return [chunk for chunk in chunks if chunk]


class DocumentIngestionService:
    """Prepare all derived data before one repository transaction persists it."""

    def __init__(self, repository_factory, extractor, embedder, max_document_bytes,
                 max_document_chars, max_chunks=1000):
        self.repository_factory = repository_factory
        self.extractor = extractor
        self.embedder = embedder
        self.max_document_bytes = max_document_bytes
        self.max_document_chars = max_document_chars
        self.max_chunks = max_chunks

    def ingest(self, name, raw, model):
        prepared = self._prepare(name, raw, model)
        repository = self.repository_factory()
        existing = repository.find_document_by_digest(prepared['digest'])
        if existing:
            raise ValueError('این فایل قبلاً وارد پایگاه دانش شده است (شناسه ' +
                             str(existing['id']) + ')')
        identifier = repository.create_document_with_chunks(
            prepared['name'], prepared['digest'], prepared['char_count'], prepared['chunks'], model,
            datetime.now().isoformat(timespec='seconds'))
        return {'id': identifier, 'name': prepared['name'], 'chunks': len(prepared['chunks']),
                'mode': 'semantic' if model else 'lexical'}

    def update_metadata(self, identifier, name):
        self.repository_factory().update_document_metadata(
            identifier, self._safe_name(name), datetime.now().isoformat(timespec='seconds'))
        return {'id': int(identifier), 'updated': True}

    def replace(self, identifier, name, raw, model, confirmed):
        if confirmed is not True:
            raise ValueError('جایگزینی سند نیازمند تأیید صریح است')
        repository = self.repository_factory()
        if not repository.get_document(identifier):
            raise ValueError('سند موردنظر یافت نشد')
        prepared = self._prepare(name, raw, model)
        stable_id = repository.replace_document_with_chunks(
            identifier, prepared['name'], prepared['digest'], prepared['char_count'], prepared['chunks'], model,
            datetime.now().isoformat(timespec='seconds'))
        return {'id': stable_id, 'name': prepared['name'], 'chunks': len(prepared['chunks']),
                'replaced': True, 'mode': 'semantic' if model else 'lexical'}

    def _prepare(self, name, raw, model):
        name = self._safe_name(name)
        if len(raw) > self.max_document_bytes:
            raise ValueError('حجم سند بیشتر از ۸ مگابایت است')
        text = self.extractor.extract(name, raw)
        if not text.strip():
            raise ValueError('متنی در سند یافت نشد؛ PDF اسکن‌شده به OCR نیاز دارد')
        if len(text) > self.max_document_chars:
            raise ValueError('متن سند بیش از حد بزرگ است؛ آن را به چند فایل تقسیم کنید')
        chunks = chunk_document(text)
        if len(chunks) > self.max_chunks:
            raise ValueError('تعداد بخش‌های سند بیش از حد زیاد است')
        vectors = [self.embedder(chunk, model) for chunk in chunks] if model else [None] * len(chunks)
        return {'name': name, 'digest': hashlib.sha256(raw).hexdigest(), 'char_count': len(text),
                'chunks': [(index + 1, chunk, json.dumps(vector) if vector else '')
                           for index, (chunk, vector) in enumerate(zip(chunks, vectors))]}

    @staticmethod
    def _safe_name(name):
        if not name or len(name) > 260:
            raise ValueError('نام فایل نامعتبر است')
        name = Path(name.replace('\\', '/')).name
        if not name or name in ('.', '..'):
            raise ValueError('نام فایل نامعتبر است')
        return name
