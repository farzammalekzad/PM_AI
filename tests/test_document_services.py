"""Synthetic tests for offline document extraction and ingestion services."""
import io
import sqlite3
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from app.db.connection import connect
from app.db.migrations import initialize_database
from app.db.repositories import Repositories
from app.services.document_extractor import DocumentExtractor
from app.services.document_ingestion import DocumentIngestionService, chunk_document


class DocumentServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        path = Path(self.temp_dir.name) / 'synthetic.sqlite3'
        self.connect = lambda: connect(path)
        initialize_database(self.connect)
        self.repository = Repositories(self.connect)
        self.extractor = DocumentExtractor(3000, 24 * 1024 * 1024)

    def tearDown(self):
        self.temp_dir.cleanup()

    def ingestion(self, embedder=lambda text, model: [1.0, 0.0]):
        return DocumentIngestionService(lambda: self.repository, self.extractor, embedder,
                                        8 * 1024 * 1024, 1_000_000)

    def test_text_encodings_chunk_boundaries_and_unsupported_formats(self):
        self.assertEqual(self.extractor.extract('persian.txt', 'سلام'.encode()), 'سلام')
        self.assertEqual(self.extractor.extract('windows.txt', 'سلام'.encode('cp1256')), 'سلام')
        self.assertEqual(self.extractor.extract('utf16.txt', 'سلام'.encode('utf-16')), 'سلام')
        chunks = chunk_document('الف ' * 100, chunk_chars=80, overlap=15)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(chunks[0][-10:].strip() in chunks[1])
        with self.assertRaisesRegex(ValueError, 'پشتیبانی'):
            self.extractor.extract('unsupported.exe', b'synthetic')

    def test_docx_xlsx_pdf_and_archive_safety(self):
        docx = io.BytesIO()
        with zipfile.ZipFile(docx, 'w') as archive:
            archive.writestr('word/document.xml',
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                '<w:body><w:p><w:r><w:t>بند اول</w:t></w:r></w:p>'
                '<w:p><w:r><w:t>بند دوم</w:t></w:r></w:p></w:body></w:document>')
        self.assertEqual(self.extractor.extract('sample.docx', docx.getvalue()), 'بند اول\nبند دوم')
        xlsx = io.BytesIO()
        with zipfile.ZipFile(xlsx, 'w') as archive:
            archive.writestr('xl/sharedStrings.xml', '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><si><t>بودجه</t></si></sst>')
            archive.writestr('xl/worksheets/sheet1.xml', '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row><c r="A1" t="s"><v>0</v></c><c r="B1" t="inlineStr"><is><t>۱۴۰۵</t></is></c></row></sheetData></worksheet>')
        text = self.extractor.extract('sample.xlsx', xlsx.getvalue())
        self.assertIn('A1=بودجه', text)
        self.assertIn('B1=۱۴۰۵', text)

        class Page:
            def extract_text(self): return 'PDF synthetic text'
        class Reader:
            def __init__(self, stream): self.pages = [Page()]
        with mock.patch.dict(sys.modules, {'pypdf': types.SimpleNamespace(PdfReader=Reader)}):
            self.assertEqual(self.extractor.extract('sample.pdf', b'%PDF synthetic'), 'PDF synthetic text')
        with self.assertRaises(zipfile.BadZipFile):
            self.extractor.extract('broken.docx', b'not a zip')
        with self.assertRaisesRegex(ValueError, 'فشرده'):
            DocumentExtractor(1, 10).extract('large.docx', docx.getvalue())

    def test_ingestion_duplicate_empty_embedding_failure_and_atomic_rollback(self):
        result = self.ingestion().ingest('policy.txt', 'سند ساختگی'.encode(), '')
        self.assertEqual(result['mode'], 'lexical')
        with self.assertRaisesRegex(ValueError, 'قبلاً'):
            self.ingestion().ingest('copy.txt', 'سند ساختگی'.encode(), '')
        with self.assertRaisesRegex(ValueError, 'متنی'):
            self.ingestion().ingest('empty.txt', b'   ', '')
        with self.assertRaisesRegex(ValueError, 'synthetic embedding failure'):
            self.ingestion(lambda text, model: (_ for _ in ()).throw(ValueError('synthetic embedding failure'))).ingest(
                'failure.txt', b'new synthetic document', 'qwen3-synthetic')
        with self.assertRaises(sqlite3.IntegrityError):
            self.repository.create_document_with_chunks(
                'atomic.txt', 'a' * 64, 10, [(1, 'first', ''), (1, 'duplicate', '')], '', 'now')
        with self.connect() as database:
            self.assertEqual(database.execute('SELECT COUNT(*) FROM documents').fetchone()[0], 1)
            self.assertEqual(database.execute('SELECT COUNT(*) FROM doc_chunks').fetchone()[0], 1)

    def test_listing_metadata_same_name_and_exact_duplicate_detection(self):
        first = self.ingestion().ingest('shared-name.txt', b'first synthetic body', 'synthetic-model')
        second = self.ingestion().ingest('shared-name.txt', b'second synthetic body', '')
        self.assertNotEqual(first['id'], second['id'])
        listed = {row['id']: row for row in self.repository.list_documents()}
        self.assertEqual(listed[first['id']]['embed_model'], 'synthetic-model')
        self.assertEqual(listed[first['id']]['vector_chunks'], listed[first['id']]['chunks'])
        self.assertEqual(listed[second['id']]['vector_chunks'], 0)
        self.assertEqual(listed[first['id']]['status'], 'active')
        with self.assertRaisesRegex(ValueError, 'قبلاً'):
            self.ingestion().ingest('different-name.txt', b'first synthetic body', '')

    def test_metadata_edit_and_replacement_preserve_stable_identifier(self):
        original = self.ingestion().ingest(
            'original.txt', ('old synthetic content ' * 200).encode(), 'synthetic-model')
        self.ingestion().update_metadata(original['id'], 'renamed.txt')
        before = self.repository.document_records()
        self.assertGreater(len(before), 1)
        self.assertEqual(self.repository.get_document(original['id'])['name'], 'renamed.txt')
        replacement = self.ingestion().replace(
            original['id'], 'replacement.txt', b'new synthetic content', 'synthetic-model', True)
        self.assertEqual(replacement['id'], original['id'])
        records = self.repository.document_records()
        self.assertEqual([row['document_id'] for row in records], [original['id']])
        self.assertEqual([row['content'] for row in records], ['new synthetic content'])
        self.assertEqual(self.repository.get_document(original['id'])['name'], 'replacement.txt')

    def test_failed_replacement_keeps_original_chunks_and_vectors(self):
        original = self.ingestion().ingest('original.txt', b'original retained', 'synthetic-model')
        before = [dict(row) for row in self.repository.document_records()]
        failing = self.ingestion(lambda text, model: (_ for _ in ()).throw(ValueError('embedding failure')))
        with self.assertRaisesRegex(ValueError, 'embedding failure'):
            failing.replace(original['id'], 'replacement.txt', b'replacement fails', 'synthetic-model', True)
        self.assertEqual([dict(row) for row in self.repository.document_records()], before)
        with self.assertRaisesRegex(ValueError, 'تأیید'):
            self.ingestion().replace(original['id'], 'replacement.txt', b'new body', '', False)


if __name__ == '__main__':
    unittest.main()
