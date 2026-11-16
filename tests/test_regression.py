# -*- coding: utf-8 -*-
"""Offline regression coverage using only synthetic files and a temporary DB."""
import importlib.util
import io
import sqlite3
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('assistant_app', ROOT / 'ollama_persian_rag.py')
app = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(app)


class OfflineAssistantRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        app.DATA = Path(self.temp_dir.name)
        app.DB = app.DATA / 'synthetic.sqlite3'
        app.initialize()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_persian_normalization_and_chunk_overlap(self):
        self.assertEqual(app.norm('  كاربَر\u200c ي  '), 'کاربر ی')
        self.assertEqual(app.norm('سَلام'), 'سلام')
        chunks = app.chunk_document('الف ' * 100, chunk_chars=80, overlap=15)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunks))

    def test_knowledge_and_lexical_document_retrieval(self):
        now = '2026-01-01T00:00:00'
        with app.connect() as db:
            db.execute('INSERT INTO knowledge(title,category,content,source,approved,created_at,updated_at) VALUES(?,?,?,?,?,?,?)',
                       ('سیاست مرخصی', 'منابع انسانی', 'مرخصی سالانه بیست روز است.', 'مصوبه', 1, now, now))
        facts = app.search_knowledge('مرخصی سالانه چگونه است؟')
        self.assertEqual(facts[0]['title'], 'سیاست مرخصی')
        result = app.ingest_document('policy.txt', 'راهنمای ماموریت اداری و هزینه سفر'.encode(), '')
        self.assertEqual(result['mode'], 'lexical')
        found = app.search_documents('هزینه ماموریت')
        self.assertEqual(found[0]['document_id'], result['id'])
        with self.assertRaisesRegex(ValueError, 'قبلاً'):
            app.ingest_document('copy.txt', 'راهنمای ماموریت اداری و هزینه سفر'.encode(), '')

    def test_rejects_empty_filename_after_path_sanitization(self):
        with self.assertRaisesRegex(ValueError, 'نام فایل'):
            app.ingest_document('/', b'plain synthetic text', '')
        with self.assertRaisesRegex(ValueError, 'پشتیبانی'):
            app.ingest_document('unsupported.exe', b'synthetic content', '')

    def test_docx_and_xlsx_extraction(self):
        docx = io.BytesIO()
        with zipfile.ZipFile(docx, 'w') as z:
            z.writestr('word/document.xml', '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>متن فارسی</w:t></w:r></w:p></w:body></w:document>')
        self.assertIn('متن فارسی', app.document_text('sample.docx', docx.getvalue()))
        xlsx = io.BytesIO()
        with zipfile.ZipFile(xlsx, 'w') as z:
            z.writestr('xl/sharedStrings.xml', '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><si><t>بودجه</t></si></sst>')
            z.writestr('xl/worksheets/sheet1.xml', '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row><c r="A1" t="s"><v>0</v></c><c r="B1" t="inlineStr"><is><t>۱۴۰۵</t></is></c></row></sheetData></worksheet>')
        text = app.document_text('budget.xlsx', xlsx.getvalue())
        self.assertIn('A1=بودجه', text)
        self.assertIn('B1=۱۴۰۵', text)

    def test_rejects_oversized_office_archive_and_keeps_foreign_keys(self):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('word/document.xml', 'x' * (app.MAX_ARCHIVE_UNCOMPRESSED + 1))
        with self.assertRaisesRegex(ValueError, 'فشرده'):
            app.document_text('large.docx', archive.getvalue())
        with self.assertRaises(sqlite3.IntegrityError):
            with app.connect() as db:
                db.execute("INSERT INTO messages(chat_id,role,content,created_at) VALUES ('missing','user','x','now')")

    def test_existing_database_is_migrated_without_losing_history(self):
        chat_id = 'a' * 32
        with app.connect() as db:
            db.execute('INSERT INTO chats(id,title,created_at) VALUES (?, ?, ?)',
                       (chat_id, 'Synthetic chat', 'now'))
            db.execute('INSERT INTO messages(chat_id,role,content,created_at) VALUES (?, ?, ?, ?)',
                       (chat_id, 'user', 'Synthetic history', 'now'))
            db.execute('PRAGMA user_version=0')
        app.initialize()
        with app.connect() as db:
            self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0], app.SCHEMA_VERSION)
            self.assertEqual(db.execute('SELECT content FROM messages').fetchone()[0], 'Synthetic history')

    def test_backup_uses_sqlite_api_and_is_restorable(self):
        app.ingest_document('backup.txt', 'داده ساختگی برای پشتیبان'.encode(), '')
        backup = app.backup_database(Path(self.temp_dir.name) / 'backup.sqlite3')
        self.assertTrue(backup.exists())
        with sqlite3.connect(backup) as restored:
            self.assertEqual(restored.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
            self.assertEqual(restored.execute('SELECT COUNT(*) FROM documents').fetchone()[0], 1)

    def test_health_reports_failed_ollama_connection_without_database_failure(self):
        with mock.patch.object(app.urllib.request, 'urlopen', side_effect=app.urllib.error.URLError('offline')):
            health = app.health_status()
        self.assertFalse(health['ollama']['connected'])
        self.assertTrue(health['database']['connected'])
        self.assertEqual(health['database']['integrity'], 'ok')
        self.assertEqual(health['embedding']['status'], 'not_configured')

    def test_health_endpoint_returns_diagnostics(self):
        class FakeHandler:
            path = '/api/health'

            def json_response(self, result, code=200):
                self.result = result
                self.code = code

        handler = FakeHandler()
        expected = {'ollama': {'connected': False}, 'embedding': {}, 'database': {'connected': True}}
        with mock.patch.object(app, 'health_status', return_value=expected):
            app.Handler.do_GET(handler)
        self.assertEqual(handler.code, 200)
        self.assertEqual(handler.result, expected)

    def test_interrupted_stream_does_not_store_partial_messages(self):
        class InterruptedResponse:
            def __enter__(self):
                return self

            def __exit__(self, *unused):
                return False

            def __iter__(self):
                yield b'{"message":{"content":"partial"},"done":false}\n'
                raise ConnectionResetError('synthetic interruption')

        class FakeHandler:
            def __init__(self):
                self.wfile = io.BytesIO()

            def send_response(self, *unused):
                pass

            def send_header(self, *unused):
                pass

            def end_headers(self):
                pass

        handler = FakeHandler()
        request = {'chat_id': None, 'question': 'سوال ساختگی', 'content': 'سوال ساختگی',
                   'model': 'gemma-synthetic', 'ctx': 4096}
        with mock.patch.object(app.urllib.request, 'urlopen', return_value=InterruptedResponse()):
            app.Handler.chat_stream(handler, request)
        with app.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM messages').fetchone()[0], 0)


if __name__ == '__main__':
    unittest.main()
