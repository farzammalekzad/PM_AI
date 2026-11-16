"""Synthetic tests for the extracted SQLite database package."""
import sqlite3
import tempfile
import unittest
from pathlib import Path

from app.db.backup import backup_database, database_status
from app.db.connection import connect
from app.db.migrations import SCHEMA_VERSION, initialize_database
from app.db.repositories import Repositories


class DatabaseLayerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / 'synthetic.sqlite3'
        self.connect = lambda: connect(self.database_path)
        initialize_database(self.connect)
        self.repositories = Repositories(self.connect)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_connection_migration_and_legacy_history_compatibility(self):
        chat_id = 'a' * 32
        with self.connect() as database:
            self.assertEqual(database.execute('PRAGMA foreign_keys').fetchone()[0], 1)
            database.execute('INSERT INTO chats(id,title,created_at) VALUES(?,?,?)',
                             (chat_id, 'Synthetic chat', 'now'))
            database.execute('INSERT INTO messages(chat_id,role,content,created_at) VALUES(?,?,?,?)',
                             (chat_id, 'user', 'Synthetic history', 'now'))
            database.execute('PRAGMA user_version=0')
        initialize_database(self.connect)
        self.assertEqual(database_status(self.connect)['schema_version'], SCHEMA_VERSION)
        self.assertEqual(self.repositories.chat_history(chat_id)[0]['content'], 'Synthetic history')

    def test_knowledge_crud_preserves_repository_contract(self):
        created = self.repositories.save_knowledge(
            None, 'Synthetic policy', 'Synthetic category', 'Initial content', 'Synthetic source', 'now')
        self.assertEqual(self.repositories.approved_knowledge()[0]['id'], created)
        self.repositories.save_knowledge(
            created, 'Synthetic policy', 'Synthetic category', 'Updated content', 'Synthetic source', 'later')
        self.assertEqual(self.repositories.list_knowledge()[0]['content'], 'Updated content')
        self.repositories.delete_knowledge(created)
        self.assertEqual(self.repositories.list_knowledge(), [])

    def test_document_chunks_and_chat_history_persist(self):
        document_id = self.repositories.create_document_with_chunks(
            'synthetic.txt', 'a' * 64, 20,
            [(1, 'Synthetic first chunk', ''), (2, 'Synthetic second chunk', '')], '', 'now')
        records = self.repositories.document_records()
        self.assertEqual([record['chunk_no'] for record in records], [1, 2])
        self.assertEqual(records[0]['document_id'], document_id)
        chat_id = 'b' * 32
        self.repositories.create_chat(chat_id, 'Synthetic chat', 'now')
        self.repositories.store_exchange(chat_id, 'Synthetic question', 'Synthetic answer', 'now')
        self.assertEqual([row['role'] for row in self.repositories.chat_history(chat_id)],
                         ['user', 'assistant'])

    def test_backup_and_integrity_status(self):
        self.repositories.create_document_with_chunks(
            'backup.txt', 'b' * 64, 10, [(1, 'Synthetic backup chunk', '')], '', 'now')
        destination = Path(self.temp_dir.name) / 'restored.sqlite3'
        backup_database(self.connect, self.temp_dir.name, destination)
        with sqlite3.connect(destination) as restored:
            self.assertEqual(restored.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
            self.assertEqual(restored.execute('SELECT COUNT(*) FROM doc_chunks').fetchone()[0], 1)
        status = database_status(self.connect)
        self.assertEqual(status['integrity'], 'ok')
        self.assertEqual(status['document_count'], 1)
        self.assertEqual(status['chunk_count'], 1)


if __name__ == '__main__':
    unittest.main()
