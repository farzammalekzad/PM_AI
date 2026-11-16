"""Additive, idempotent SQLite schema initialization and migrations."""

SCHEMA_VERSION = 1

INITIAL_SCHEMA = '''
CREATE TABLE IF NOT EXISTS knowledge (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT NOT NULL,
  category TEXT NOT NULL DEFAULT 'عمومی',
  content TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT '',
  approved INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chats (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id TEXT NOT NULL REFERENCES chats(id),
  role TEXT NOT NULL CHECK (role IN ('user','assistant')),
  content TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id,id);
CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE,
  chars INTEGER NOT NULL, chunks INTEGER NOT NULL DEFAULT 0,
  embed_model TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS doc_chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  chunk_no INTEGER NOT NULL,
  content TEXT NOT NULL,
  vector TEXT NOT NULL DEFAULT '',
  UNIQUE(document_id,chunk_no)
);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_document ON doc_chunks(document_id);
'''


def migrate_v1(database):
    """Record the pre-existing schema as version 1 without altering rows."""
    database.execute('CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id,id)')
    database.execute('CREATE INDEX IF NOT EXISTS idx_doc_chunks_document ON doc_chunks(document_id)')


MIGRATIONS = {1: migrate_v1}


def apply_migrations(database, log_event=None):
    """Apply only missing known migrations in one caller-owned transaction."""
    current = database.execute('PRAGMA user_version').fetchone()[0]
    if current > SCHEMA_VERSION:
        raise RuntimeError('نسخه پایگاه‌داده از نسخه برنامه جدیدتر است')
    for version in range(current + 1, SCHEMA_VERSION + 1):
        MIGRATIONS[version](database)
        database.execute(f'PRAGMA user_version={version}')
        if log_event:
            log_event('info', 'database_migrated', version=version)


def initialize_database(connect, log_event=None):
    """Create missing historical tables and apply safe incremental migrations."""
    with connect() as database:
        database.execute('PRAGMA journal_mode=WAL')
        database.executescript(INITIAL_SCHEMA)
        apply_migrations(database, log_event)
