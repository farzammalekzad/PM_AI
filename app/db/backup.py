"""Non-destructive SQLite integrity diagnostics and backup helpers."""
import sqlite3
from datetime import datetime
from pathlib import Path


def database_status(connect):
    """Return non-sensitive health data for an initialized SQLite database."""
    with connect() as database:
        integrity = database.execute('PRAGMA integrity_check').fetchone()[0]
        foreign_key_violations = len(database.execute('PRAGMA foreign_key_check').fetchall())
        document_count = database.execute('SELECT COUNT(*) FROM documents').fetchone()[0]
        chunk_count = database.execute('SELECT COUNT(*) FROM doc_chunks').fetchone()[0]
        schema_version = database.execute('PRAGMA user_version').fetchone()[0]
    return {
        'connected': True,
        'schema_version': schema_version,
        'integrity': integrity,
        'foreign_key_violations': foreign_key_violations,
        'document_count': document_count,
        'chunk_count': chunk_count,
    }


def backup_database(connect, data_directory, destination=None, log_event=None):
    """Create a new consistent SQLite backup and never overwrite a destination."""
    if destination is None:
        backup_directory = Path(data_directory) / 'backups'
        backup_directory.mkdir(parents=True, exist_ok=True)
        destination = backup_directory / ('company_memory-' +
                                          datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '.sqlite3')
    destination = Path(destination)
    if destination.exists():
        raise ValueError('فایل پشتیبان از قبل وجود دارد')
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with connect() as source, sqlite3.connect(destination, timeout=15) as target:
            source.backup(target)
    except Exception:
        if log_event:
            log_event('error', 'database_backup_failed', error_type='backup_error')
        raise
    if log_event:
        log_event('info', 'database_backup_created', filename=destination.name)
    return destination
