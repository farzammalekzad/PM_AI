"""SQLite connection configuration shared by all database operations."""
import sqlite3


def connect(database_path):
    """Open a configured SQLite connection without changing schema or data."""
    database = sqlite3.connect(database_path, timeout=15)
    database.row_factory = sqlite3.Row
    database.execute('PRAGMA busy_timeout=15000')
    database.execute('PRAGMA foreign_keys=ON')
    return database
