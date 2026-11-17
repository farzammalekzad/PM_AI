# -*- coding: utf-8 -*-
"""Offline Persian Ollama chat with SQLite knowledge and conversation history.
Python 3.10+, standard library only. Optional pypdf for text PDFs.
Run: python ollama_persian_memory.py
"""
import json
import logging
import sqlite3
import threading
import webbrowser
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.config import load_config
from app.db.backup import backup_database as create_database_backup
from app.db.backup import database_status as get_database_status
from app.db.connection import connect as database_connect
from app.db.migrations import SCHEMA_VERSION, apply_migrations as run_migrations
from app.db.migrations import initialize_database
from app.db.repositories import Repositories
from app.services.diagnostics import embedding_diagnostics
from app.services.chat_service import ChatService
from app.services.document_extractor import DocumentExtractor
from app.services.document_ingestion import DocumentIngestionService
from app.services.document_ingestion import chunk_document as service_chunk_document
from app.services.ollama_client import OllamaClient
from app.services.prompt_builder import PromptBuilder
from app.services.retrieval_service import RetrievalService
from app.services.retrieval_service import normalize_persian, tokenize_persian
from app.services.rag_monitoring import RagMonitoringService
from app.web.handlers import create_handler
from app.web.server import create_server

BASE = Path(__file__).resolve().parent
CONFIG = load_config(BASE)
DATA = CONFIG.data_directory
DATA.mkdir(parents=True, exist_ok=True)
DB = CONFIG.database_path
HOST = CONFIG.host
PORT = CONFIG.port
OLLAMA = CONFIG.ollama_url
MAX_REQUEST = CONFIG.max_request_bytes
MAX_ATTACHMENT_CHARS = CONFIG.max_attachment_chars
MAX_CONTEXT = CONFIG.max_context_chars
EMBED_MODEL = CONFIG.embedding_model
MAX_DOC_SIZE = CONFIG.max_document_bytes
MAX_DOC_CHARS = CONFIG.max_document_chars
# OOXML files are ZIP archives.  Limit their total declared uncompressed size,
# not just individual XML members, so a small compressed upload cannot cause a
# large allocation during local document extraction.
MAX_ARCHIVE_MEMBERS = CONFIG.max_archive_members
MAX_ARCHIVE_UNCOMPRESSED = CONFIG.max_archive_uncompressed_bytes
LOG_MAX_BYTES = CONFIG.log_max_bytes
LOG_BACKUP_COUNT = CONFIG.log_backup_count
ALLOWED_CONTEXT_WINDOWS = CONFIG.allowed_context_windows
def connect():
    """Compatibility wrapper for the established module-level database path."""
    return database_connect(DB)


def configure_logging():
    """Configure local operational logging without request or document content."""
    logger = logging.getLogger('offline_persian_assistant')
    target = str(DATA / 'assistant.log')
    for handler in logger.handlers:
        if getattr(handler, '_assistant_log_path', None) == target:
            return logger
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    DATA.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(target, maxBytes=LOG_MAX_BYTES,
                                  backupCount=LOG_BACKUP_COUNT, encoding='utf-8')
    handler._assistant_log_path = target
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def log_event(level, event, **details):
    """Write structured metadata only; never pass content, prompts, or credentials."""
    getattr(configure_logging(), level)(json.dumps(
        {'event': event, **details}, ensure_ascii=False, sort_keys=True))


def repositories():
    """Create a repository facade bound to the existing configured database."""
    return Repositories(connect)


def apply_migrations(database):
    """Compatibility wrapper for callers using the previous module function."""
    return run_migrations(database, log_event)


def initialize():
    initialize_database(connect, log_event)


def database_status():
    return get_database_status(connect)


def backup_database(destination=None):
    return create_database_backup(connect, DATA, destination, log_event)


def ollama_client(tags_timeout=None):
    """Create a local Ollama client from the compatibility configuration values."""
    return OllamaClient(
        OLLAMA,
        tags_timeout=CONFIG.ollama_tags_timeout if tags_timeout is None else tags_timeout,
        embedding_timeout=CONFIG.ollama_embedding_timeout,
        chat_timeout=CONFIG.ollama_chat_timeout)


def ollama_status():
    return {'connected': ollama_client(CONFIG.ollama_health_timeout).is_available()}


def embedding_status():
    return embedding_diagnostics(ollama_client(), EMBED_MODEL)


def health_status():
    try:
        database = database_status()
    except sqlite3.Error:
        log_event('error', 'database_health_failed', error_type='sqlite_error')
        database = {'connected': False}
    return {
        'ollama': ollama_status(),
        'embedding': embedding_status(),
        'database': database,
    }


def norm(text):
    """Compatibility wrapper for Persian normalization."""
    return normalize_persian(text)


def search_knowledge(question, limit=5):
    return retrieval_service().search_knowledge(question, limit)

def tokenize(text):
    return tokenize_persian(text)

def chunk_document(text, chunk_chars=1300, overlap=180):
    """Compatibility wrapper for the established retrieval chunk boundaries."""
    return service_chunk_document(text, chunk_chars, overlap)

def ollama_embedding(text, model):
    return ollama_client().embed(text, model)


def chat_service():
    """Create the compatibility chat coordinator for the current application wiring."""
    return ChatService(repositories, ollama_client, search_knowledge,
                       retrieval_service().retrieve_documents,
                       PromptBuilder(), ALLOWED_CONTEXT_WINDOWS)


def retrieval_service():
    """Create the compatibility retrieval service using the configured local model."""
    return RetrievalService(repositories, ollama_embedding, EMBED_MODEL)


def rag_monitoring_service():
    """Create read-only RAG index monitoring from existing SQLite metadata."""
    return RagMonitoringService(repositories, EMBED_MODEL)


def rag_monitoring_status():
    health = health_status()
    return {
        'health': health,
        'index': rag_monitoring_service().index_statistics(
            health.get('embedding', {}).get('dimension')),
    }


def document_extractor():
    """Create the compatibility extractor with the configured archive limits."""
    return DocumentExtractor(MAX_ARCHIVE_MEMBERS, MAX_ARCHIVE_UNCOMPRESSED)


def document_ingestion_service():
    """Create the compatibility ingestion service without changing embedding selection."""
    return DocumentIngestionService(repositories, document_extractor(), ollama_embedding,
                                    MAX_DOC_SIZE, MAX_DOC_CHARS)

def ingest_document(name, raw, model):
    return document_ingestion_service().ingest(name, raw, model)

def update_document_metadata(identifier, name):
    return document_ingestion_service().update_metadata(identifier, name)

def replace_document(identifier, name, raw, model, confirmed):
    return document_ingestion_service().replace(identifier, name, raw, model, confirmed)

def search_documents(query,limit=5):
    return retrieval_service().retrieve_documents(query, limit).records


def document_text(name, content):
    return document_extractor().extract(name, content)


def validate_office_archive(archive):
    """Compatibility wrapper for archive safety checks."""
    return document_extractor().validate_archive(archive)


PAGE = (BASE / 'app' / 'web' / 'templates' / 'index.html').read_text(encoding='utf-8')


class _WebDependencies:
    """Late-bound compatibility callbacks used by extracted local web handlers."""

    max_request = MAX_REQUEST
    max_attachment_chars = MAX_ATTACHMENT_CHARS
    max_document_bytes = MAX_DOC_SIZE
    embedding_model = EMBED_MODEL

    @staticmethod
    def repositories():
        return repositories()

    @staticmethod
    def ollama_client():
        return ollama_client()

    @staticmethod
    def health_status():
        return health_status()

    @staticmethod
    def rag_monitoring_status():
        return rag_monitoring_status()

    @staticmethod
    def ingest_document(name, raw, model):
        return ingest_document(name, raw, model)

    @staticmethod
    def update_document_metadata(identifier, name):
        return update_document_metadata(identifier, name)

    @staticmethod
    def replace_document(identifier, name, raw, model, confirmed):
        return replace_document(identifier, name, raw, model, confirmed)

    @staticmethod
    def document_text(name, raw):
        return document_text(name, raw)

    @staticmethod
    def chat_service():
        return chat_service()

    @staticmethod
    def log_event(level, event, **details):
        return log_event(level, event, **details)


Handler = create_handler(_WebDependencies, BASE / 'app' / 'web')

def main():
    initialize()
    try:server=create_server(HOST, PORT, Handler)
    except OSError as e:
        print('پورت در دسترس نیست. نسخه قبلی را ببندید یا OLLAMA_UI_PORT را تغییر دهید:',e)
        return
    url=f'http://{HOST}:{PORT}/'
    print('Offline Persian Company Assistant:',url)
    print('SQLite database:',DB)
    print('Stop with Ctrl+C')
    threading.Timer(0.7,lambda:webbrowser.open(url)).start()
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()
