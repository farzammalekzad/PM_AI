"""Offline application configuration loaded from the established environment variables."""
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Immutable runtime configuration with safe localhost defaults."""

    base_directory: Path
    data_directory: Path
    database_path: Path
    host: str
    port: int
    ollama_url: str
    embedding_model: str
    max_request_bytes: int
    max_attachment_chars: int
    max_context_chars: int
    max_document_bytes: int
    max_document_chars: int
    max_archive_members: int
    max_archive_uncompressed_bytes: int
    ollama_tags_timeout: int
    ollama_health_timeout: int
    ollama_embedding_timeout: int
    ollama_chat_timeout: int
    allowed_context_windows: tuple
    log_max_bytes: int
    log_backup_count: int


def load_config(base_directory=None, environment=None):
    """Load configuration without network or filesystem writes."""
    environment = os.environ if environment is None else environment
    base_directory = Path(base_directory) if base_directory is not None else Path(__file__).resolve().parents[1]
    data_directory = base_directory / 'ollama_company_data'
    return AppConfig(
        base_directory=base_directory,
        data_directory=data_directory,
        database_path=data_directory / 'company_memory.sqlite3',
        host='127.0.0.1',
        port=int(environment.get('OLLAMA_UI_PORT', '8765')),
        ollama_url=environment.get('OLLAMA_URL', 'http://127.0.0.1:11434').rstrip('/'),
        embedding_model=environment.get('OLLAMA_EMBED_MODEL', '').strip(),
        max_request_bytes=15 * 1024 * 1024,
        max_attachment_chars=20_000,
        max_context_chars=13_000,
        max_document_bytes=8 * 1024 * 1024,
        max_document_chars=1_000_000,
        max_archive_members=3_000,
        max_archive_uncompressed_bytes=24 * 1024 * 1024,
        ollama_tags_timeout=8,
        ollama_health_timeout=3,
        ollama_embedding_timeout=240,
        ollama_chat_timeout=1_200,
        allowed_context_windows=(4096, 8192, 16384, 32768),
        log_max_bytes=1 * 1024 * 1024,
        log_backup_count=3,
    )
