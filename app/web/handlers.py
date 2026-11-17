"""Compatibility-preserving local HTTP routes for the offline application."""
import base64
import json
import re
import xml.etree.ElementTree as element_tree
import zipfile
from datetime import datetime
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from app.web.responses import json_response, send_bytes


STATIC_TYPES = {
    'styles.css': 'text/css; charset=utf-8',
    'app.js': 'application/javascript; charset=utf-8',
    'fonts/Vazir.woff2': 'font/woff2',
    'fonts/Vazir-Bold.woff2': 'font/woff2',
}
POST_PATHS = frozenset({
    '/api/knowledge-save', '/api/knowledge-delete', '/api/chat-stream', '/api/extract',
    '/api/document-add', '/api/document-delete', '/api/document-delete-confirmed',
    '/api/document-metadata', '/api/document-replace',
})
PAGE_CSP = "default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'"


def create_handler(dependencies, asset_root):
    """Build a handler class from application callbacks without importing the entry point."""
    asset_root = Path(asset_root)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def json_response(self, result, code=200):
            json_response(self, result, code)

        def _serve_static(self, path):
            name = path.removeprefix('/static/')
            if name not in STATIC_TYPES:
                self.json_response({'error': 'مسیر نامعتبر است'}, 404)
                return
            target = asset_root / 'static' / name
            send_bytes(self, target.read_bytes(), STATIC_TYPES[name], cache_control='public, max-age=3600')

        def _serve_page(self):
            raw = (asset_root / 'templates' / 'index.html').read_bytes()
            send_bytes(self, raw, 'text/html; charset=utf-8', cache_control='no-store',
                       extra_headers={'Content-Security-Policy': PAGE_CSP})

        def do_GET(self):
            path = self.path.split('?', 1)[0]
            try:
                if path == '/':
                    self._serve_page()
                elif path.startswith('/static/'):
                    self._serve_static(path)
                elif path == '/api/health':
                    self.json_response(dependencies.health_status())
                elif path == '/api/rag-monitoring':
                    self.json_response(dependencies.rag_monitoring_status())
                elif path == '/api/models':
                    self.json_response({'models': dependencies.ollama_client().list_models()})
                elif path == '/api/documents':
                    self.json_response({'documents': [dict(row) for row in dependencies.repositories().list_documents()]})
                elif path == '/api/knowledge':
                    self.json_response({'facts': [dict(row) for row in dependencies.repositories().list_knowledge()]})
                elif path == '/api/chats':
                    self.json_response({'chats': [dict(row) for row in dependencies.repositories().list_chats()]})
                elif path == '/api/chat-history':
                    identifier = parse_qs(urlsplit(self.path).query).get('id', [''])[0]
                    if not re.fullmatch(r'[a-f0-9]{32}', identifier):
                        raise ValueError('شناسه گفت‌وگو نامعتبر است')
                    rows = dependencies.repositories().chat_history(identifier)
                    self.json_response({'messages': [dict(row) for row in rows]})
                else:
                    self.json_response({'error': 'مسیر نامعتبر است'}, 404)
            except Exception as error:
                dependencies.log_event('error', 'api_get_failed', error_type=type(error).__name__, path=path)
                self.json_response({'error': 'خطای داخلی سرور'}, 502)

        def do_POST(self):
            path = self.path.split('?', 1)[0]
            if path not in POST_PATHS:
                self.json_response({'error': 'مسیر نامعتبر است'}, 404)
                return
            try:
                if self.headers.get_content_type() != 'application/json':
                    raise ValueError('نوع محتوای درخواست نامعتبر است')
                length = int(self.headers.get('Content-Length', '0'))
                if length <= 0 or length > dependencies.max_request:
                    raise ValueError('حجم درخواست نامعتبر است')
                value = json.loads(self.rfile.read(length))
                if not isinstance(value, dict):
                    raise ValueError('درخواست نامعتبر است')
                if path == '/api/document-add':
                    raw = base64.b64decode(value['data'], validate=True)
                    self.json_response(dependencies.ingest_document(str(value.get('name', '')), raw,
                                                                    dependencies.embedding_model))
                elif path == '/api/document-metadata':
                    self.json_response(dependencies.update_document_metadata(
                        value['id'], str(value.get('name', ''))))
                elif path == '/api/document-replace':
                    raw = base64.b64decode(value['data'], validate=True)
                    self.json_response(dependencies.replace_document(
                        value['id'], str(value.get('name', '')), raw,
                        dependencies.embedding_model, value.get('confirmed')))
                elif path == '/api/document-delete-confirmed':
                    if value.get('confirmed') is not True:
                        raise ValueError('حذف سند نیازمند تأیید صریح است')
                    dependencies.repositories().delete_document(value['id'])
                    self.json_response({'deleted': True})
                elif path == '/api/document-delete':
                    # Preserve the established endpoint for local API compatibility.
                    dependencies.repositories().delete_document(value['id'])
                    self.json_response({'deleted': True})
                elif path == '/api/knowledge-save':
                    title = str(value.get('title', '')).strip()
                    category = str(value.get('category', 'عمومی')).strip() or 'عمومی'
                    content = str(value.get('content', '')).strip()
                    source = str(value.get('source', '')).strip()
                    if not title or not content or len(title) > 180 or len(category) > 100 or len(content) > 10000 or len(source) > 300:
                        raise ValueError('عنوان/متن دانش خالی یا بیش از حد طولانی است')
                    identifier = dependencies.repositories().save_knowledge(
                        value.get('id'), title, category, content, source,
                        datetime.now().isoformat(timespec='seconds'))
                    self.json_response({'id': identifier, 'saved': True})
                elif path == '/api/knowledge-delete':
                    dependencies.repositories().delete_knowledge(value['id'])
                    self.json_response({'deleted': True})
                elif path == '/api/extract':
                    raw = base64.b64decode(value['data'], validate=True)
                    if len(raw) > dependencies.max_document_bytes:
                        raise ValueError('فایل بیش از حد بزرگ است')
                    output = dependencies.document_text(str(value.get('name', '')), raw)
                    self.json_response({'text': output[:dependencies.max_attachment_chars],
                                        'truncated': len(output) > dependencies.max_attachment_chars})
                else:
                    self.chat_stream(value)
            except (ValueError, TypeError, KeyError, UnicodeError, zipfile.BadZipFile, element_tree.ParseError) as error:
                self.json_response({'error': str(error)}, 400)
            except Exception as error:
                dependencies.log_event('error', 'api_post_failed', error_type=type(error).__name__, path=path)
                self.json_response({'error': 'خطای داخلی سرور'}, 502)

        def chat_stream(self, value):
            session = dependencies.chat_service().start(value)
            diagnostics = session.diagnostics
            if diagnostics.truncated_sections:
                dependencies.log_event('info', 'prompt_truncated', section_count=len(diagnostics.truncated_sections),
                                       estimated_chars=diagnostics.estimated_input_chars,
                                       budget_chars=diagnostics.input_budget_chars)
            self.send_response(200)
            self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache, no-transform')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            def emit(payload):
                self.wfile.write((json.dumps(payload, ensure_ascii=False) + '\n').encode('utf-8'))
                self.wfile.flush()
            try:
                for event in session.events():
                    emit(event)
            except (BrokenPipeError, ConnectionResetError):
                dependencies.log_event('info', 'stream_interrupted')
            except Exception as error:
                dependencies.log_event('error', 'stream_failed', error_type=type(error).__name__)
                try:
                    emit({'error': 'خطا در دریافت پاسخ از Ollama'})
                except (BrokenPipeError, ConnectionResetError):
                    pass
    return Handler
