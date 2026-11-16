"""Socket-level tests using only temporary SQLite data and a local mock Ollama."""
import base64
import importlib.util
import json
import tempfile
import threading
import unittest
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('http_integration_app', ROOT / 'ollama_persian_rag.py')
app = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(app)


class MockOllama(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == '/api/tags':
            self._json({'models': [{'name': 'gemma-synthetic'}, {'name': 'embed-synthetic'}]})
        else:
            self._json({'error': 'missing'}, 404)

    def do_POST(self):
        size = int(self.headers.get('Content-Length', 0))
        payload = json.loads(self.rfile.read(size) or b'{}')
        if self.path == '/api/embed':
            self._json({'embeddings': [[1.0, 0.0, 0.0] if payload['input'] else [0.0, 1.0, 0.0]]})
        elif self.path == '/api/chat':
            lines = [
                {'message': {'thinking': 'synthetic thought'}, 'done': False},
                {'message': {'content': 'پاسخ ساختگی'}, 'done': True},
            ]
            raw = ''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in lines).encode('utf-8')
            self.send_response(200); self.send_header('Content-Type', 'application/x-ndjson'); self.end_headers(); self.wfile.write(raw)
        else:
            self._json({'error': 'missing'}, 404)

    def _json(self, value, status=200):
        raw = json.dumps(value, ensure_ascii=False).encode('utf-8')
        self.send_response(status); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(raw))); self.end_headers(); self.wfile.write(raw)


class HttpIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.original = (app.DATA, app.DB, app.OLLAMA, app.EMBED_MODEL, app._WebDependencies.embedding_model)
        try:
            self.mock = ThreadingHTTPServer(('127.0.0.1', 0), MockOllama)
        except PermissionError:
            self.temp.cleanup()
            self.skipTest('the current sandbox does not permit localhost socket binding')
        self.mock_thread = threading.Thread(target=self.mock.serve_forever, daemon=True); self.mock_thread.start()
        app.DATA = Path(self.temp.name); app.DB = app.DATA / 'synthetic.sqlite3'
        app.OLLAMA = f'http://127.0.0.1:{self.mock.server_port}'
        app.EMBED_MODEL = 'embed-synthetic'; app._WebDependencies.embedding_model = 'embed-synthetic'
        app.initialize()
        self.server = app.create_server('127.0.0.1', 0, app.Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2)
        self.mock.shutdown(); self.mock.server_close(); self.mock_thread.join(timeout=2)
        app.DATA, app.DB, app.OLLAMA, app.EMBED_MODEL, app._WebDependencies.embedding_model = self.original
        self.temp.cleanup()

    def request(self, path, payload=None, content_type='application/json'):
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
        request = urllib.request.Request(self.base + path, data=data, headers={'Content-Type': content_type} if data else {})
        return urllib.request.urlopen(request, timeout=3)

    def test_complete_local_http_flow_and_stream(self):
        self.assertIn('dir="rtl"', self.request('/').read().decode('utf-8'))
        self.assertIn('textContent', self.request('/static/app.js').read().decode('utf-8'))
        self.assertTrue(json.load(self.request('/api/health'))['database']['connected'])
        self.assertIn('index', json.load(self.request('/api/rag-monitoring')))
        self.assertEqual(json.load(self.request('/api/models'))['models'][0], 'gemma-synthetic')
        fact = json.load(self.request('/api/knowledge-save', {'title': 'سیاست ساختگی', 'content': 'متن ساختگی'}))
        self.assertTrue(fact['saved'])
        raw = base64.b64encode('سند ساختگی پروژه'.encode()).decode()
        document = json.load(self.request('/api/document-add', {'name': 'synthetic.txt', 'data': raw}))
        self.assertEqual(document['chunks'], 1)
        stream = self.request('/api/chat-stream', {'chat_id': None, 'question': 'سیاست ساختگی', 'content': 'سیاست ساختگی', 'model': 'gemma-synthetic', 'ctx': 4096})
        events = [json.loads(line) for line in stream.read().decode('utf-8').splitlines()]
        self.assertIn(1, events[0]['source_ids'])
        self.assertEqual(events[1]['thinking'], 'synthetic thought')
        self.assertEqual(events[2]['content'], 'پاسخ ساختگی')

    def test_invalid_requests_concurrency_and_shutdown(self):
        with self.assertRaises(Exception):
            self.request('/static/../templates/index.html')
        with self.assertRaises(Exception):
            self.request('/api/knowledge-save', {'title': 'x'}, 'text/plain')
        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(lambda _: json.load(self.request('/api/health'))['database']['connected'], range(4)))
        self.assertEqual(responses, [True] * 4)


if __name__ == '__main__':
    unittest.main()
