"""Synthetic compatibility and security checks for local web extraction."""
import importlib.util
import io
import tempfile
import unittest
from email.message import Message
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('web_app', ROOT / 'ollama_persian_rag.py')
app = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(app)


class Capture:
    def __init__(self, path, headers=None, body=b''):
        self.path = path
        self.headers = headers or Message()
        self.rfile = io.BytesIO(body)
        self.wfile = io.BytesIO()
        self.status = None
        self.response_headers = {}

    def send_response(self, status):
        self.status = status

    def send_header(self, name, value):
        self.response_headers[name] = value

    def end_headers(self):
        pass

    def json_response(self, value, status=200):
        app.Handler.json_response(self, value, status)


class WebExtractionTests(unittest.TestCase):
    def _get(self, path):
        capture = Capture(path)
        capture._serve_page = app.Handler._serve_page.__get__(capture)
        capture._serve_static = app.Handler._serve_static.__get__(capture)
        app.Handler.do_GET(capture)
        return capture

    def test_page_and_static_assets_are_local_and_rtl(self):
        page = self._get('/')
        self.assertEqual(page.status, 200)
        body = page.wfile.getvalue().decode('utf-8')
        self.assertIn('<html lang="fa" dir="rtl">', body)
        self.assertIn('/static/styles.css', body)
        self.assertIn('/static/app.js', body)
        self.assertIn("script-src 'self'", page.response_headers['Content-Security-Policy'])
        self.assertNotIn('http://', body)
        self.assertNotIn('https://', body)
        script = self._get('/static/app.js')
        self.assertEqual(script.status, 200)
        self.assertIn('textContent', script.wfile.getvalue().decode('utf-8'))
        self.assertIn('keydown', script.wfile.getvalue().decode('utf-8'))
        styles = self._get('/static/styles.css')
        self.assertEqual(styles.response_headers['Content-Type'], 'text/css; charset=utf-8')
        self.assertIn('unicode-bidi:plaintext', styles.wfile.getvalue().decode('utf-8'))

    def test_static_paths_are_allowlisted_and_cannot_traverse(self):
        traversal = self._get('/static/../templates/index.html')
        self.assertEqual(traversal.status, 404)
        unknown = self._get('/static/missing.js')
        self.assertEqual(unknown.status, 404)

    def test_invalid_content_type_is_rejected_before_body_processing(self):
        headers = Message()
        headers['Content-Type'] = 'text/plain'
        headers['Content-Length'] = '2'
        capture = Capture('/api/knowledge-save', headers, b'{}')
        app.Handler.do_POST(capture)
        self.assertEqual(capture.status, 400)
        self.assertIn('نوع محتوای درخواست', capture.wfile.getvalue().decode('utf-8'))


if __name__ == '__main__':
    unittest.main()
