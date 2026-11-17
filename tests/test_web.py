"""Synthetic compatibility and security checks for local web extraction."""
import importlib.util
import io
import base64
import json
import shutil
import subprocess
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
        self.assertIn("font-src 'self'", page.response_headers['Content-Security-Policy'])
        self.assertNotIn('http://', body)
        self.assertNotIn('https://', body)
        script = self._get('/static/app.js')
        self.assertEqual(script.status, 200)
        self.assertIn('textContent', script.wfile.getvalue().decode('utf-8'))
        self.assertIn('renderCitedSources', script.wfile.getvalue().decode('utf-8'))
        self.assertIn('cited_doc_sources', script.wfile.getvalue().decode('utf-8'))
        self.assertIn('citation_integrity', script.wfile.getvalue().decode('utf-8'))
        self.assertIn('renderCitationIntegrity', script.wfile.getvalue().decode('utf-8'))
        self.assertIn('mode_diagnostics', script.wfile.getvalue().decode('utf-8'))
        self.assertIn('organizational_retrieval_executed', script.wfile.getvalue().decode('utf-8'))
        self.assertIn('document-replace', script.wfile.getvalue().decode('utf-8'))
        self.assertIn('document-delete-confirmed', script.wfile.getvalue().decode('utf-8'))
        self.assertIn('replaceConfirm', body)
        self.assertIn('answerMode', body)
        self.assertIn('مستند به اسناد شرکت', body)
        self.assertIn('keydown', script.wfile.getvalue().decode('utf-8'))
        styles = self._get('/static/styles.css')
        self.assertEqual(styles.response_headers['Content-Type'], 'text/css; charset=utf-8')
        css = styles.wfile.getvalue().decode('utf-8')
        self.assertIn('unicode-bidi:plaintext', css)
        self.assertIn('@font-face', css)
        self.assertIn('font-display:swap', css)
        self.assertIn('/static/fonts/Vazir.woff2', css)
        self.assertIn('font-family:"Vazir"', css)
        self.assertIn('ui-monospace', css)

    def test_vazir_fonts_are_explicitly_allowlisted_with_correct_mime_type(self):
        regular = self._get('/static/fonts/Vazir.woff2')
        bold = self._get('/static/fonts/Vazir-Bold.woff2')
        self.assertEqual(regular.status, 200)
        self.assertEqual(bold.status, 200)
        self.assertEqual(regular.response_headers['Content-Type'], 'font/woff2')
        self.assertEqual(bold.response_headers['Content-Type'], 'font/woff2')
        self.assertGreater(len(regular.wfile.getvalue()), 1000)

    def test_markdown_rendering_uses_safe_dom_and_rtl_citation_styles(self):
        script = self._get('/static/app.js')
        javascript = script.wfile.getvalue().decode('utf-8')
        self.assertIn('function renderMarkdown', javascript)
        self.assertIn('function appendInlineMarkdown', javascript)
        self.assertIn("document.createElement('pre')", javascript)
        self.assertIn("element.className='citation'", javascript)
        self.assertIn("identifier.dir='ltr'", javascript)
        self.assertIn('bubble.textContent=full', javascript)
        self.assertIn('renderMarkdown(bubble,full)', javascript)
        self.assertIn("thinkBox.querySelector('pre').textContent=thinking", javascript)
        self.assertNotIn('innerHTML', javascript)

        styles = self._get('/static/styles.css')
        css = styles.wfile.getvalue().decode('utf-8')
        self.assertIn('.assistant-markdown', css)
        self.assertIn('.assistant-markdown pre', css)
        self.assertIn('.citation{display:inline-block;direction:ltr;unicode-bidi:isolate;white-space:nowrap', css)
        self.assertIn('.cited-sources', css)
        self.assertIn('.citation-status', css)
        self.assertIn('font-family:"Vazir"', css)
        self.assertIn('ui-monospace', css)

    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for the browser-renderer contract test')
    def test_markdown_renderer_contract(self):
        """Exercise the actual local JavaScript renderer with synthetic untrusted content."""
        result = subprocess.run(
            ['node', str(ROOT / 'tests' / 'markdown_renderer_test.js')],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    def test_static_paths_are_allowlisted_and_cannot_traverse(self):
        traversal = self._get('/static/../templates/index.html')
        self.assertEqual(traversal.status, 404)
        unknown = self._get('/static/missing.js')
        self.assertEqual(unknown.status, 404)
        font_traversal = self._get('/static/fonts/../styles.css')
        self.assertEqual(font_traversal.status, 404)
        unknown_font = self._get('/static/fonts/unknown.woff2')
        self.assertEqual(unknown_font.status, 404)

    def test_invalid_content_type_is_rejected_before_body_processing(self):
        headers = Message()
        headers['Content-Type'] = 'text/plain'
        headers['Content-Length'] = '2'
        capture = Capture('/api/knowledge-save', headers, b'{}')
        app.Handler.do_POST(capture)
        self.assertEqual(capture.status, 400)
        self.assertIn('نوع محتوای درخواست', capture.wfile.getvalue().decode('utf-8'))

    def test_document_lifecycle_api_is_additive_and_requires_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            original = (app.DATA, app.DB, app.EMBED_MODEL, app._WebDependencies.embedding_model)
            try:
                app.DATA = Path(directory)
                app.DB = app.DATA / 'synthetic.sqlite3'
                app.EMBED_MODEL = ''
                app._WebDependencies.embedding_model = ''
                app.initialize()

                def post(path, value):
                    raw = json.dumps(value).encode('utf-8')
                    headers = Message()
                    headers['Content-Type'] = 'application/json'
                    headers['Content-Length'] = str(len(raw))
                    capture = Capture(path, headers, raw)
                    app.Handler.do_POST(capture)
                    return capture, json.loads(capture.wfile.getvalue())

                content = base64.b64encode(b'first synthetic content').decode()
                capture, added = post('/api/document-add', {'name': 'synthetic.txt', 'data': content})
                self.assertEqual(capture.status, 200)
                capture, rejected = post('/api/document-replace', {
                    'id': added['id'], 'name': 'replacement.txt', 'data': content, 'confirmed': False})
                self.assertEqual(capture.status, 400)
                self.assertIn('تأیید', rejected['error'])
                replacement = base64.b64encode(b'replacement synthetic content').decode()
                capture, replaced = post('/api/document-replace', {
                    'id': added['id'], 'name': 'replacement.txt', 'data': replacement, 'confirmed': True})
                self.assertEqual(capture.status, 200)
                self.assertEqual(replaced['id'], added['id'])
                capture, _ = post('/api/document-delete-confirmed', {'id': added['id'], 'confirmed': False})
                self.assertEqual(capture.status, 400)
                capture, deleted = post('/api/document-delete-confirmed', {'id': added['id'], 'confirmed': True})
                self.assertEqual(capture.status, 200)
                self.assertTrue(deleted['deleted'])
            finally:
                app.DATA, app.DB, app.EMBED_MODEL, app._WebDependencies.embedding_model = original


if __name__ == '__main__':
    unittest.main()
