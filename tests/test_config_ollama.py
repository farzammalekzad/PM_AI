"""Synthetic configuration, Ollama client, and embedding diagnostics tests."""
import io
import urllib.error
import unittest
from pathlib import Path
from unittest import mock

from app.config import load_config
from app.services.diagnostics import embedding_diagnostics
from app.services.ollama_client import OllamaClient, OllamaResponseError


class MockResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *unused):
        return False


class ConfigurationAndOllamaTests(unittest.TestCase):
    def test_environment_configuration_preserves_existing_variables(self):
        config = load_config(Path('/synthetic/project'), {
            'OLLAMA_URL': 'http://127.0.0.1:19999/',
            'OLLAMA_UI_PORT': '9000',
            'OLLAMA_EMBED_MODEL': 'qwen3-synthetic',
        })
        self.assertEqual(config.ollama_url, 'http://127.0.0.1:19999')
        self.assertEqual(config.port, 9000)
        self.assertEqual(config.embedding_model, 'qwen3-synthetic')
        self.assertEqual(config.database_path, Path('/synthetic/project/ollama_company_data/company_memory.sqlite3'))
        self.assertEqual(config.host, '127.0.0.1')

    def test_invalid_url_and_timeout_are_reported_as_unavailable(self):
        self.assertFalse(OllamaClient('not a URL').is_available())
        client = OllamaClient('http://127.0.0.1:11434')
        with mock.patch('app.services.ollama_client.urllib.request.urlopen', side_effect=TimeoutError):
            self.assertFalse(client.is_available())
            with self.assertRaises(TimeoutError):
                client.list_models()

    def test_model_listing_and_malformed_json(self):
        client = OllamaClient('http://127.0.0.1:11434')
        with mock.patch('app.services.ollama_client.urllib.request.urlopen',
                        return_value=MockResponse(b'{"models":[{"name":"gemma4"}]}')):
            self.assertEqual(client.list_models(), ['gemma4'])
        with mock.patch('app.services.ollama_client.urllib.request.urlopen',
                        return_value=MockResponse(b'not-json')):
            with self.assertRaises(OllamaResponseError):
                client.list_models()
        with mock.patch('app.services.ollama_client.urllib.request.urlopen',
                        return_value=MockResponse(b'[]')):
            with self.assertRaises(OllamaResponseError):
                client.list_models()

    def test_embedding_rejects_invalid_vectors(self):
        client = OllamaClient('http://127.0.0.1:11434')
        for payload in (b'{"embeddings":[[0,0]]}', b'{"embeddings":[["bad"]]}', b'{"embeddings":[]}'):
            with mock.patch('app.services.ollama_client.urllib.request.urlopen',
                            return_value=MockResponse(payload)):
                with self.assertRaises(OllamaResponseError):
                    client.embed('synthetic', 'qwen3-synthetic')
        http_error = urllib.error.HTTPError(
            'http://127.0.0.1:11434/api/embed', 404, 'not found', {}, io.BytesIO(b'model missing'))
        with mock.patch('app.services.ollama_client.urllib.request.urlopen', side_effect=http_error):
            with self.assertRaises(OllamaResponseError):
                client.embed('synthetic', 'qwen3-synthetic')

    def test_stream_rejects_malformed_ndjson_and_propagates_interruption(self):
        client = OllamaClient('http://127.0.0.1:11434')
        with mock.patch('app.services.ollama_client.urllib.request.urlopen',
                        return_value=MockResponse(b'not-json\n')):
            with self.assertRaises(OllamaResponseError):
                list(client.stream_chat('gemma4', [], 4096))

        class InterruptedResponse:
            def __enter__(self):
                return self

            def __exit__(self, *unused):
                return False

            def __iter__(self):
                raise ConnectionResetError('synthetic interruption')
                yield b''

        with mock.patch('app.services.ollama_client.urllib.request.urlopen',
                        return_value=InterruptedResponse()):
            with self.assertRaises(ConnectionResetError):
                list(client.stream_chat('gemma4', [], 4096))

    def test_embedding_diagnostics_reports_model_and_fallback_state(self):
        class AvailableClient:
            def list_models(self):
                return ['qwen3-synthetic']

            def embed(self, text, model):
                self.assertion = (text, model)
                return [0.6, 0.8]

        available = embedding_diagnostics(AvailableClient(), 'qwen3-synthetic')
        self.assertTrue(available['available'])
        self.assertEqual(available['dimension'], 2)
        self.assertFalse(available['lexical_fallback'])

        class MissingClient:
            def list_models(self):
                return []

        missing = embedding_diagnostics(MissingClient(), 'qwen3-synthetic')
        self.assertEqual(missing['status'], 'unavailable')
        self.assertEqual(missing['error'], 'model_not_found')
        self.assertTrue(missing['lexical_fallback'])

        class FailingClient:
            def list_models(self):
                raise TimeoutError('synthetic timeout')

        failed = embedding_diagnostics(FailingClient(), 'qwen3-synthetic')
        self.assertEqual(failed['status'], 'unavailable')
        self.assertEqual(failed['error'], 'TimeoutError')


if __name__ == '__main__':
    unittest.main()
