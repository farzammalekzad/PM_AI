"""Synthetic prompt-builder and chat-orchestration regression tests."""
import unittest

from app.services.chat_service import ChatService
from app.services.prompt_builder import PromptBuilder, TRUNCATION_MARKER


class FakeRepository:
    def __init__(self, exists=False, history=None):
        self.exists = exists
        self.history = history or []
        self.new_exchanges = []
        self.exchanges = []

    def chat_exists(self, identifier):
        return self.exists

    def recent_messages(self, identifier, limit):
        return list(reversed(self.history[-limit:]))

    def create_chat_with_exchange(self, *values):
        self.new_exchanges.append(values)

    def store_exchange(self, *values):
        self.exchanges.append(values)


class FakeClient:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def stream_chat(self, model, messages, context_window):
        self.calls.append((model, messages, context_window))
        return iter(self.chunks)


class PromptBuilderTests(unittest.TestCase):
    def test_budget_prioritizes_current_turn_and_orders_citations(self):
        facts = [{'id': 7, 'title': 'Synthetic fact', 'category': 'Synthetic',
                  'source': 'Synthetic source', 'content': 'F' * 5000}]
        documents = [{'document_id': 3, 'chunk_no': 2, 'name': 'synthetic.txt',
                      'content': 'D' * 5000}]
        history = [
            {'role': 'user', 'content': 'old ' * 2000},
            {'role': 'assistant', 'content': 'recent ' * 1000},
        ]
        result = PromptBuilder().build('سوال بلند ' * 2000, facts, documents, history, 4096)
        system = result.messages[0]['content']
        self.assertLess(system.index('[K7]'), system.index('[D3-C2]'))
        self.assertIn(TRUNCATION_MARKER, system)
        self.assertIn('current_user', result.diagnostics.truncated_sections)
        self.assertIn('source:K7', result.diagnostics.truncated_sections)
        self.assertIn('source:D3-C2', result.diagnostics.truncated_sections)
        self.assertLessEqual(result.diagnostics.estimated_input_chars,
                             result.diagnostics.input_budget_chars)
        self.assertTrue(result.messages[-1]['content'].startswith('سوال بلند'))

    def test_recent_history_is_preferred_and_truncation_is_diagnostic(self):
        history = [
            {'role': 'user', 'content': 'old-message ' * 1000},
            {'role': 'assistant', 'content': 'recent-message ' * 1000},
        ]
        result = PromptBuilder().build('سوال', [], [], history, 4096)
        history_text = '\n'.join(message['content'] for message in result.messages[1:-1])
        self.assertIn('recent-message', history_text)
        self.assertTrue(any(section.startswith('history:')
                            for section in result.diagnostics.truncated_sections))


class ChatServiceTests(unittest.TestCase):
    def _service(self, repository, client):
        facts = [{'id': 1, 'title': 'Synthetic fact', 'category': 'Synthetic',
                  'source': '', 'content': 'Synthetic fact content'}]
        documents = [{'document_id': 2, 'chunk_no': 1, 'name': 'synthetic.txt',
                      'content': 'Synthetic document content'}]
        return ChatService(lambda: repository, lambda: client,
                           lambda question: facts, lambda question, limit: documents,
                           PromptBuilder(), (4096, 8192))

    def test_successful_stream_emits_sources_thinking_and_persists_new_chat(self):
        repository = FakeRepository()
        client = FakeClient([
            {'message': {'thinking': 'Synthetic reasoning'}, 'done': False},
            {'message': {'content': 'Synthetic answer'}, 'done': True},
        ])
        session = self._service(repository, client).start({
            'chat_id': None, 'question': 'Synthetic question', 'content': 'Synthetic question',
            'model': 'gemma-synthetic', 'ctx': 4096,
        })
        self.assertEqual(repository.new_exchanges, [])
        events = list(session.events())
        self.assertEqual(events[0]['source_ids'], [1])
        self.assertEqual(events[0]['doc_sources'], ['D2-C1 (synthetic.txt)'])
        self.assertEqual(events[1]['thinking'], 'Synthetic reasoning')
        self.assertEqual(events[2]['content'], 'Synthetic answer')
        self.assertEqual(len(repository.new_exchanges), 1)
        self.assertEqual(repository.new_exchanges[0][3], 'Synthetic answer')

    def test_empty_or_interrupted_new_stream_does_not_create_chat(self):
        repository = FakeRepository()
        empty = FakeClient([{'message': {}, 'done': True}])
        session = self._service(repository, empty).start({
            'chat_id': None, 'question': 'Synthetic question', 'content': 'Synthetic question',
            'model': 'gemma-synthetic', 'ctx': 4096,
        })
        list(session.events())
        self.assertEqual(repository.new_exchanges, [])

        def interrupted_chunks():
            yield {'message': {'content': 'partial'}, 'done': False}
            raise ConnectionResetError('synthetic interruption')

        interrupted = FakeClient(interrupted_chunks())
        session = self._service(repository, interrupted).start({
            'chat_id': None, 'question': 'Synthetic question', 'content': 'Synthetic question',
            'model': 'gemma-synthetic', 'ctx': 4096,
        })
        with self.assertRaises(ConnectionResetError):
            list(session.events())
        self.assertEqual(repository.new_exchanges, [])

    def test_existing_history_is_prompted_and_completed_turn_is_appended(self):
        chat_id = 'a' * 32
        repository = FakeRepository(True, [
            {'role': 'user', 'content': 'Earlier question'},
            {'role': 'assistant', 'content': 'Earlier answer'},
        ])
        client = FakeClient([{'message': {'content': 'New answer'}, 'done': True}])
        session = self._service(repository, client).start({
            'chat_id': chat_id, 'question': 'New question', 'content': 'New question',
            'model': 'gemma-synthetic', 'ctx': 4096,
        })
        list(session.events())
        prompted_roles = [message['role'] for message in client.calls[0][1]]
        self.assertEqual(prompted_roles, ['system', 'user', 'assistant', 'user'])
        self.assertEqual(repository.exchanges[0][0], chat_id)
        self.assertEqual(repository.exchanges[0][2], 'New answer')


if __name__ == '__main__':
    unittest.main()
