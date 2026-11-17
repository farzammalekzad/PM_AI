"""Synthetic prompt-builder and chat-orchestration regression tests."""
import unittest

from app.services.answer_modes import GROUNDED_NO_EVIDENCE
from app.services.chat_service import ChatService
from app.services.prompt_builder import PromptBuilder, TRUNCATION_MARKER
from app.services.retrieval_service import DocumentRetrievalResult, RetrievalDiagnostics


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

    def test_answer_modes_are_explicit_and_default_to_grounded(self):
        documents = [{'document_id': 3, 'chunk_no': 2, 'name': 'synthetic.txt',
                      'content': 'Synthetic document content'}]
        builder = PromptBuilder()
        default = builder.build('سؤال مصنوعی', [], documents, [], 4096)
        explicit = builder.build('سؤال مصنوعی', [], documents, [], 4096, 'grounded')
        hybrid = builder.build('سؤال مصنوعی', [], documents, [], 4096, 'hybrid')
        general = builder.build('سؤال مصنوعی', [], documents, [], 4096, 'general')
        self.assertEqual(default.messages[0]['content'], explicit.messages[0]['content'])
        self.assertIn('حالت پاسخ: مستند', explicit.messages[0]['content'])
        self.assertIn('حالت پاسخ: ترکیبی', hybrid.messages[0]['content'])
        self.assertIn('حالت پاسخ: عمومی', general.messages[0]['content'])
        self.assertIn('[D3-C2]', explicit.messages[0]['content'])


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
            {'message': {'content': 'Synthetic answer [D2-C1]'}, 'done': True},
        ])
        session = self._service(repository, client).start({
            'chat_id': None, 'question': 'Synthetic question', 'content': 'Synthetic question',
            'model': 'gemma-synthetic', 'answer_mode': 'grounded', 'ctx': 4096,
        })
        self.assertEqual(repository.new_exchanges, [])
        events = list(session.events())
        self.assertEqual(events[0]['source_ids'], [1])
        self.assertEqual(events[0]['doc_sources'], ['D2-C1 (synthetic.txt)'])
        self.assertEqual(events[1]['thinking'], 'Synthetic reasoning')
        self.assertEqual(events[2]['content'], 'Synthetic answer [D2-C1]')
        self.assertEqual(events[3]['cited_doc_sources'],
                         [{'identifier': 'D2-C1', 'name': 'synthetic.txt', 'location': ''}])
        self.assertEqual(events[4]['citation_integrity'], {
            'mode': 'grounded', 'status': 'cited', 'citation_required': True,
            'valid_document_citation_count': 1,
        })
        self.assertEqual(len(repository.new_exchanges), 1)
        self.assertEqual(repository.new_exchanges[0][3], 'Synthetic answer [D2-C1]')

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

    def test_retrieval_diagnostics_are_isolated_per_chat_session(self):
        repository = FakeRepository()
        client = FakeClient([{'message': {'content': 'answer'}, 'done': True}])

        def retrieve(question, limit):
            document_id = 10 if question == 'first' else 20
            diagnostics = RetrievalDiagnostics(
                'synthetic', True, True, False, 1, 1,
                (f'D{document_id}-C1',), 'hybrid', None, 1, ())
            return DocumentRetrievalResult(
                [{'document_id': document_id, 'chunk_no': 1,
                  'name': 'synthetic.txt', 'content': 'content'}], diagnostics)

        service = ChatService(
            lambda: repository, lambda: client, lambda question: [], retrieve,
            PromptBuilder(), (4096, 8192))
        first = service.start({'chat_id': None, 'question': 'first', 'content': 'first',
                               'model': 'gemma-synthetic', 'ctx': 4096})
        second = service.start({'chat_id': None, 'question': 'second', 'content': 'second',
                                'model': 'gemma-synthetic', 'ctx': 4096})
        first_source = next(first.events())
        second_source = next(second.events())
        self.assertEqual(first_source['retrieval_diagnostics']['selected_citation_identifiers'],
                         ['D10-C1'])
        self.assertEqual(second_source['retrieval_diagnostics']['selected_citation_identifiers'],
                         ['D20-C1'])

    def test_grounded_mode_flags_uncited_answer_and_preserves_hybrid_general_modes(self):
        request = {'chat_id': None, 'question': 'Synthetic question', 'content': 'Synthetic question',
                   'model': 'gemma-synthetic', 'answer_mode': 'grounded', 'ctx': 4096}

        grounded_repository, grounded_client = FakeRepository(), FakeClient([
            {'message': {'content': 'Unsupported synthetic answer'}, 'done': True}])
        grounded_events = list(self._service(grounded_repository, grounded_client).start(request).events())
        self.assertEqual(grounded_events[-1]['citation_integrity']['status'], 'missing_citation')
        self.assertEqual(grounded_events[-2]['cited_doc_sources'], [])

        hybrid_repository, hybrid_client = FakeRepository(), FakeClient([
            {'message': {'content': 'General synthetic answer'}, 'done': True}])
        hybrid_events = list(self._service(hybrid_repository, hybrid_client).start(
            dict(request, answer_mode='hybrid')).events())
        self.assertEqual(hybrid_events[-1]['citation_integrity']['status'], 'not_cited')
        self.assertFalse(hybrid_events[-1]['citation_integrity']['citation_required'])

        general_repository, general_client = FakeRepository(), FakeClient([
            {'message': {'content': 'General synthetic answer'}, 'done': True}])
        general_events = list(self._service(general_repository, general_client).start(
            dict(request, answer_mode='general')).events())
        self.assertEqual(general_events[-1]['citation_integrity']['status'], 'not_required')

    def test_grounded_mode_abstains_without_retrieved_documents_but_keeps_greetings(self):
        repository, client = FakeRepository(), FakeClient([
            {'message': {'content': 'Greeting answer'}, 'done': True}])
        service = ChatService(lambda: repository, lambda: client, lambda question: [],
                              lambda question, limit: [], PromptBuilder(), (4096, 8192))
        request = {'chat_id': None, 'question': 'سؤال مستند', 'content': 'سؤال مستند',
                   'model': 'gemma-synthetic', 'answer_mode': 'grounded', 'ctx': 4096}
        events = list(service.start(request).events())
        self.assertEqual(events[1]['content'], GROUNDED_NO_EVIDENCE)
        self.assertEqual(events[-1]['citation_integrity']['status'], 'insufficient_evidence')
        self.assertEqual(client.calls, [])

        greeting = service.start(dict(request, question='سلام', content='سلام'))
        list(greeting.events())
        self.assertEqual(len(client.calls), 1)

    def test_explicit_source_request_does_not_change_grounded_policy_or_retrieval(self):
        repository = FakeRepository()
        client = FakeClient([{'message': {'content': 'پاسخ [D2-C1]'}, 'done': True}])
        retrieved_questions = []

        def retrieve(question, limit):
            retrieved_questions.append((question, limit))
            return [{'document_id': 2, 'chunk_no': 1, 'name': 'synthetic.txt',
                     'content': 'Synthetic document content'}]

        service = ChatService(lambda: repository, lambda: client, lambda question: [], retrieve,
                              PromptBuilder(), (4096, 8192))
        base = {'chat_id': None, 'question': 'مسئول فعالیت کیست؟',
                'content': 'مسئول فعالیت کیست؟', 'model': 'gemma-synthetic',
                'answer_mode': 'grounded', 'ctx': 4096}
        explicit = dict(base, question='مسئول فعالیت کیست؟ فقط از اسناد استفاده کن.',
                        content='مسئول فعالیت کیست؟ فقط از اسناد استفاده کن.')
        first = service.start(base)
        second = service.start(explicit)
        self.assertEqual(first.response_mode, 'grounded')
        self.assertEqual(second.response_mode, 'grounded')
        self.assertEqual(retrieved_questions, [(base['question'], 5), (explicit['question'], 5)])
        self.assertIn('حالت پاسخ: مستند', client.calls[0][1][0]['content'])
        self.assertIn('حالت پاسخ: مستند', client.calls[1][1][0]['content'])


if __name__ == '__main__':
    unittest.main()
