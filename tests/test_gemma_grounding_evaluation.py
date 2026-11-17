"""Synthetic tests for direct, oracle, and RAG Gemma experiment reporting."""
import unittest

from app.evaluation.gemma_grounding_evaluation import (
    DIRECT_MESSAGES, END_TO_END_RAG, ORACLE_CONTEXT, run_gemma_experiments, unavailable_report,
)
from app.evaluation.persian_answer_dataset import ANSWER_CASES
from app.evaluation.persian_retrieval_dataset import SYNTHETIC_DOCUMENTS


class FakeGemmaClient:
    def stream_chat(self, model, messages, context_window):
        system = messages[0]['content'] if messages and messages[0]['role'] == 'system' else ''
        question = messages[-1]['content']
        if 'رمز عبور' in question:
            answer = 'اطلاعات کافی در منابع یافت نشد.'
        elif '[D1-C2]' in system:
            answer = 'تاخیر بیش از سه روز باید گزارش شود. [D1-C2]'
        elif '[D3-C1]' in system:
            answer = 'درخواست کتبی و تایید طرفین لازم است. [D3-C1]'
        elif '[D4-C1]' in system:
            answer = 'منبع فقط موانع را ذکر می‌کند. [D4-C1]'
        else:
            answer = 'پاسخ مستقیم آزمایشی.'
        yield {'message': {'content': answer}, 'done': True}


class GemmaGroundingEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.report = run_gemma_experiments(FakeGemmaClient(), 'fake-gemma', ANSWER_CASES,
                                            SYNTHETIC_DOCUMENTS, evaluation_kind='mocked_test_double')
        self.by_mode = {item['mode']: item for item in self.report['experiments']}

    def test_direct_oracle_and_rag_are_reported_separately(self):
        self.assertEqual(self.report['evaluation_kind'], 'mocked_test_double')
        self.assertEqual(set(self.by_mode), {DIRECT_MESSAGES, ORACLE_CONTEXT, END_TO_END_RAG})
        self.assertEqual(self.by_mode[DIRECT_MESSAGES]['metrics']['citation_validity_rate'], None)
        self.assertEqual(self.by_mode[ORACLE_CONTEXT]['metrics']['retrieval_success_rate'], 1.0)

    def test_rag_records_retrieval_and_unanswerable_false_positive(self):
        cases = {item['id']: item for item in self.by_mode[END_TO_END_RAG]['cases']}
        self.assertIn('D1-C2', cases['answerable_delay']['retrieved_ids'])
        self.assertFalse(cases['unanswerable_password']['retrieval_success'])
        self.assertTrue(cases['unanswerable_password']['deterministic']['abstention_correct'])
        self.assertEqual(self.by_mode[END_TO_END_RAG]['metrics']['unanswerable_false_answer_rate'], 0.0)

    def test_citation_checks_and_human_grounding_remain_separate(self):
        item = self.by_mode[ORACLE_CONTEXT]['cases'][0]
        self.assertTrue(item['deterministic']['citations_refer_to_retrieved_sources'])
        self.assertEqual(item['deterministic']['human_review']['status'], 'required')
        self.assertIsNone(item['deterministic']['human_review']['answer_supported_by_evidence'])

    def test_unavailable_report_has_no_substitute_results(self):
        report = unavailable_report('gemma4:e4b-it-qat', 'URLError')
        self.assertEqual(report['evaluation_kind'], 'real_local_gemma_unavailable')
        self.assertTrue(all(item['status'] == 'unavailable' for item in report['experiments']))
        self.assertTrue(all(item['cases'] == [] for item in report['experiments']))


if __name__ == '__main__':
    unittest.main()
