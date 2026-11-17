"""Regression tests for synthetic grounded-answer evaluation boundaries."""
import unittest

from app.evaluation.answer_evaluation import deterministic_checks, evaluate_answers
from app.evaluation.persian_answer_dataset import ANSWER_CASES, MOCKED_ANSWERS


class AnswerEvaluationTests(unittest.TestCase):
    def test_citations_must_belong_to_retrieved_evidence(self):
        checks = deterministic_checks(ANSWER_CASES[0], 'پاسخ ساختگی [D9-C9]')
        self.assertFalse(checks['citations_refer_to_retrieved_sources'])
        self.assertEqual(checks['invalid_citations'], ['D9-C9'])

    def test_missing_evidence_requires_abstention_marker(self):
        case = ANSWER_CASES[-1]
        self.assertTrue(deterministic_checks(case, 'اطلاعات کافی در منابع یافت نشد.')['abstention_correct'])
        self.assertFalse(deterministic_checks(case, 'رمز نمونه است.')['abstention_correct'])

    def test_mocked_and_real_results_remain_separate(self):
        mocked = evaluate_answers(ANSWER_CASES, MOCKED_ANSWERS, 'mocked_answers')
        real_unavailable = evaluate_answers(ANSWER_CASES, {}, 'real_local_gemma_unavailable')
        self.assertEqual(mocked['evaluation_kind'], 'mocked_answers')
        self.assertEqual(real_unavailable['evaluation_kind'], 'real_local_gemma_unavailable')
        self.assertEqual(real_unavailable['metrics']['evaluated_cases'], 0)
        self.assertEqual(mocked['metrics']['human_reviewed_cases'], 0)

    def test_human_grounding_is_never_automatically_claimed(self):
        result = evaluate_answers(ANSWER_CASES[:1], MOCKED_ANSWERS, 'mocked_answers')['cases'][0]
        self.assertEqual(result['deterministic']['human_review']['status'], 'required')
        self.assertIsNone(result['deterministic']['human_review']['answer_supported_by_evidence'])

    def test_weakly_matching_source_is_not_automatically_answer_evidence(self):
        case = dict(ANSWER_CASES[-1], retrieved_ids=['D5-C1'])
        checks = deterministic_checks(case, 'رمز در منبع ثبت نشده است. [D5-C1]')
        self.assertTrue(checks['citations_refer_to_retrieved_sources'])
        self.assertEqual(checks['human_review']['status'], 'required')
        self.assertIsNone(checks['human_review']['unsupported_factual_claims'])


if __name__ == '__main__':
    unittest.main()
