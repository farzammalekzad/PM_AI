"""Deterministic checks and human-review placeholders for synthetic Persian answers."""
import re


_CITATION = re.compile(r'\[(?:D\d+-C\d+|K\d+)\]')
_ABSTENTION = ('اطلاعات کافی', 'اطلاعی ندارم', 'در منابع', 'یافت نشد')


def cited_identifiers(answer):
    """Extract supported public source identifier syntax, preserving first occurrence."""
    return list(dict.fromkeys(item[1:-1] for item in _CITATION.findall(answer)))


def deterministic_checks(case, answer):
    """Check identifiers and expected abstention; never infer factual grounding."""
    answer = str(answer)
    citations = cited_identifiers(answer)
    retrieved = set(case['retrieved_ids'])
    invalid = [identifier for identifier in citations if identifier not in retrieved]
    abstained = any(marker in answer for marker in _ABSTENTION)
    required = case.get('required_persian_phrases', [])
    return {
        'citations': citations,
        'citations_refer_to_retrieved_sources': not invalid,
        'invalid_citations': invalid,
        'expected_abstention': case['expected_abstention'],
        'abstained': abstained,
        'abstention_correct': abstained == case['expected_abstention'],
        'required_persian_phrases': required,
        'required_persian_wording_present': all(phrase in answer for phrase in required),
        'human_review': {
            'answer_supported_by_evidence': None,
            'unsupported_factual_claims': None,
            'reviewer_notes': None,
            'status': 'required',
        },
    }


def evaluate_answers(cases, answers, evaluation_kind):
    """Evaluate supplied answers, explicitly separating deterministic and human checks."""
    results = []
    for case in cases:
        answer = answers.get(case['id'])
        if answer is None:
            results.append({'id': case['id'], 'status': 'not_run', 'answer': None})
            continue
        results.append({
            'id': case['id'], 'status': 'evaluated', 'question': case['question'],
            'retrieved_ids': case['retrieved_ids'], 'answer': answer,
            'deterministic': deterministic_checks(case, answer),
        })
    evaluated = [item for item in results if item['status'] == 'evaluated']
    return {
        'evaluation_kind': evaluation_kind,
        'synthetic': True,
        'automatic_claim': ('Citation syntax, retrieval membership, abstention markers, and required '
                            'wording are checked deterministically. They do not prove factual grounding.'),
        'metrics': {
            'cases': len(results), 'evaluated_cases': len(evaluated),
            'citation_validity_rate': sum(item['deterministic']['citations_refer_to_retrieved_sources']
                                           for item in evaluated) / max(1, len(evaluated)),
            'abstention_accuracy': sum(item['deterministic']['abstention_correct']
                                        for item in evaluated) / max(1, len(evaluated)),
            'required_wording_rate': sum(item['deterministic']['required_persian_wording_present']
                                          for item in evaluated) / max(1, len(evaluated)),
            'human_reviewed_cases': 0,
        },
        'cases': results,
    }
