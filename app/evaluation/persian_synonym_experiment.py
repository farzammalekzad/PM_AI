"""Opt-in, synthetic-only Persian synonym experiment; not used by production retrieval."""
from app.services.retrieval_service import normalize_persian, tokenize_persian


SYNONYM_GROUPS = (
    frozenset(('دیرکرد', 'تاخیر', 'تأخیر')),
    frozenset(('صورتجلسه', 'صورت‌جلسه')),
    frozenset(('تایید', 'تأیید')),
)


def experimental_normalize(text):
    """Canonicalize only the explicitly reviewed multiword meeting-minutes variant."""
    return normalize_persian(text).replace('صورت جلسه', 'صورتجلسه')


def experimental_terms(text):
    """Expand a deliberately small, reviewed set of query terms for measurement only."""
    terms = tokenize_persian(experimental_normalize(text))
    expanded = set(terms)
    for group in SYNONYM_GROUPS:
        if terms & group:
            expanded.update(group)
    return expanded
