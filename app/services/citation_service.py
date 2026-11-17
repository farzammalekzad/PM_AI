"""Shared citation identifiers and evidence formatting for chat retrieval."""
import re

from app.services.answer_modes import GENERAL, GROUNDED


DOCUMENT_CITATION = re.compile(r'\[(D\d+-C\d+)\]')


def knowledge_identifier(fact):
    return 'K' + str(fact['id'])


def document_identifier(document):
    return 'D%s-C%s' % (document['document_id'], document['chunk_no'])


def source_location(document):
    """Format only persisted, reliable locations; unknown legacy provenance stays absent."""
    if document.get('page_start'):
        start, end = document['page_start'], document.get('page_end') or document['page_start']
        return 'صفحه ' + str(start) + ('' if start == end else ' تا ' + str(end))
    if document.get('worksheet'):
        start, end = document.get('row_start'), document.get('row_end')
        rows = '' if not start else '، ردیف ' + str(start) + ('' if start == end else ' تا ' + str(end))
        return 'کاربرگ ' + document['worksheet'] + rows
    if document.get('heading'):
        return 'بخش ' + document['heading']
    return ''


def evidence_items(facts, documents):
    """Return knowledge first, then document evidence in established source order."""
    items = []
    for fact in facts:
        identifier = knowledge_identifier(fact)
        items.append((identifier,
                      '[' + identifier + '] ' + fact['title'] + ' | دسته: ' + fact['category'] +
                      ' | منبع: ' + (fact['source'] or 'ثبت دستی') + '\n' + fact['content']))
    for document in documents:
        identifier = document_identifier(document)
        location = source_location(document)
        items.append((identifier, '[' + identifier + '] سند: ' + document['name'] +
                      ((' | ' + location) if location else '') + '\n' + document['content']))
    return items


def source_metadata(facts, documents):
    """Build the existing public NDJSON source fields without exposing new data."""
    return {
        'source_ids': [fact['id'] for fact in facts],
        'doc_sources': [
            '%s (%s)' % (document_identifier(document), document['name'])
            for document in documents
        ],
        'doc_source_details': [
            {'identifier': document_identifier(document), 'name': document['name'],
             'location': source_location(document)}
            for document in documents
        ],
    }


def cited_document_sources(answer, documents):
    """Return each actually cited, retrieved document source once in answer order."""
    by_identifier = {
        document_identifier(document): {
            'identifier': document_identifier(document), 'name': document['name'],
            'location': source_location(document),
        }
        for document in documents
    }
    identifiers = []
    for identifier in DOCUMENT_CITATION.findall(answer):
        if identifier in by_identifier and identifier not in identifiers:
            identifiers.append(identifier)
    return [by_identifier[identifier] for identifier in identifiers]


def citation_integrity(answer, documents, response_mode):
    """Describe citation membership without claiming factual correctness."""
    cited = cited_document_sources(answer, documents)
    if response_mode == GROUNDED:
        status = 'cited' if cited else ('insufficient_evidence' if not documents else 'missing_citation')
        required = True
    elif response_mode == GENERAL:
        status, required = 'not_required', False
    else:
        status, required = ('cited' if cited else 'not_cited'), False
    return {
        'mode': response_mode,
        'status': status,
        'citation_required': required,
        'valid_document_citation_count': len(cited),
    }
