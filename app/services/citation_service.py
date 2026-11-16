"""Shared citation identifiers and evidence formatting for chat retrieval."""


def knowledge_identifier(fact):
    return 'K' + str(fact['id'])


def document_identifier(document):
    return 'D%s-C%s' % (document['document_id'], document['chunk_no'])


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
        items.append((identifier,
                      '[' + identifier + '] سند: ' + document['name'] + '\n' + document['content']))
    return items


def source_metadata(facts, documents):
    """Build the existing public NDJSON source fields without exposing new data."""
    return {
        'source_ids': [fact['id'] for fact in facts],
        'doc_sources': [
            '%s (%s)' % (document_identifier(document), document['name'])
            for document in documents
        ],
    }
