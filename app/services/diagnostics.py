"""Non-sensitive diagnostics for explicitly configured local embedding models."""


def embedding_diagnostics(client, model):
    """Report model availability and lexical fallback without changing model state."""
    if not model:
        return {
            'model': '', 'available': False, 'status': 'not_configured',
            'model_exists': False, 'dimension': None, 'error': None,
            'lexical_fallback': True,
        }
    try:
        model_exists = model in client.list_models()
    except Exception as error:
        return {
            'model': model, 'available': False, 'status': 'unavailable',
            'model_exists': None, 'dimension': None, 'error': type(error).__name__,
            'lexical_fallback': True,
        }
    if not model_exists:
        return {
            'model': model, 'available': False, 'status': 'unavailable',
            'model_exists': False, 'dimension': None, 'error': 'model_not_found',
            'lexical_fallback': True,
        }
    try:
        vector = client.embed('health check', model)
    except Exception as error:
        return {
            'model': model, 'available': False, 'status': 'unavailable',
            'model_exists': True, 'dimension': None, 'error': type(error).__name__,
            'lexical_fallback': True,
        }
    return {
        'model': model, 'available': True, 'status': 'available',
        'model_exists': True, 'dimension': len(vector), 'error': None,
        'lexical_fallback': False,
    }
