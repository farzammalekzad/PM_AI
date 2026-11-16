"""Safe HTTP response helpers shared by local request handlers."""
import json


def send_bytes(handler, body, content_type, status=200, cache_control='no-store', extra_headers=None):
    """Write a complete local response with explicit type and length."""
    handler.send_response(status)
    handler.send_header('Content-Type', content_type)
    handler.send_header('Cache-Control', cache_control)
    handler.send_header('X-Content-Type-Options', 'nosniff')
    for name, value in (extra_headers or {}).items():
        handler.send_header(name, value)
    handler.send_header('Content-Length', str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def json_response(handler, value, status=200):
    """Write a UTF-8 JSON response without serializing secrets separately."""
    send_bytes(handler, json.dumps(value, ensure_ascii=False).encode('utf-8'),
               'application/json; charset=utf-8', status)
