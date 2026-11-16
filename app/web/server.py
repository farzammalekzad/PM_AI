"""Factory for the localhost-only standard-library HTTP server."""
from http.server import ThreadingHTTPServer


def create_server(host, port, handler):
    """Create the established threaded server; binding policy stays in configuration."""
    return ThreadingHTTPServer((host, port), handler)
