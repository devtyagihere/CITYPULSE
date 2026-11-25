"""Backward-compatible re-export of the FastAPI app from app.py."""

from app import (
    RefreshRequest,
    app,
    get_snapshot,
    health,
    refresh_snapshot,
    serve_dashboard,
)

__all__ = [
    "app",
    "RefreshRequest",
    "serve_dashboard",
    "health",
    "get_snapshot",
    "refresh_snapshot",
]
