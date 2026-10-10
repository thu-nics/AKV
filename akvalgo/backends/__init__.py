"""Optional HTTP helpers; the separately installed System server is never imported."""

from .chat import send_chat_request

__all__ = ["send_chat_request"]
