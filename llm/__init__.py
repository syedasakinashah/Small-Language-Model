"""Answer generation: pick the best available engine and speak like a tutor."""

from .backends import Backend, get_backend, backend_status
from .tutor import Tutor

__all__ = ["Backend", "get_backend", "backend_status", "Tutor"]
