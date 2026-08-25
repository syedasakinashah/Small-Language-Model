"""PDF ingestion: turn uploaded files into clean, page-tagged text."""

from .loader import Document, Page, load_pdf, load_pdf_bytes

__all__ = ["Document", "Page", "load_pdf", "load_pdf_bytes"]
