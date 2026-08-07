from app.parser.pymupdf_parser import parse_digital_pdf
from app.parser.router import detect_profile, pages_from_ocr, parse_document
from app.parser.tables import extract_tables

__all__ = [
    "detect_profile",
    "extract_tables",
    "pages_from_ocr",
    "parse_digital_pdf",
    "parse_document",
]
