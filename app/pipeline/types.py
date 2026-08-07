"""Shared datatypes for the ingestion pipeline."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class DocumentProfile:
    document_type: str  # digital_pdf | scanned_pdf | mixed | image | table_heavy
    has_bookmarks: bool = False
    has_tables: bool = False
    avg_indic_ratio: float = 0.0
    page_count: int = 0


@dataclass
class TextBlock:
    block_type: str  # paragraph | heading | list | table | qa | caption | footnote
    text: str
    page_start: int
    page_end: int
    confidence: float = 1.0
    heading_level: int | None = None
    font_size: float | None = None


@dataclass
class ParsedPage:
    page_number: int
    text: str
    confidence: float
    engine: str
    blocks: list[TextBlock] = field(default_factory=list)
    indic_ratio: float = 0.0


@dataclass
class Section:
    section_id: uuid.UUID
    title: str | None
    heading_path: list[str]
    text: str
    page_start: int
    page_end: int
    blocks: list[TextBlock] = field(default_factory=list)

    @property
    def page_number(self) -> int:
        return self.page_start


@dataclass
class ChunkDraft:
    chunk_index: int
    text: str
    section_id: uuid.UUID | None
    section_title: str | None
    heading_path: list[str]
    page_start: int | None
    page_end: int | None
    block_type: str
    token_count: int
    quality_score: float
    language: str
    text_hash: str
    ocr_confidence: float | None = None

    @property
    def page_number(self) -> int | None:
        return self.page_start
