"""
문서 분할
- RAG: 검색에 쓰일 짧은 조각. 문단·줄·문장 경계를 우선하고 긴 텍스트는 글자 단위로 자른다.
- 계약서: 제N조·문단을 검토 단위로 유지하고, 긴 조항만 추가 분할한다.
"""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CONTRACT_CHUNK_OVERLAP,
    CONTRACT_CHUNK_SIZE,
    CONTRACT_SEPARATORS,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_SEPARATORS,
)

ARTICLE_SPLIT_RE = re.compile(r"(?=제\s*\d+\s*조)")
ARTICLE_HEAD_RE = re.compile(r"제\s*\d+\s*조")


def detect_clause(text: str) -> str:
    """조각 앞부분에서 제N조 표기를 찾는다. 없으면 빈 문자열."""
    match = ARTICLE_HEAD_RE.search((text or "")[:80])
    return match.group(0).replace(" ", "") if match else ""


def _splitter(chunk_size: int, chunk_overlap: int, separators: list[str]) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
    )


def split_rag_documents(documents: list[Document]) -> list[Document]:
    splitter = _splitter(RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP, list(RAG_SEPARATORS))
    return splitter.split_documents(documents)


def split_contract_text(text: str) -> list[str]:
    """한 페이지 텍스트를 조항·문단 단위로 나눈 뒤, 긴 조각만 길이 제한으로 자른다."""
    raw = (text or "").strip()
    if not raw:
        return []

    if ARTICLE_SPLIT_RE.search(raw):
        parts = [p.strip() for p in ARTICLE_SPLIT_RE.split(raw) if p.strip()]
    else:
        parts = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
        if not parts:
            parts = [raw]

    limiter = _splitter(CONTRACT_CHUNK_SIZE, CONTRACT_CHUNK_OVERLAP, list(CONTRACT_SEPARATORS))
    units: list[str] = []
    for part in parts:
        if len(part) <= CONTRACT_CHUNK_SIZE:
            units.append(part)
        else:
            units.extend(s.strip() for s in limiter.split_text(part) if s.strip())
    return units


def page_number_from_metadata(metadata: dict | None) -> int | None:
    if not metadata:
        return None
    page = metadata.get("page")
    if isinstance(page, int):
        return page + 1
    if isinstance(page, str) and page.isdigit():
        return int(page) + 1
    return None
