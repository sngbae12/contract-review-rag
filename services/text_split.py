"""
문서 분할
- RAG: 검색에 쓰일 짧은 조각. 문단·줄·문장 경계를 우선하고 긴 텍스트는 글자 단위로 자른다.
- 계약서: 줄 시작의 조항 제목을 검토 단위로 유지하고, 긴 조항만 추가 분할한다.

지원하는 조항 제목(줄 시작 또는 문서 맨 앞):
- 제1조
- 제1조 목적
- 제1조 (대금)
- 제 1 조 (대금)

본문 참조는 분할하지 않는다. 예: "제2조에 따라", "제3조를 준용한다"
조항 제목을 찾지 못하면 빈 줄(문단) 기준으로 나눈다.
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

# 줄 시작의 조항 제목만. 제N조 뒤에 공백·괄호·문장부호·줄끝이어야 한다.
ARTICLE_HEAD_AT_LINE_RE = re.compile(
    r"(?:^|\n)(?P<prefix>\s*)(?P<label>제\s*\d+\s*조)(?=\s|[\(（]|[.．]|$)"
)
CLAUSE_AT_START_RE = re.compile(r"^\s*(제\s*\d+\s*조)(?=\s|[\(（]|[.．]|$)")


def normalize_clause_label(label: str) -> str:
    return re.sub(r"\s+", "", label or "")


def detect_clause(text: str) -> str:
    """조각 맨 앞이 조항 제목이면 번호를 반환한다. 없으면 빈 문자열."""
    match = CLAUSE_AT_START_RE.match(text or "")
    return normalize_clause_label(match.group(1)) if match else ""


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


def split_into_article_parts(text: str) -> list[str]:
    """줄 시작의 조항 제목에서만 나눈다. 없으면 문단 기준."""
    raw = text or ""
    if not raw.strip():
        return []

    matches = list(ARTICLE_HEAD_AT_LINE_RE.finditer(raw))
    if not matches:
        parts = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
        return parts or [raw.strip()]

    parts: list[str] = []
    head = raw[: matches[0].start()].strip()
    if head:
        parts.append(head)

    for index, match in enumerate(matches):
        start = match.start() + len(match.group("prefix"))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        piece = raw[start:end].strip()
        if piece:
            parts.append(piece)
    return parts


def split_contract_text(text: str, *, clause: str | None = None) -> list[dict]:
    """
    조항·문단 단위로 나눈 뒤, 긴 조각만 길이 제한으로 자른다.
    반환: [{"text": str, "clause": str}, ...]
    clause가 있으면 제목이 없는 조각에도 이어받는다. 원문에 번호를 다시 넣지 않는다.
    """
    raw = (text or "").strip()
    if not raw:
        return []

    parts = split_into_article_parts(raw)
    limiter = _splitter(CONTRACT_CHUNK_SIZE, CONTRACT_CHUNK_OVERLAP, list(CONTRACT_SEPARATORS))
    units: list[dict] = []
    current_clause = clause or ""

    for part in parts:
        head_clause = detect_clause(part)
        if head_clause:
            current_clause = head_clause
        part_clause = head_clause or current_clause

        if len(part) <= CONTRACT_CHUNK_SIZE:
            units.append({"text": part, "clause": part_clause})
            continue

        for piece in limiter.split_text(part):
            piece = piece.strip()
            if not piece:
                continue
            piece_clause = detect_clause(piece) or part_clause
            units.append({"text": piece, "clause": piece_clause})

    return units


def split_contract_pages(
    pages: list[tuple[str, int | None]],
    *,
    source_file: str = "",
) -> list[dict]:
    """페이지를 순서대로 나누고, 새 제목이 나오기 전까지 조항 번호를 이어받는다."""
    units: list[dict] = []
    carried_clause = ""
    for text, page in pages:
        page_units = split_contract_text(text, clause=carried_clause or None)
        for unit in page_units:
            clause = unit.get("clause") or carried_clause
            if clause:
                carried_clause = clause
            units.append(
                {
                    "text": unit["text"],
                    "clause": clause,
                    "page": page,
                    "source_file": source_file,
                }
            )
    return units


def page_number_from_metadata(metadata: dict | None) -> int | None:
    """PyPDFLoader의 0-based page를 1-based 표시 번호로 바꾼다."""
    if not metadata:
        return None
    page = metadata.get("page")
    if isinstance(page, int):
        return page + 1
    if isinstance(page, str) and page.isdigit():
        return int(page) + 1
    return None
