"""
RAG 서비스
- 가이드라인/약관 PDF를 PyPDFLoader로 읽고
- RecursiveCharacterTextSplitter로 분할한 뒤
- text-embedding-3-small 임베딩을 Chroma DB에 저장한다.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path
from typing import Generator, Iterable

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_SEPARATORS,
    RETRIEVE_K,
)
from services.openai_runtime import current_api_key, require_api_key, user_facing_openai_error

logger = logging.getLogger(__name__)

_store_lock = threading.RLock()
_vectorstore: Chroma | None = None
_cached_count = 0
_bound_key_fingerprint = ""


def _key_fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=require_api_key())


def get_vectorstore() -> Chroma:
    global _vectorstore, _bound_key_fingerprint
    key = require_api_key()
    fingerprint = _key_fingerprint(key)
    with _store_lock:
        if _vectorstore is None or _bound_key_fingerprint != fingerprint:
            _vectorstore = Chroma(
                collection_name=CHROMA_COLLECTION,
                embedding_function=_embeddings(),
                persist_directory=str(CHROMA_DIR),
            )
            _bound_key_fingerprint = fingerprint
        return _vectorstore


def get_retriever(k: int = RETRIEVE_K):
    return get_vectorstore().as_retriever(search_kwargs={"k": k})


def rag_document_count() -> int:
    global _cached_count
    try:
        if not current_api_key() and _vectorstore is None:
            return _cached_count
        with _store_lock:
            _cached_count = int(get_vectorstore()._collection.count())
        return _cached_count
    except Exception:
        logger.warning("chroma count failed; using cache=%s", _cached_count)
        return _cached_count


def _split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
        separators=RAG_SEPARATORS,
        length_function=len,
    )
    return splitter.split_documents(documents)


def index_pdfs(file_paths: Iterable[Path]) -> Generator[dict, None, None]:
    paths = [Path(p) for p in file_paths]
    if not paths:
        yield {"type": "error", "stage": "error", "message": "업로드된 PDF가 없습니다."}
        return

    try:
        raw_docs: list[Document] = []
        yield {
            "type": "progress",
            "stage": "extracting",
            "message": "텍스트 추출 중입니다...",
        }

        for path in tqdm(paths, desc="RAG PDF 로딩", ncols=80):
            try:
                loader = PyPDFLoader(str(path))
                loaded = loader.load()
            except Exception:
                logger.exception("pdf extract failed name=%s", path.name)
                yield {
                    "type": "error",
                    "stage": "error",
                    "message": f"{path.name} 에서 텍스트를 읽지 못했습니다. 파일이 손상되었거나 PDF가 아닐 수 있습니다.",
                }
                return

            for doc in loaded:
                doc.metadata["source_file"] = path.name
            raw_docs.extend(loaded)
            yield {
                "type": "progress",
                "stage": "extracting",
                "message": f"{path.name} 텍스트 추출 완료 (누적 페이지 {len(raw_docs)})",
            }

        if not raw_docs:
            yield {"type": "error", "stage": "error", "message": "PDF에서 텍스트를 읽지 못했습니다."}
            return

        yield {
            "type": "progress",
            "stage": "splitting",
            "message": "문서를 분할하는 중입니다...",
        }
        chunks = _split_documents(raw_docs)
        chunks = [c for c in chunks if c.page_content and c.page_content.strip()]

        if not chunks:
            yield {"type": "error", "stage": "error", "message": "분할된 문서 조각이 없습니다."}
            return

        total = len(chunks)
        yield {
            "type": "progress",
            "stage": "embedding",
            "current": 0,
            "total": total,
            "percent": 0,
            "message": f"임베딩 중입니다... 0/{total}",
        }

        vectorstore = get_vectorstore()
        batch_size = 20
        stored = 0

        with tqdm(total=total, desc="RAG 임베딩/저장", ncols=80) as pbar:
            for start in range(0, total, batch_size):
                batch = chunks[start : start + batch_size]
                try:
                    with _store_lock:
                        vectorstore.add_documents(batch)
                except Exception as exc:
                    logger.error("embedding/store failed stored=%s total=%s", stored, total)
                    yield {
                        "type": "error",
                        "stage": "error",
                        "message": user_facing_openai_error(exc),
                    }
                    return
                stored += len(batch)
                pbar.update(len(batch))
                yield {
                    "type": "progress",
                    "stage": "embedding",
                    "percent": int(stored / total * 100),
                    "message": f"임베딩 중입니다... {stored}/{total}",
                    "current": stored,
                    "total": total,
                }

        collection_count = rag_document_count()
        file_names = ", ".join(p.name for p in paths)
        yield {
            "type": "done",
            "stage": "ready",
            "percent": 100,
            "message": (
                f"RAG 인덱싱이 완료되었습니다.\n"
                f"- 파일: {file_names}\n"
                f"- 이번 업로드 조각: {total}개\n"
                f"- 총 조각: {collection_count}개"
            ),
            "file_count": len(paths),
            "chunk_count": total,
            "collection_count": collection_count,
        }
    except Exception as exc:
        logger.error("index_pdfs failed")
        yield {
            "type": "error",
            "stage": "error",
            "message": user_facing_openai_error(exc),
        }
