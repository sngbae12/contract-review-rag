"""
RAG 서비스
- 가이드라인/약관 PDF를 PyPDFLoader로 읽고 분할한 뒤
- 요청 범위에서만 만든 임베딩 클라이언트로 Chroma에 저장한다.
- 문서 개수 조회는 로컬 Chroma만 사용하며 OpenAI 키가 필요 없다.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import uuid
from pathlib import Path
from typing import Generator, Iterable

import chromadb
from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

from config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    RETRIEVE_K,
)
from services.openai_runtime import require_api_key, user_facing_openai_error
from services.text_split import page_number_from_metadata, split_rag_documents

logger = logging.getLogger(__name__)

_store_lock = threading.RLock()
_local_client: chromadb.ClientAPI | None = None


def reset_chroma_client() -> None:
    """테스트나 경로 변경 후 로컬 클라이언트를 다시 연다."""
    global _local_client
    with _store_lock:
        _local_client = None


def _chroma_sqlite_path() -> Path:
    return Path(CHROMA_DIR) / "chroma.sqlite3"


def _make_embeddings() -> OpenAIEmbeddings:
    """요청 범위의 API 키로만 임베딩 클라이언트를 만든다. 전역에 보관하지 않는다."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=require_api_key())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_chroma_client() -> chromadb.ClientAPI | None:
    """이미 있는 DB만 연다. 파일이 없으면 클라이언트를 만들지 않는다."""
    global _local_client
    if not _chroma_sqlite_path().exists():
        return None
    with _store_lock:
        if _local_client is None:
            _local_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        return _local_client


def get_existing_collection():
    """컬렉션이 있을 때만 반환한다. 없으면 생성하지 않는다."""
    client = _local_chroma_client()
    if client is None:
        return None
    try:
        names = [item.name for item in client.list_collections()]
        if CHROMA_COLLECTION not in names:
            return None
        return client.get_collection(name=CHROMA_COLLECTION, embedding_function=None)
    except NotFoundError:
        return None
    except Exception:
        logger.warning("chroma collection lookup failed")
        return None


def rag_document_count() -> int:
    """로컬 DB의 조각 수. OpenAI API 키와 임베딩 호출이 필요 없다."""
    with _store_lock:
        collection = get_existing_collection()
        if collection is None:
            return 0
        try:
            return int(collection.count())
        except Exception:
            logger.warning("chroma count failed")
            return 0


def existing_content_hashes() -> set[str]:
    collection = get_existing_collection()
    if collection is None:
        return set()
    try:
        data = collection.get(include=["metadatas"])
    except Exception:
        logger.warning("chroma metadata read failed")
        return set()
    hashes: set[str] = set()
    for meta in data.get("metadatas") or []:
        if not meta:
            continue
        value = meta.get("content_hash")
        if value:
            hashes.add(str(value))
    return hashes


def delete_ids(ids: list[str]) -> None:
    if not ids:
        return
    collection = get_existing_collection()
    if collection is None:
        return
    try:
        collection.delete(ids=ids)
    except Exception:
        logger.warning("chroma delete failed count=%s", len(ids))


def _vectorstore_for_write() -> Chroma:
    """이번 저장 작업에서만 임베딩 클라이언트를 붙인다."""
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=_make_embeddings(),
        persist_directory=str(CHROMA_DIR),
        create_collection_if_not_exists=True,
    )


def _vectorstore_for_search() -> Chroma | None:
    if get_existing_collection() is None:
        return None
    client = _local_chroma_client()
    if client is None:
        return None
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=_make_embeddings(),
        client=client,
        create_collection_if_not_exists=False,
    )


def similarity_search(query: str, k: int = RETRIEVE_K) -> list[Document]:
    """
    유사 조각 검색.
    컬렉션이 없거나 결과가 없으면 빈 목록.
    임베딩/검색 호출 실패는 예외로 올린다.
    """
    store = _vectorstore_for_search()
    if store is None:
        return []
    return store.similarity_search(query, k=k)


def _clean_metadata(metadata: dict) -> dict:
    cleaned = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


def index_pdfs(
    file_paths: Iterable[Path],
    original_names: list[str] | None = None,
) -> Generator[dict, None, None]:
    paths = [Path(p) for p in file_paths]
    if not paths:
        yield {"type": "error", "stage": "error", "message": "업로드된 PDF가 없습니다."}
        return

    names = list(original_names or [])
    while len(names) < len(paths):
        names.append(paths[len(names)].name)

    job_id = uuid.uuid4().hex
    added_ids: list[str] = []
    committed = False

    try:
        yield {
            "type": "progress",
            "stage": "extracting",
            "message": "텍스트 추출 중입니다...",
        }

        known_hashes = existing_content_hashes()
        raw_docs: list[Document] = []
        failed_extract: list[str] = []
        skipped_duplicates: list[str] = []
        accepted_names: list[str] = []

        for path, original in zip(paths, names):
            content_hash = file_sha256(path)
            if content_hash in known_hashes:
                skipped_duplicates.append(original)
                yield {
                    "type": "progress",
                    "stage": "extracting",
                    "message": f"{original} 은(는) 이미 저장된 내용과 같아 건너뜁니다.",
                }
                continue
            try:
                loader = PyPDFLoader(str(path))
                loaded = loader.load()
            except Exception:
                logger.exception("pdf extract failed name=%s", original)
                failed_extract.append(original)
                yield {
                    "type": "progress",
                    "stage": "extracting",
                    "message": f"{original} 에서 텍스트를 읽지 못했습니다. 이 파일은 건너뜁니다.",
                }
                continue

            page_docs = 0
            for doc in loaded:
                if not doc.page_content or not doc.page_content.strip():
                    continue
                meta = dict(doc.metadata or {})
                meta["source_file"] = original
                meta["content_hash"] = content_hash
                meta["upload_job_id"] = job_id
                page = page_number_from_metadata(meta)
                if page is not None:
                    meta["page"] = page
                doc.metadata = _clean_metadata(meta)
                raw_docs.append(doc)
                page_docs += 1

            if page_docs == 0:
                failed_extract.append(original)
                yield {
                    "type": "progress",
                    "stage": "extracting",
                    "message": f"{original} 에서 텍스트를 읽지 못했습니다. 이 파일은 건너뜁니다.",
                }
                continue

            known_hashes.add(content_hash)
            accepted_names.append(original)
            yield {
                "type": "progress",
                "stage": "extracting",
                "message": f"{original} 텍스트 추출 완료 (페이지 {page_docs})",
            }

        if not raw_docs:
            if failed_extract and not skipped_duplicates:
                yield {
                    "type": "error",
                    "stage": "error",
                    "message": (
                        "PDF에서 텍스트를 읽지 못했습니다: "
                        + ", ".join(failed_extract)
                    ),
                }
                return
            collection_count = rag_document_count()
            skip_note = (
                f"이미 저장된 문서 {len(skipped_duplicates)}개를 건너뛰었습니다."
                if skipped_duplicates
                else "추가할 새 조각이 없습니다."
            )
            fail_note = (
                f"\n추출 실패: {', '.join(failed_extract)}" if failed_extract else ""
            )
            yield {
                "type": "done",
                "stage": "ready" if collection_count > 0 else "idle",
                "percent": 100,
                "message": f"{skip_note}{fail_note}\n- 총 조각: {collection_count}개",
                "file_count": 0,
                "chunk_count": 0,
                "collection_count": collection_count,
                "skipped_duplicates": skipped_duplicates,
                "failed_files": failed_extract,
            }
            committed = True
            return

        yield {
            "type": "progress",
            "stage": "splitting",
            "message": "문서를 분할하는 중입니다...",
        }
        chunks = split_rag_documents(raw_docs)
        chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
        for chunk in chunks:
            chunk.metadata = _clean_metadata(dict(chunk.metadata or {}))

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

        vectorstore = _vectorstore_for_write()
        batch_size = 20
        stored = 0

        with tqdm(total=total, desc="RAG 임베딩/저장", ncols=80) as pbar:
            for start in range(0, total, batch_size):
                batch = chunks[start : start + batch_size]
                ids = [f"{job_id}:{start + offset:06d}" for offset in range(len(batch))]
                try:
                    with _store_lock:
                        vectorstore.add_documents(batch, ids=ids)
                except Exception as exc:
                    logger.error("embedding/store failed stored=%s total=%s", stored, total)
                    delete_ids(added_ids)
                    yield {
                        "type": "error",
                        "stage": "error",
                        "message": (
                            "이번 업로드 저장에 실패해 새로 추가한 조각만 되돌렸습니다. "
                            + user_facing_openai_error(exc)
                        ),
                    }
                    return
                added_ids.extend(ids)
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

        reset_chroma_client()
        collection_count = rag_document_count()
        extra_fail = (
            f"\n추출 실패(인덱싱하지 않음): {', '.join(failed_extract)}"
            if failed_extract
            else ""
        )
        extra_skip = (
            f"\n중복 건너뜀: {', '.join(skipped_duplicates)}"
            if skipped_duplicates
            else ""
        )
        yield {
            "type": "done",
            "stage": "ready",
            "percent": 100,
            "message": (
                f"RAG 인덱싱이 완료되었습니다.\n"
                f"- 파일: {', '.join(accepted_names)}\n"
                f"- 이번 업로드 조각: {total}개\n"
                f"- 총 조각: {collection_count}개"
                f"{extra_skip}{extra_fail}"
            ),
            "file_count": len(accepted_names),
            "chunk_count": total,
            "collection_count": collection_count,
            "skipped_duplicates": skipped_duplicates,
            "failed_files": failed_extract,
        }
        committed = True
    except Exception as exc:
        logger.error("index_pdfs failed")
        if added_ids:
            delete_ids(added_ids)
        yield {
            "type": "error",
            "stage": "error",
            "message": user_facing_openai_error(exc),
        }
    finally:
        if not committed and added_ids:
            delete_ids(added_ids)
            reset_chroma_client()
