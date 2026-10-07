# -*- coding: utf-8 -*-
"""
RAG 서비스
- 가이드라인/약관 PDF를 분할·임베딩 후 Chroma에 저장한다.
- 검토용 검색·준비 상태는 문서 전체가 완료된 조각만 사용한다.
- 구버전 메타데이터는 백업 후 미확인으로 표시하고, PDF 해시만으로 완료 승격하지 않는다.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import threading
import uuid
from collections import defaultdict
from datetime import datetime, timezone
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
    CHROMA_BACKUP_DIR,
    CHROMA_COLLECTION,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    RAG_UPLOAD_DIR,
    RETRIEVE_K,
    SAMPLE_DIR,
)
from services.openai_runtime import require_api_key, user_facing_openai_error
from services.text_split import page_number_from_metadata, split_rag_documents

logger = logging.getLogger(__name__)

_store_lock = threading.RLock()
_local_client: chromadb.ClientAPI | None = None
_migration_done_for: str | None = None

MIGRATION_FLAG = "migration_v1"
MIGRATION_UNVERIFIED = "unverified_needs_reindex"
SEARCH_FILTER = {"index_complete": True}
STATUS_COMPLETE = "complete"
STATUS_INCOMPLETE = "incomplete"
STATUS_UNVERIFIED = "unverified"
_NO_JOB = "__nojob__"


def reset_chroma_client() -> None:
    """테스트나 경로 변경 후 로컬 클라이언트를 다시 연다."""
    global _local_client, _migration_done_for
    with _store_lock:
        _local_client = None
        _migration_done_for = None


def _chroma_sqlite_path() -> Path:
    return Path(CHROMA_DIR) / "chroma.sqlite3"


def _make_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=require_api_key())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_chroma_client() -> chromadb.ClientAPI | None:
    global _local_client
    if not _chroma_sqlite_path().exists():
        return None
    with _store_lock:
        if _local_client is None:
            _local_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        return _local_client


def get_existing_collection():
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


def _truthy_complete(value) -> bool:
    return value is True or value == "true" or value == 1


def _has_complete_field(meta: dict | None) -> bool:
    if not meta:
        return False
    return "index_complete" in meta


def _chunk_index_status(meta: dict | None) -> str:
    """조각 단위 상태: 완료 / 미완료(인덱싱 실패) / 미확인(구버전)."""
    meta = meta or {}
    flag = str(meta.get(MIGRATION_FLAG) or "").strip()
    if flag == MIGRATION_UNVERIFIED:
        return STATUS_UNVERIFIED
    if not _has_complete_field(meta):
        return STATUS_UNVERIFIED
    if _truthy_complete(meta.get("index_complete")):
        return STATUS_COMPLETE
    return STATUS_INCOMPLETE


def _content_hash_key(meta: dict, doc_id: str) -> str:
    content_hash = str(meta.get("content_hash") or "").strip()
    return content_hash or f"__id__:{doc_id}"


def _upload_job_key(meta: dict) -> str:
    job = str(meta.get("upload_job_id") or "").strip()
    return job or _NO_JOB


def _group_chunks_by_hash(collection) -> dict[str, list[tuple[str, dict]]]:
    data = collection.get(include=["metadatas"])
    groups: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for doc_id, meta in zip(data.get("ids") or [], data.get("metadatas") or []):
        meta = dict(meta or {})
        groups[_content_hash_key(meta, doc_id)].append((doc_id, meta))
    return groups


def _group_chunks_by_hash_and_job(
    collection,
) -> dict[str, dict[str, list[tuple[str, dict, str]]]]:
    """content_hash → upload_job_id → [(id, meta, status)]."""
    data = collection.get(include=["metadatas"])
    groups: dict[str, dict[str, list[tuple[str, dict, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for doc_id, meta in zip(data.get("ids") or [], data.get("metadatas") or []):
        meta = dict(meta or {})
        content_hash = _content_hash_key(meta, doc_id)
        job_id = _upload_job_key(meta)
        status = _chunk_index_status(meta)
        groups[content_hash][job_id].append((doc_id, meta, status))
    return groups


def analyze_document_statuses(collection=None) -> dict:
    """
    저장 작업(upload_job_id) 단위로 완료를 판단한 뒤 content_hash로 묶는다.
    - 한 작업의 모든 조각이 완료면 그 작업은 usable
    - 이전 실패 조각이 같은 content_hash여도 신규 완료 작업을 무효화하지 않음
    - 미확인(구버전)과 미완료(인덱싱 실패)는 별도 집계
    """
    collection = collection or get_existing_collection()
    result = {
        "usable_hashes": set(),
        "incomplete_hashes": set(),
        "unverified_hashes": set(),
        "usable_chunk_ids": [],
        "incomplete_chunk_ids": [],
        "unverified_chunk_ids": [],
        "usable_count": 0,
        "incomplete_count": 0,
        "unverified_count": 0,
        "total_count": 0,
    }
    if collection is None:
        return result

    try:
        by_hash = _group_chunks_by_hash_and_job(collection)
    except Exception:
        logger.warning("chroma analyze failed")
        return result

    for content_hash, jobs in by_hash.items():
        complete_ids: list[str] = []
        incomplete_ids: list[str] = []
        unverified_ids: list[str] = []

        for _job_id, items in jobs.items():
            result["total_count"] += len(items)
            statuses = [status for _doc_id, _meta, status in items]
            ids = [doc_id for doc_id, _meta, _status in items]
            if statuses and all(status == STATUS_COMPLETE for status in statuses):
                complete_ids.extend(ids)
            elif statuses and all(status == STATUS_UNVERIFIED for status in statuses):
                unverified_ids.extend(ids)
            else:
                for doc_id, _meta, status in items:
                    if status == STATUS_UNVERIFIED:
                        unverified_ids.append(doc_id)
                    else:
                        incomplete_ids.append(doc_id)

        if complete_ids:
            result["usable_hashes"].add(content_hash)
            result["usable_chunk_ids"].extend(complete_ids)
            result["usable_count"] += len(complete_ids)
        if incomplete_ids:
            result["incomplete_hashes"].add(content_hash)
            result["incomplete_chunk_ids"].extend(incomplete_ids)
            result["incomplete_count"] += len(incomplete_ids)
        if unverified_ids:
            result["unverified_hashes"].add(content_hash)
            result["unverified_chunk_ids"].extend(unverified_ids)
            result["unverified_count"] += len(unverified_ids)
    return result


def rag_document_count() -> int:
    """전체 조각 수(미완료 포함). 상태 진단용."""
    with _store_lock:
        collection = get_existing_collection()
        if collection is None:
            return 0
        try:
            return int(collection.count())
        except Exception:
            logger.warning("chroma count failed")
            return 0


def usable_rag_count() -> int:
    """계약서 검토에 쓸 수 있는 완료 문서의 조각 수."""
    with _store_lock:
        ensure_legacy_migration()
        return int(analyze_document_statuses().get("usable_count") or 0)


def rag_status_summary() -> dict:
    """화면·API용 RAG 상태. ready는 usable 기준."""
    with _store_lock:
        ensure_legacy_migration()
        analysis = analyze_document_statuses()
    usable = int(analysis["usable_count"])
    incomplete = int(analysis["incomplete_count"])
    unverified = int(analysis["unverified_count"])
    total = int(analysis["total_count"])
    messages = []
    if incomplete:
        messages.append(
            f"미완료 조각 {incomplete}개가 있어 검토 검색에서 제외했습니다. "
            "같은 PDF를 다시 업로드하면 전체 인덱싱을 다시 시도할 수 있습니다."
        )
    if unverified:
        messages.append(
            f"구버전·미확인 조각 {unverified}개는 원본 PDF로 검증되지 않아 검색·중복 방지에 쓰지 않습니다. "
            "해당 가이드라인 PDF를 다시 업로드해 주세요."
        )
    return {
        "rag_ready": usable > 0,
        "usable_count": usable,
        "incomplete_count": incomplete,
        "unverified_count": unverified,
        "total_count": total,
        "message": " ".join(messages),
        "usable_hashes": set(analysis["usable_hashes"]),
        "incomplete_hashes": set(analysis["incomplete_hashes"]),
        "unverified_hashes": set(analysis["unverified_hashes"]),
    }


def existing_content_hashes() -> set[str]:
    """완전히 저장된 문서의 content_hash만 중복으로 본다."""
    with _store_lock:
        ensure_legacy_migration()
        analysis = analyze_document_statuses()
    hashes = set()
    for value in analysis["usable_hashes"]:
        if value and not str(value).startswith("__id__:"):
            hashes.add(str(value))
    return hashes


def delete_ids(ids: list[str]) -> dict:
    unique_ids = list(dict.fromkeys(ids or []))
    result = {
        "requested": len(unique_ids),
        "deleted": 0,
        "remaining": 0,
        "ok": True,
    }
    if not unique_ids:
        return result

    collection = get_existing_collection()
    if collection is None:
        return result

    try:
        collection.delete(ids=unique_ids)
    except Exception:
        logger.warning("chroma delete failed count=%s", len(unique_ids))
        result["ok"] = False

    try:
        remaining_data = collection.get(ids=unique_ids, include=[])
        remaining = list(remaining_data.get("ids") or [])
    except Exception:
        remaining = unique_ids
        result["ok"] = False

    result["remaining"] = len(remaining)
    result["deleted"] = result["requested"] - result["remaining"]
    result["ok"] = result["remaining"] == 0
    return result


def mark_ids_complete(ids: list[str]) -> bool:
    if not ids:
        return True
    collection = get_existing_collection()
    if collection is None:
        return False
    try:
        data = collection.get(ids=ids, include=["metadatas"])
        found_ids = list(data.get("ids") or [])
        metadatas = list(data.get("metadatas") or [])
        if len(found_ids) != len(ids):
            return False
        updated = []
        for meta in metadatas:
            item = dict(meta or {})
            item["index_complete"] = True
            updated.append(item)
        collection.update(ids=found_ids, metadatas=updated)
        return True
    except Exception:
        logger.warning("chroma mark complete failed count=%s", len(ids))
        return False


def _rollback_message(rollback: dict, api_message: str) -> str:
    if rollback.get("ok"):
        return (
            "이번 업로드 저장에 실패해 새로 추가하려던 조각을 되돌렸습니다. "
            + api_message
            + " 같은 파일을 다시 업로드해 전체를 인덱싱할 수 있습니다."
        )
    remaining = rollback.get("remaining", 0)
    deleted = rollback.get("deleted", 0)
    return (
        f"이번 업로드 저장에 실패했습니다. 신규 조각 중 {deleted}개는 지웠지만 "
        f"{remaining}개는 남아 있을 수 있습니다. "
        "미완료 조각은 검토 검색에 쓰이지 않습니다. 같은 파일을 다시 업로드해 주세요. "
        + api_message
    )


def _backup_chroma_dir() -> Path | None:
    if not Path(CHROMA_DIR).exists() or not _chroma_sqlite_path().exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = Path(CHROMA_BACKUP_DIR)
    backup_root.mkdir(parents=True, exist_ok=True)
    target = backup_root / f"chroma_db_{stamp}_{uuid.uuid4().hex[:6]}"
    shutil.copytree(CHROMA_DIR, target)
    return target


def _discover_pdf_hashes(roots: list[Path]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for root in roots:
        if not root or not Path(root).exists():
            continue
        for path in Path(root).rglob("*.pdf"):
            try:
                digest = file_sha256(path)
            except Exception:
                continue
            found.setdefault(digest, path)
    return found


def ensure_legacy_migration() -> dict:
    """
    구버전(index_complete 없음) 데이터를 백업 후 처리한다.
    - 원본 PDF 해시 일치만으로는 전체 조각 저장 완료를 입증할 수 없으므로 완료로 승격하지 않는다.
    - 미확인으로 표시하고 검색·중복 방지에서 제외한 뒤 재인덱싱을 안내한다.
    - OpenAI 임베딩을 호출하지 않는다.
    - 반복 실행해도 중복 생성·손상 없음
    """
    global _migration_done_for
    marker = str(Path(CHROMA_DIR).resolve())
    if _migration_done_for == marker:
        return {"skipped": True, "reason": "already_ran"}

    collection = get_existing_collection()
    if collection is None:
        _migration_done_for = marker
        return {"skipped": True, "reason": "no_collection"}

    legacy_ids: list[str] = []
    try:
        data = collection.get(include=["metadatas"])
        for doc_id, meta in zip(data.get("ids") or [], data.get("metadatas") or []):
            meta = dict(meta or {})
            if not _has_complete_field(meta) and str(meta.get(MIGRATION_FLAG) or "") != MIGRATION_UNVERIFIED:
                legacy_ids.append(doc_id)
    except Exception:
        logger.warning("chroma legacy scan failed")
        legacy_ids = []

    if not legacy_ids:
        repaired = _repair_partial_complete_documents(collection)
        _migration_done_for = marker
        return {"skipped": True, "reason": "nothing_unverified", "repaired": repaired}

    backup = _backup_chroma_dir()
    reset_chroma_client()
    collection = get_existing_collection()
    if collection is None:
        return {"ok": False, "backup": str(backup) if backup else None}

    # PDF 존재 여부는 안내 참고용. 완료 승격에는 쓰지 않는다.
    pdf_hashes = _discover_pdf_hashes([RAG_UPLOAD_DIR, SAMPLE_DIR])
    unverified_ids: list[str] = []
    unverified_hashes: list[str] = []
    pdf_present_hashes: list[str] = []

    try:
        data = collection.get(include=["metadatas"])
        for doc_id, meta in zip(data.get("ids") or [], data.get("metadatas") or []):
            meta = dict(meta or {})
            if _has_complete_field(meta):
                continue
            if str(meta.get(MIGRATION_FLAG) or "") == MIGRATION_UNVERIFIED:
                continue
            unverified_ids.append(doc_id)
            content_hash = _content_hash_key(meta, doc_id)
            if content_hash not in unverified_hashes:
                unverified_hashes.append(content_hash)
            if content_hash in pdf_hashes and not str(content_hash).startswith("__id__:"):
                if content_hash not in pdf_present_hashes:
                    pdf_present_hashes.append(content_hash)
    except Exception:
        logger.warning("chroma legacy classify failed")
        return {"ok": False, "backup": str(backup) if backup else None}

    if unverified_ids:
        data = collection.get(ids=unverified_ids, include=["metadatas"])
        updated = []
        for meta in data.get("metadatas") or []:
            item = dict(meta or {})
            item["index_complete"] = False
            item[MIGRATION_FLAG] = MIGRATION_UNVERIFIED
            updated.append(item)
        collection.update(ids=list(data.get("ids") or []), metadatas=updated)

    repaired = _repair_partial_complete_documents(collection)
    reset_chroma_client()
    _migration_done_for = marker
    return {
        "ok": True,
        "backup": str(backup) if backup else None,
        "verified_hashes": [],
        "pdf_present_but_unverified_hashes": pdf_present_hashes,
        "unverified_hashes": unverified_hashes,
        "verified_chunks": 0,
        "unverified_chunks": len(unverified_ids),
        "repaired": repaired,
    }


def _repair_partial_complete_documents(collection) -> int:
    """같은 upload_job_id 안에서 상태가 섞이면 그 작업만 미완료로 맞춘다."""
    repaired = 0
    by_hash = _group_chunks_by_hash_and_job(collection)
    for _content_hash, jobs in by_hash.items():
        for _job_id, items in jobs.items():
            statuses = [status for _doc_id, _meta, status in items]
            if STATUS_COMPLETE in statuses and (
                STATUS_INCOMPLETE in statuses or STATUS_UNVERIFIED in statuses
            ):
                ids = [doc_id for doc_id, _meta, _status in items]
                data = collection.get(ids=ids, include=["metadatas"])
                updated = []
                for meta in data.get("metadatas") or []:
                    item = dict(meta or {})
                    item["index_complete"] = False
                    if str(item.get(MIGRATION_FLAG) or "") != MIGRATION_UNVERIFIED:
                        item["index_note"] = "partial_job_downgraded"
                    updated.append(item)
                collection.update(ids=list(data.get("ids") or []), metadatas=updated)
                repaired += len(ids)
    return repaired


def purge_superseded_chunks(keep_ids: list[str]) -> dict:
    """신규 완료 조각과 같은 content_hash의 이전 실패·구버전 조각을 삭제한다."""
    result = {"deleted": 0, "ok": True, "remaining_stale": 0}
    unique_keep = list(dict.fromkeys(keep_ids or []))
    if not unique_keep:
        return result
    collection = get_existing_collection()
    if collection is None:
        return result
    try:
        kept = collection.get(ids=unique_keep, include=["metadatas"])
        keep_set = set(kept.get("ids") or [])
        target_hashes: set[str] = set()
        for _doc_id, meta in zip(kept.get("ids") or [], kept.get("metadatas") or []):
            meta = dict(meta or {})
            content_hash = str(meta.get("content_hash") or "").strip()
            if content_hash:
                target_hashes.add(content_hash)
        if not target_hashes:
            return result
        all_data = collection.get(include=["metadatas"])
        stale_ids: list[str] = []
        for doc_id, meta in zip(all_data.get("ids") or [], all_data.get("metadatas") or []):
            if doc_id in keep_set:
                continue
            meta = dict(meta or {})
            content_hash = str(meta.get("content_hash") or "").strip()
            if content_hash in target_hashes:
                stale_ids.append(doc_id)
        if not stale_ids:
            return result
        deleted = delete_ids(stale_ids)
        result["deleted"] = int(deleted.get("deleted") or 0)
        result["remaining_stale"] = int(deleted.get("remaining") or 0)
        result["ok"] = bool(deleted.get("ok"))
        return result
    except Exception:
        logger.warning("purge superseded chunks failed")
        result["ok"] = False
        return result


def _vectorstore_for_write() -> Chroma:
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=_make_embeddings(),
        persist_directory=str(CHROMA_DIR),
        create_collection_if_not_exists=True,
    )


def _vectorstore_for_search() -> Chroma | None:
    if get_existing_collection() is None:
        return None
    if usable_rag_count() == 0:
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
    """완료된 문서 조각만 검색한다."""
    with _store_lock:
        ensure_legacy_migration()
    store = _vectorstore_for_search()
    if store is None:
        return []
    # 여유 있게 가져온 뒤, 문서 단위로 다시 걸러 일부 완료 표시 오용을 막는다.
    fetch_k = max(k * 3, k)
    docs = store.similarity_search(query, k=fetch_k, filter=SEARCH_FILTER)
    usable = existing_content_hashes()
    filtered = []
    for doc in docs:
        meta = doc.metadata or {}
        if not _truthy_complete(meta.get("index_complete")):
            continue
        content_hash = str(meta.get("content_hash") or "")
        if content_hash and content_hash not in usable:
            continue
        if not content_hash:
            continue
        filtered.append(doc)
        if len(filtered) >= k:
            break
    return filtered


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
    failure_reported = False

    try:
        ensure_legacy_migration()
        yield {
            "type": "progress",
            "stage": "extracting",
            "message": "텍스트 추출 중입니다...",
        }

        known_hashes = existing_content_hashes()
        batch_seen_hashes: set[str] = set()
        raw_docs: list[Document] = []
        failed_extract: list[str] = []
        skipped_duplicates: list[str] = []
        accepted_names: list[str] = []

        for path, original in zip(paths, names):
            content_hash = file_sha256(path)
            if content_hash in known_hashes or content_hash in batch_seen_hashes:
                skipped_duplicates.append(original)
                reason = (
                    "이미 저장된 내용과 같아"
                    if content_hash in known_hashes
                    else "이번 업로드 안에서 같은 내용이 있어"
                )
                yield {
                    "type": "progress",
                    "stage": "extracting",
                    "message": f"{original} 은(는) {reason} 건너뜁니다.",
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
            page_local: list[Document] = []
            for doc in loaded:
                if not doc.page_content or not doc.page_content.strip():
                    continue
                meta = dict(doc.metadata or {})
                meta["source_file"] = original
                meta["content_hash"] = content_hash
                meta["upload_job_id"] = job_id
                meta["index_complete"] = False
                page = page_number_from_metadata(meta)
                if page is not None:
                    meta["page"] = page
                doc.metadata = _clean_metadata(meta)
                page_local.append(doc)
                page_docs += 1

            if page_docs == 0:
                failed_extract.append(original)
                yield {
                    "type": "progress",
                    "stage": "extracting",
                    "message": f"{original} 에서 텍스트를 읽지 못했습니다. 이 파일은 건너뜁니다.",
                }
                continue

            # 추출에 성공한 뒤에만 배치 내 중복으로 본다. 완료 문서 해시는 저장 성공 후에만 known에 반영된다.
            batch_seen_hashes.add(content_hash)
            accepted_names.append(original)
            raw_docs.extend(page_local)
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
                failure_reported = True
                return
            summary = rag_status_summary()
            collection_count = summary["usable_count"]
            skip_note = (
                f"중복으로 건너뛴 파일 {len(skipped_duplicates)}개: "
                + ", ".join(skipped_duplicates)
                if skipped_duplicates
                else "추가할 새 조각이 없습니다."
            )
            fail_note = (
                f"\n추출 실패: {', '.join(failed_extract)}" if failed_extract else ""
            )
            extra_status = f"\n{summary['message']}" if summary.get("message") else ""
            done_event = {
                "type": "done",
                "stage": "ready" if collection_count > 0 else "idle",
                "percent": 100,
                "message": (
                    f"{skip_note}{fail_note}{extra_status}\n"
                    f"- 검토 가능 조각: {collection_count}개\n"
                    f"- 전체 조각: {summary['total_count']}개"
                ),
                "file_count": 0,
                "chunk_count": 0,
                "collection_count": collection_count,
                "usable_count": collection_count,
                "total_count": summary["total_count"],
                "skipped_duplicates": skipped_duplicates,
                "failed_files": failed_extract,
            }
            committed = True
            yield done_event
            return

        yield {
            "type": "progress",
            "stage": "splitting",
            "message": "문서를 분할하는 중입니다...",
        }
        chunks = split_rag_documents(raw_docs)
        chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
        for chunk in chunks:
            meta = dict(chunk.metadata or {})
            meta["index_complete"] = False
            chunk.metadata = _clean_metadata(meta)

        if not chunks:
            yield {"type": "error", "stage": "error", "message": "분할된 문서 조각이 없습니다."}
            failure_reported = True
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
                added_ids.extend(ids)
                try:
                    with _store_lock:
                        vectorstore.add_documents(batch, ids=ids)
                except Exception as exc:
                    logger.error("embedding/store failed stored=%s total=%s", stored, total)
                    rollback = delete_ids(added_ids)
                    reset_chroma_client()
                    failure_reported = True
                    yield {
                        "type": "error",
                        "stage": "error",
                        "message": _rollback_message(rollback, user_facing_openai_error(exc)),
                        "rollback_ok": bool(rollback.get("ok")),
                        "rollback_remaining": int(rollback.get("remaining") or 0),
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

        reset_chroma_client()
        if not mark_ids_complete(added_ids):
            rollback = delete_ids(added_ids)
            reset_chroma_client()
            failure_reported = True
            yield {
                "type": "error",
                "stage": "error",
                "message": _rollback_message(
                    rollback,
                    "저장 완료 표시에 실패했습니다.",
                ),
                "rollback_ok": bool(rollback.get("ok")),
                "rollback_remaining": int(rollback.get("remaining") or 0),
            }
            return

        # 같은 content_hash의 이전 실패·미확인 조각을 정리해 재업로드 누적을 막는다.
        purge_superseded_chunks(added_ids)
        reset_chroma_client()
        summary = rag_status_summary()
        collection_count = summary["usable_count"]
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
        extra_status = f"\n{summary['message']}" if summary.get("message") else ""
        done_event = {
            "type": "done",
            "stage": "ready" if collection_count > 0 else "idle",
            "percent": 100,
            "message": (
                f"RAG 인덱싱이 완료되었습니다.\n"
                f"- 파일: {', '.join(accepted_names)}\n"
                f"- 이번 업로드 조각: {total}개\n"
                f"- 검토 가능 조각: {collection_count}개\n"
                f"- 전체 조각: {summary['total_count']}개"
                f"{extra_skip}{extra_fail}{extra_status}"
            ),
            "file_count": len(accepted_names),
            "chunk_count": total,
            "collection_count": collection_count,
            "usable_count": collection_count,
            "total_count": summary["total_count"],
            "skipped_duplicates": skipped_duplicates,
            "failed_files": failed_extract,
        }
        committed = True
        yield done_event
    except Exception as exc:
        logger.error("index_pdfs failed")
        if not failure_reported:
            rollback = {"ok": True, "deleted": 0, "remaining": 0}
            if added_ids and not committed:
                rollback = delete_ids(added_ids)
                reset_chroma_client()
            yield {
                "type": "error",
                "stage": "error",
                "message": _rollback_message(rollback, user_facing_openai_error(exc))
                if added_ids
                else user_facing_openai_error(exc),
                "rollback_ok": bool(rollback.get("ok")),
                "rollback_remaining": int(rollback.get("remaining") or 0),
            }
    finally:
        if not committed and added_ids:
            delete_ids(added_ids)
            reset_chroma_client()