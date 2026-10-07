# -*- coding: utf-8 -*-
"""미완료 검색 제외, 배치 내 중복, 구버전 DB 호환성 회귀 테스트."""

from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from config import CHROMA_COLLECTION, SAMPLE_DIR


def _events(gen):
    return list(gen)


class RagIntegrityTests(unittest.TestCase):
    def setUp(self):
        import services.rag_service as rag_service

        self.rag = rag_service
        self.tmp = Path(tempfile.mkdtemp(prefix="crr_integrity_"))
        self.chroma = self.tmp / "chroma"
        self.backup = self.tmp / "backups"
        self.uploads = self.tmp / "uploads"
        self.uploads.mkdir(parents=True)
        self.backup.mkdir(parents=True)
        self._old = {
            "CHROMA_DIR": rag_service.CHROMA_DIR,
            "CHROMA_BACKUP_DIR": rag_service.CHROMA_BACKUP_DIR,
            "RAG_UPLOAD_DIR": rag_service.RAG_UPLOAD_DIR,
            "SAMPLE_DIR": rag_service.SAMPLE_DIR,
        }
        rag_service.CHROMA_DIR = self.chroma
        rag_service.CHROMA_BACKUP_DIR = self.backup
        rag_service.RAG_UPLOAD_DIR = self.uploads
        rag_service.SAMPLE_DIR = SAMPLE_DIR
        rag_service.reset_chroma_client()

    def tearDown(self):
        for key, value in self._old.items():
            setattr(self.rag, key, value)
        self.rag.reset_chroma_client()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _seed(self, items):
        import chromadb

        client = chromadb.PersistentClient(path=str(self.chroma))
        col = client.get_or_create_collection(name=CHROMA_COLLECTION, embedding_function=None)
        ids, docs, embeds, metas = [], [], [], []
        for index, item in enumerate(items):
            ids.append(item.get("id", f"id-{index}"))
            docs.append(item["text"])
            embeds.append([[0.01 * (index + 1)] * 8][0])
            metas.append(item["meta"])
        col.add(ids=ids, documents=docs, embeddings=embeds, metadatas=metas)
        self.rag.reset_chroma_client()
        return col

    def _index(self, paths, names, split_n=None):
        fake = DeterministicFakeEmbedding(size=8)
        patches = [patch.object(self.rag, "_make_embeddings", return_value=fake)]
        if split_n is not None:

            def fake_split(docs):
                meta = dict(docs[0].metadata or {})
                return [
                    Document(page_content=f"가이드 조각 {i} 내용입니다.", metadata=dict(meta))
                    for i in range(split_n)
                ]

            patches.append(patch.object(self.rag, "split_rag_documents", side_effect=fake_split))
        with patches[0]:
            if split_n is not None:
                with patches[1]:
                    return _events(self.rag.index_pdfs(paths, names))
            return _events(self.rag.index_pdfs(paths, names))

    def test_incomplete_only_rag_not_ready(self):
        self._seed(
            [
                {
                    "id": "bad-1",
                    "text": "미완료 가이드",
                    "meta": {
                        "source_file": "x.pdf",
                        "content_hash": "hash-bad",
                        "index_complete": False,
                    },
                }
            ]
        )
        summary = self.rag.rag_status_summary()
        self.assertFalse(summary["rag_ready"])
        self.assertEqual(summary["usable_count"], 0)
        self.assertEqual(summary["incomplete_count"], 1)
        self.assertGreater(summary["total_count"], 0)

    def test_incomplete_excluded_from_search(self):
        self._seed(
            [
                {
                    "id": "bad-1",
                    "text": "미완료만 있는 고유문구 XYZUNIQUE",
                    "meta": {
                        "source_file": "x.pdf",
                        "content_hash": "hash-bad",
                        "index_complete": False,
                    },
                },
                {
                    "id": "good-1",
                    "text": "정상 완료 가이드라인 대금 지급",
                    "meta": {
                        "source_file": "g.pdf",
                        "content_hash": "hash-good",
                        "index_complete": True,
                    },
                },
            ]
        )
        fake = DeterministicFakeEmbedding(size=8)
        with patch.object(self.rag, "_make_embeddings", return_value=fake):
            docs = self.rag.similarity_search("XYZUNIQUE", k=5)
        texts = [d.page_content for d in docs]
        self.assertTrue(all("미완료" not in t for t in texts))
        for doc in docs:
            self.assertTrue(doc.metadata.get("index_complete") is True)

    def test_mixed_data_searches_only_complete_documents(self):
        self._seed(
            [
                {
                    "id": "inc-1",
                    "text": "실패조각 ALPHA",
                    "meta": {
                        "source_file": "a.pdf",
                        "content_hash": "h-inc",
                        "index_complete": False,
                    },
                },
                {
                    "id": "ok-1",
                    "text": "성공조각 BETA",
                    "meta": {
                        "source_file": "b.pdf",
                        "content_hash": "h-ok",
                        "index_complete": True,
                    },
                },
                {
                    "id": "partial-true",
                    "text": "부분완료 TRUE",
                    "meta": {
                        "source_file": "c.pdf",
                        "content_hash": "h-partial",
                        "index_complete": True,
                    },
                },
                {
                    "id": "partial-false",
                    "text": "부분완료 FALSE",
                    "meta": {
                        "source_file": "c.pdf",
                        "content_hash": "h-partial",
                        "index_complete": False,
                    },
                },
            ]
        )
        # 문서 단위: partial은 True가 있어도 미완료
        analysis = self.rag.analyze_document_statuses()
        self.assertIn("h-ok", analysis["usable_hashes"])
        self.assertIn("h-inc", analysis["incomplete_hashes"])
        self.assertIn("h-partial", analysis["incomplete_hashes"])
        self.assertNotIn("h-partial", analysis["usable_hashes"])

        fake = DeterministicFakeEmbedding(size=8)
        with patch.object(self.rag, "_make_embeddings", return_value=fake):
            docs = self.rag.similarity_search("성공조각", k=5)
        hashes = {d.metadata.get("content_hash") for d in docs}
        self.assertNotIn("h-inc", hashes)
        self.assertNotIn("h-partial", hashes)

    def test_same_upload_duplicate_content_indexed_once(self):
        sample = SAMPLE_DIR / "virtual_guideline.pdf"
        copy = self.tmp / "copy_guideline.pdf"
        shutil.copy2(sample, copy)
        events = self._index([sample, copy], ["virtual_guideline.pdf", "copy_guideline.pdf"], split_n=2)
        done = next(e for e in events if e["type"] in ("done", "error"))
        self.assertEqual(done["type"], "done")
        self.assertEqual(done["file_count"], 1)
        self.assertEqual(done["chunk_count"], 2)
        self.assertEqual(len(done.get("skipped_duplicates") or []), 1)
        self.assertEqual(self.rag.usable_rag_count(), 2)

    def test_different_files_multi_upload(self):
        g = SAMPLE_DIR / "virtual_guideline.pdf"
        c = SAMPLE_DIR / "virtual_contract.pdf"
        events = self._index([g, c], ["virtual_guideline.pdf", "virtual_contract.pdf"], split_n=2)
        # split_n applies same fake split per call to split_rag_documents once for all docs
        done = next(e for e in events if e["type"] in ("done", "error"))
        self.assertEqual(done["type"], "done")
        self.assertEqual(done["file_count"], 2)
        self.assertEqual(done["chunk_count"], 2)
        self.assertEqual(self.rag.usable_rag_count(), 2)

    def test_legacy_with_pdf_stays_unverified(self):
        sample = SAMPLE_DIR / "virtual_guideline.pdf"
        digest = self.rag.file_sha256(sample)
        shutil.copy2(sample, self.uploads / "kept.pdf")
        self._seed(
            [
                {
                    "id": "legacy-1",
                    "text": "구버전 일부 저장 조각",
                    "meta": {
                        "source_file": "kept.pdf",
                        "content_hash": digest,
                        # index_complete 필드 없음
                    },
                }
            ]
        )
        result = self.rag.ensure_legacy_migration()
        self.assertTrue(result.get("ok"))
        self.assertTrue(result.get("backup"))
        # PDF 해시만으로는 완료 승격하지 않음
        self.assertEqual(self.rag.usable_rag_count(), 0)
        self.assertNotIn(digest, self.rag.existing_content_hashes())
        summary = self.rag.rag_status_summary()
        self.assertEqual(summary["unverified_count"], 1)
        self.assertEqual(summary["incomplete_count"], 0)
        # 반복 실행
        again = self.rag.ensure_legacy_migration()
        self.assertTrue(again.get("skipped") or again.get("ok") is not False)
        self.assertEqual(self.rag.usable_rag_count(), 0)
        self.assertEqual(self.rag.rag_document_count(), 1)

    def test_legacy_unverified_excluded_and_allows_reindex(self):
        self._seed(
            [
                {
                    "id": "legacy-u",
                    "text": "원본 없는 구버전",
                    "meta": {
                        "source_file": "missing.pdf",
                        "content_hash": "deadbeef" * 8,
                    },
                }
            ]
        )
        result = self.rag.ensure_legacy_migration()
        self.assertTrue(result.get("ok"))
        summary = self.rag.rag_status_summary()
        self.assertFalse(summary["rag_ready"])
        self.assertEqual(summary["usable_count"], 0)
        self.assertGreater(summary["incomplete_count"] + summary["unverified_count"], 0)
        # 중복으로 막히지 않아야 함
        self.assertNotIn("deadbeef" * 8, self.rag.existing_content_hashes())

    def test_partial_true_does_not_mark_document_complete(self):
        self._seed(
            [
                {
                    "id": "p1",
                    "text": "조각1",
                    "meta": {
                        "source_file": "p.pdf",
                        "content_hash": "hash-p",
                        "index_complete": True,
                    },
                },
                {
                    "id": "p2",
                    "text": "조각2",
                    "meta": {
                        "source_file": "p.pdf",
                        "content_hash": "hash-p",
                        "index_complete": False,
                    },
                },
            ]
        )
        analysis = self.rag.analyze_document_statuses()
        self.assertNotIn("hash-p", analysis["usable_hashes"])
        self.assertIn("hash-p", analysis["incomplete_hashes"])
        self.assertEqual(self.rag.usable_rag_count(), 0)


class FullFlowMockTests(unittest.TestCase):
    """업로드→상태→계약서→검토→채팅 흐름(모델은 대역)."""

    def setUp(self):
        import services.rag_service as rag_service
        from app import APP_STATE, app

        self.rag = rag_service
        self.app = app
        self.APP_STATE = APP_STATE
        self.tmp = Path(tempfile.mkdtemp(prefix="crr_flow_"))
        self._old = rag_service.CHROMA_DIR
        self._old_backup = rag_service.CHROMA_BACKUP_DIR
        rag_service.CHROMA_DIR = self.tmp / "chroma"
        rag_service.CHROMA_BACKUP_DIR = self.tmp / "backups"
        rag_service.CHROMA_BACKUP_DIR.mkdir(parents=True)
        rag_service.reset_chroma_client()
        self._state = dict(APP_STATE)
        APP_STATE.update(
            {
                "rag_ready": False,
                "rag_stage": "idle",
                "rag_chunk_count": 0,
                "rag_total_count": 0,
                "rag_incomplete_count": 0,
                "rag_unverified_count": 0,
                "rag_status_message": "",
                "contract_ready": False,
                "contract_stage": "idle",
                "contract_filename": "",
                "contract_units": [],
                "reviewing": False,
            }
        )

    def tearDown(self):
        self.rag.CHROMA_DIR = self._old
        self.rag.CHROMA_BACKUP_DIR = self._old_backup
        self.rag.reset_chroma_client()
        self.APP_STATE.clear()
        self.APP_STATE.update(self._state)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_status_ready_uses_usable_not_total(self):
        import chromadb

        client = chromadb.PersistentClient(path=str(self.rag.CHROMA_DIR))
        col = client.get_or_create_collection(name=CHROMA_COLLECTION, embedding_function=None)
        col.add(
            ids=["a", "b"],
            documents=["완료", "미완료"],
            embeddings=[[0.1] * 8, [0.2] * 8],
            metadatas=[
                {"source_file": "a.pdf", "content_hash": "ha", "index_complete": True},
                {"source_file": "b.pdf", "content_hash": "hb", "index_complete": False},
            ],
        )
        self.rag.reset_chroma_client()
        client_app = self.app.test_client()
        data = client_app.get("/api/status").get_json()
        self.assertTrue(data["rag_ready"])
        self.assertEqual(data["rag_chunk_count"], 1)
        self.assertEqual(data["rag_total_count"], 2)
        self.assertEqual(data["rag_incomplete_count"], 1)

    def test_flow_with_mocked_llm(self):
        from services import review_agent
        from services.openai_runtime import bind_api_key, reset_api_key

        sample = SAMPLE_DIR / "virtual_guideline.pdf"
        fake = DeterministicFakeEmbedding(size=8)

        def fake_split(docs):
            meta = dict(docs[0].metadata or {})
            return [Document(page_content="가이드 조항 대금", metadata=dict(meta))]

        token = bind_api_key("sk-test-flow")
        try:
            with patch.object(self.rag, "_make_embeddings", return_value=fake), patch.object(
                self.rag, "split_rag_documents", side_effect=fake_split
            ):
                client_app = self.app.test_client()
                with open(sample, "rb") as handle:
                    res = client_app.post(
                        "/api/upload_rag",
                        data={
                            "files": (handle, "virtual_guideline.pdf"),
                            "openai_api_key": "sk-test-flow",
                        },
                        content_type="multipart/form-data",
                        headers={"X-OpenAI-Api-Key": "sk-test-flow"},
                    )
                    body = res.data.decode("utf-8", errors="replace")
                self.assertIn('"type": "done"', body)

                status = client_app.get("/api/status").get_json()
                self.assertTrue(status["rag_ready"])

                with open(SAMPLE_DIR / "virtual_contract.pdf", "rb") as handle:
                    cres = client_app.post(
                        "/api/upload_contract",
                        data={"file": (handle, "virtual_contract.pdf")},
                        content_type="multipart/form-data",
                    )
                    cbody = cres.data.decode("utf-8", errors="replace")
                self.assertIn('"type": "done"', cbody)
                self.assertTrue(self.APP_STATE["contract_ready"])

                with patch.object(
                    review_agent,
                    "run_analysis",
                    return_value={
                        "item_status": "reviewed",
                        "has_issue": False,
                        "revised_text": "x",
                        "reason": "ok",
                    },
                ), patch.object(
                    review_agent,
                    "similarity_search",
                    return_value=[
                        Document(
                            page_content="가이드",
                            metadata={"source_file": "g.pdf", "index_complete": True, "content_hash": "h"},
                        )
                    ],
                ):
                    rres = client_app.post(
                        "/api/review",
                        headers={"X-OpenAI-Api-Key": "sk-test-flow"},
                    )
                    rbody = rres.data.decode("utf-8", errors="replace")
                self.assertIn('"type": "done"', rbody)

                with patch("services.chat_service.ChatOpenAI") as chat_cls:
                    class FakeChunk:
                        def __init__(self, text):
                            self.content = text

                    instance = chat_cls.return_value
                    instance.stream.return_value = [FakeChunk("안녕"), FakeChunk("하세요")]
                    qres = client_app.post(
                        "/api/chat",
                        json={"question": "안녕하세요"},
                        headers={"X-OpenAI-Api-Key": "sk-test-flow"},
                    )
                    qbody = qres.data.decode("utf-8", errors="replace")
                self.assertIn('"type": "done"', qbody)
                self.assertIn("안녕", qbody)
        finally:
            reset_api_key(token)


if __name__ == "__main__":
    unittest.main()
