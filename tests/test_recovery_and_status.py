# -*- coding: utf-8 -*-
"""미완료 재업로드 복구, 구버전 미확인, 상태 집계, 수정문구 미생성 회귀 테스트."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from config import CHROMA_COLLECTION, SAMPLE_DIR


def _events(gen):
    return list(gen)


class IncompleteReuploadRecoveryTests(unittest.TestCase):
    def setUp(self):
        import services.rag_service as rag_service

        self.rag = rag_service
        self.tmp = Path(tempfile.mkdtemp(prefix="crr_reupload_"))
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
            embeds.append([0.01 * (index + 1)] * 8)
            metas.append(item["meta"])
        col.add(ids=ids, documents=docs, embeddings=embeds, metadatas=metas)
        self.rag.reset_chroma_client()
        return col

    def _index(self, paths, names, split_n=3):
        fake = DeterministicFakeEmbedding(size=8)

        def fake_split(docs):
            meta = dict(docs[0].metadata or {})
            return [
                Document(page_content=f"가이드 조각 {i} 내용입니다.", metadata=dict(meta))
                for i in range(split_n)
            ]

        with patch.object(self.rag, "_make_embeddings", return_value=fake), patch.object(
            self.rag, "split_rag_documents", side_effect=fake_split
        ):
            return _events(self.rag.index_pdfs(paths, names))

    def test_failed_chunk_same_pdf_reupload_recovers_and_searchable(self):
        sample = SAMPLE_DIR / "virtual_guideline.pdf"
        digest = self.rag.file_sha256(sample)
        self._seed(
            [
                {
                    "id": "fail-old",
                    "text": "이전 실패 조각 STALEUNIQUE",
                    "meta": {
                        "source_file": "virtual_guideline.pdf",
                        "content_hash": digest,
                        "upload_job_id": "old-job",
                        "index_complete": False,
                    },
                }
            ]
        )
        before = self.rag.rag_status_summary()
        self.assertFalse(before["rag_ready"])
        self.assertEqual(before["usable_count"], 0)

        events = self._index([sample], ["virtual_guideline.pdf"], split_n=3)
        done = next(e for e in events if e["type"] in ("done", "error"))
        self.assertEqual(done["type"], "done")
        self.assertEqual(done["usable_count"], 3)
        self.assertEqual(self.rag.usable_rag_count(), 3)
        self.assertEqual(self.rag.rag_document_count(), 3)

        summary = self.rag.rag_status_summary()
        self.assertTrue(summary["rag_ready"])
        self.assertEqual(summary["incomplete_count"], 0)
        self.assertIn(digest, self.rag.existing_content_hashes())

        fake = DeterministicFakeEmbedding(size=8)
        with patch.object(self.rag, "_make_embeddings", return_value=fake):
            docs = self.rag.similarity_search("가이드 조각", k=5)
        self.assertGreater(len(docs), 0)
        self.assertTrue(all(d.metadata.get("index_complete") is True for d in docs))
        self.assertTrue(all("STALEUNIQUE" not in d.page_content for d in docs))

    def test_reupload_after_recovery_is_duplicate_without_growth(self):
        sample = SAMPLE_DIR / "virtual_guideline.pdf"
        digest = self.rag.file_sha256(sample)
        self._seed(
            [
                {
                    "id": "fail-old",
                    "text": "이전 실패",
                    "meta": {
                        "source_file": "virtual_guideline.pdf",
                        "content_hash": digest,
                        "upload_job_id": "old-job",
                        "index_complete": False,
                    },
                }
            ]
        )
        first = self._index([sample], ["virtual_guideline.pdf"], split_n=2)
        done1 = next(e for e in first if e["type"] == "done")
        self.assertEqual(done1["usable_count"], 2)
        count_after = self.rag.rag_document_count()

        second = self._index([sample], ["virtual_guideline.pdf"], split_n=2)
        done2 = next(e for e in second if e["type"] == "done")
        self.assertEqual(done2["file_count"], 0)
        self.assertEqual(len(done2.get("skipped_duplicates") or []), 1)
        self.assertEqual(self.rag.rag_document_count(), count_after)
        self.assertEqual(self.rag.usable_rag_count(), 2)

    def test_failed_reindex_preserves_other_complete_document(self):
        sample = SAMPLE_DIR / "virtual_guideline.pdf"
        other = SAMPLE_DIR / "virtual_contract.pdf"
        digest = self.rag.file_sha256(sample)
        other_hash = self.rag.file_sha256(other)
        self._seed(
            [
                {
                    "id": "ok-other",
                    "text": "다른 정상 문서 OTHERDOC",
                    "meta": {
                        "source_file": "virtual_contract.pdf",
                        "content_hash": other_hash,
                        "upload_job_id": "ok-job",
                        "index_complete": True,
                    },
                },
                {
                    "id": "fail-old",
                    "text": "실패 조각",
                    "meta": {
                        "source_file": "virtual_guideline.pdf",
                        "content_hash": digest,
                        "upload_job_id": "old-job",
                        "index_complete": False,
                    },
                },
            ]
        )
        self.assertEqual(self.rag.usable_rag_count(), 1)

        fake = DeterministicFakeEmbedding(size=8)

        def boom_add(*_a, **_k):
            raise RuntimeError("embedding failed")

        with patch.object(self.rag, "_make_embeddings", return_value=fake), patch.object(
            self.rag,
            "split_rag_documents",
            return_value=[
                Document(
                    page_content="새 조각",
                    metadata={
                        "source_file": "virtual_guideline.pdf",
                        "content_hash": digest,
                        "upload_job_id": "will-set",
                        "index_complete": False,
                    },
                )
            ],
        ), patch.object(self.rag, "_vectorstore_for_write") as vs_factory:
            store = MagicMock()
            store.add_documents.side_effect = boom_add
            vs_factory.return_value = store
            events = _events(self.rag.index_pdfs([sample], ["virtual_guideline.pdf"]))

        err = next(e for e in events if e["type"] == "error")
        self.assertEqual(err["type"], "error")
        # 다른 정상 문서는 유지
        self.assertEqual(self.rag.usable_rag_count(), 1)
        self.assertIn(other_hash, self.rag.existing_content_hashes())
        self.assertNotIn(digest, self.rag.existing_content_hashes())


class LegacyPartialAndStatusTests(unittest.TestCase):
    def setUp(self):
        import services.rag_service as rag_service

        self.rag = rag_service
        self.tmp = Path(tempfile.mkdtemp(prefix="crr_legacy_"))
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
            embeds.append([0.01 * (index + 1)] * 8)
            metas.append(item["meta"])
        col.add(ids=ids, documents=docs, embeddings=embeds, metadatas=metas)
        self.rag.reset_chroma_client()

    def test_partial_legacy_with_pdf_on_disk_not_promoted_or_skipped(self):
        sample = SAMPLE_DIR / "virtual_guideline.pdf"
        digest = self.rag.file_sha256(sample)
        shutil.copy2(sample, self.uploads / "kept.pdf")
        # 원본은 10조각이어야 하는데 DB에는 1개만 (구버전, index_complete 없음)
        self._seed(
            [
                {
                    "id": "legacy-partial",
                    "text": "구버전 일부만 저장",
                    "meta": {
                        "source_file": "kept.pdf",
                        "content_hash": digest,
                        # index_complete 없음
                    },
                }
            ]
        )
        result = self.rag.ensure_legacy_migration()
        self.assertTrue(result.get("ok"))
        self.assertTrue(result.get("backup"))
        self.assertEqual(result.get("verified_chunks"), 0)
        summary = self.rag.rag_status_summary()
        self.assertFalse(summary["rag_ready"])
        self.assertEqual(summary["usable_count"], 0)
        self.assertEqual(summary["unverified_count"], 1)
        self.assertEqual(summary["incomplete_count"], 0)
        self.assertNotIn(digest, self.rag.existing_content_hashes())

        # 재업로드가 중복으로 건너뛰지 않아야 함
        fake = DeterministicFakeEmbedding(size=8)

        def fake_split(docs):
            meta = dict(docs[0].metadata or {})
            return [
                Document(page_content=f"전체 조각 {i}", metadata=dict(meta)) for i in range(3)
            ]

        with patch.object(self.rag, "_make_embeddings", return_value=fake), patch.object(
            self.rag, "split_rag_documents", side_effect=fake_split
        ):
            events = _events(self.rag.index_pdfs([sample], ["virtual_guideline.pdf"]))
        done = next(e for e in events if e["type"] in ("done", "error"))
        self.assertEqual(done["type"], "done")
        self.assertEqual(done["file_count"], 1)
        self.assertEqual(self.rag.usable_rag_count(), 3)
        self.assertEqual(self.rag.rag_document_count(), 3)

    def test_unverified_persists_across_restart_then_reindex(self):
        sample = SAMPLE_DIR / "virtual_guideline.pdf"
        digest = self.rag.file_sha256(sample)
        shutil.copy2(sample, self.uploads / "kept.pdf")
        self._seed(
            [
                {
                    "id": "legacy-u",
                    "text": "미확인 구버전",
                    "meta": {"source_file": "kept.pdf", "content_hash": digest},
                }
            ]
        )
        self.rag.ensure_legacy_migration()
        summary1 = self.rag.rag_status_summary()
        self.assertEqual(summary1["unverified_count"], 1)
        self.assertEqual(summary1["incomplete_count"], 0)

        # 재시작 시뮬레이션
        self.rag.reset_chroma_client()
        summary2 = self.rag.rag_status_summary()
        self.assertEqual(summary2["unverified_count"], 1)
        self.assertEqual(summary2["incomplete_count"], 0)
        self.assertFalse(summary2["rag_ready"])

        fake = DeterministicFakeEmbedding(size=8)

        def fake_split(docs):
            meta = dict(docs[0].metadata or {})
            return [Document(page_content="재인덱싱 조각", metadata=dict(meta))]

        with patch.object(self.rag, "_make_embeddings", return_value=fake), patch.object(
            self.rag, "split_rag_documents", side_effect=fake_split
        ):
            events = _events(self.rag.index_pdfs([sample], ["virtual_guideline.pdf"]))
        done = next(e for e in events if e["type"] == "done")
        self.assertEqual(done["usable_count"], 1)
        summary3 = self.rag.rag_status_summary()
        self.assertTrue(summary3["rag_ready"])
        self.assertEqual(summary3["unverified_count"], 0)
        self.assertEqual(summary3["incomplete_count"], 0)

    def test_mixed_status_counts_and_search_exclusion(self):
        self._seed(
            [
                {
                    "id": "ok1",
                    "text": "완료 COMPLETEDOC",
                    "meta": {
                        "source_file": "a.pdf",
                        "content_hash": "h-ok",
                        "upload_job_id": "j1",
                        "index_complete": True,
                    },
                },
                {
                    "id": "inc1",
                    "text": "미완료 FAILDOC",
                    "meta": {
                        "source_file": "b.pdf",
                        "content_hash": "h-inc",
                        "upload_job_id": "j2",
                        "index_complete": False,
                    },
                },
                {
                    "id": "unv1",
                    "text": "미확인 LEGACYDOC",
                    "meta": {
                        "source_file": "c.pdf",
                        "content_hash": "h-unv",
                        "index_complete": False,
                        self.rag.MIGRATION_FLAG: self.rag.MIGRATION_UNVERIFIED,
                    },
                },
            ]
        )
        summary = self.rag.rag_status_summary()
        self.assertTrue(summary["rag_ready"])
        self.assertEqual(summary["usable_count"], 1)
        self.assertEqual(summary["incomplete_count"], 1)
        self.assertEqual(summary["unverified_count"], 1)
        self.assertEqual(summary["total_count"], 3)
        self.assertEqual(
            summary["usable_count"] + summary["incomplete_count"] + summary["unverified_count"],
            summary["total_count"],
        )
        self.assertEqual(self.rag.existing_content_hashes(), {"h-ok"})

        fake = DeterministicFakeEmbedding(size=8)
        with patch.object(self.rag, "_make_embeddings", return_value=fake):
            docs = self.rag.similarity_search("COMPLETEDOC", k=5)
        texts = " ".join(d.page_content for d in docs)
        self.assertIn("COMPLETEDOC", texts)
        self.assertNotIn("FAILDOC", texts)
        self.assertNotIn("LEGACYDOC", texts)

    def test_status_api_no_embedding_call(self):
        from app import APP_STATE, app

        self._seed(
            [
                {
                    "id": "unv",
                    "text": "미확인",
                    "meta": {
                        "source_file": "x.pdf",
                        "content_hash": "deadbeef" * 8,
                    },
                }
            ]
        )
        embed_calls = {"n": 0}

        def boom_embed():
            embed_calls["n"] += 1
            raise AssertionError("status must not call embeddings")

        old_state = dict(APP_STATE)
        try:
            with patch.object(self.rag, "_make_embeddings", side_effect=boom_embed):
                client = app.test_client()
                data = client.get("/api/status").get_json()
            self.assertEqual(embed_calls["n"], 0)
            self.assertFalse(data["rag_ready"])
            self.assertEqual(data["rag_unverified_count"], 1)
            self.assertEqual(data["rag_incomplete_count"], 0)
            self.assertEqual(data["rag_chunk_count"], 0)
            self.assertEqual(data["rag_total_count"], 1)
        finally:
            APP_STATE.clear()
            APP_STATE.update(old_state)


class RevisionMissingTests(unittest.TestCase):
    def test_run_analysis_keeps_issue_when_revision_same(self):
        from services import review_agent

        text = "대금을 즉시 지급한다. 이 문장은 충분히 깁니다."

        class FakeDecision:
            has_issue = True
            revised_sentence = text
            reason = "지급 기한이 없습니다."

        fake_llm = MagicMock()
        fake_llm.with_structured_output.return_value.invoke.return_value = FakeDecision()
        with patch.object(review_agent, "_llm", return_value=fake_llm):
            result = review_agent.run_analysis(text, "가이드라인 내용")

        self.assertTrue(result["has_issue"])
        self.assertTrue(result["revision_missing"])
        self.assertIn("수정문구 미생성", result["reason"])
        self.assertEqual(result["item_status"], "reviewed")

    def test_run_analysis_keeps_issue_when_revision_empty(self):
        from services import review_agent

        text = "대금을 즉시 지급한다. 이 문장은 충분히 깁니다."

        class FakeDecision:
            has_issue = True
            revised_sentence = "   "
            reason = "불리한 조항"

        fake_llm = MagicMock()
        fake_llm.with_structured_output.return_value.invoke.return_value = FakeDecision()
        with patch.object(review_agent, "_llm", return_value=fake_llm):
            result = review_agent.run_analysis(text, "가이드")

        self.assertTrue(result["has_issue"])
        self.assertTrue(result["revision_missing"])

    def test_run_analysis_no_issue_unchanged(self):
        from services import review_agent

        text = "대금을 30일 이내 지급한다. 이 문장은 충분히 깁니다."

        class FakeDecision:
            has_issue = False
            revised_sentence = text
            reason = "이상 없음"

        fake_llm = MagicMock()
        fake_llm.with_structured_output.return_value.invoke.return_value = FakeDecision()
        with patch.object(review_agent, "_llm", return_value=fake_llm):
            result = review_agent.run_analysis(text, "가이드")

        self.assertFalse(result["has_issue"])
        self.assertFalse(result["revision_missing"])

    def test_review_sentences_counts_issue_without_revision(self):
        from services import review_agent

        with patch.object(review_agent, "usable_rag_count", return_value=3), patch.object(
            review_agent,
            "similarity_search",
            return_value=[Document(page_content="가이드", metadata={"source_file": "g.pdf"})],
        ), patch.object(
            review_agent,
            "run_analysis",
            return_value={
                "item_status": "reviewed",
                "has_issue": True,
                "revised_text": "원문과 같음",
                "reason": "문제 있음 (수정문구 미생성 — 추가 검토 필요)",
                "revision_missing": True,
            },
        ):
            events = _events(
                review_agent.review_sentences(["원문과 같음. 이 문장은 충분히 긴 계약 내용입니다."])
            )

        item = next(e for e in events if e["type"] == "item")
        done = next(e for e in events if e["type"] == "done")
        self.assertTrue(item["has_issue"])
        self.assertTrue(item["revision_missing"])
        self.assertEqual(done["issue_count"], 1)
        self.assertEqual(done["revision_missing_count"], 1)
        self.assertEqual(done["reviewed_count"], 1)


class DoneCommitPreserveTests(unittest.TestCase):
    def setUp(self):
        import services.rag_service as rag_service

        self.rag = rag_service
        self.tmp = Path(tempfile.mkdtemp(prefix="crr_done_"))
        self._old = {
            "CHROMA_DIR": rag_service.CHROMA_DIR,
            "CHROMA_BACKUP_DIR": rag_service.CHROMA_BACKUP_DIR,
            "RAG_UPLOAD_DIR": rag_service.RAG_UPLOAD_DIR,
        }
        rag_service.CHROMA_DIR = self.tmp / "chroma"
        rag_service.CHROMA_BACKUP_DIR = self.tmp / "backups"
        rag_service.RAG_UPLOAD_DIR = self.tmp / "uploads"
        rag_service.CHROMA_BACKUP_DIR.mkdir(parents=True)
        rag_service.RAG_UPLOAD_DIR.mkdir(parents=True)
        rag_service.reset_chroma_client()

    def tearDown(self):
        for key, value in self._old.items():
            setattr(self.rag, key, value)
        self.rag.reset_chroma_client()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_done_then_generator_close_keeps_data(self):
        sample = SAMPLE_DIR / "virtual_guideline.pdf"
        fake = DeterministicFakeEmbedding(size=8)

        def fake_split(docs):
            meta = dict(docs[0].metadata or {})
            return [Document(page_content="완료 후 유지", metadata=dict(meta))]

        gen = None
        with patch.object(self.rag, "_make_embeddings", return_value=fake), patch.object(
            self.rag, "split_rag_documents", side_effect=fake_split
        ):
            gen = self.rag.index_pdfs([sample], ["virtual_guideline.pdf"])
            done = None
            for event in gen:
                if event["type"] == "done":
                    done = event
                    break
            self.assertIsNotNone(done)
            # SSE 종료처럼 generator close
            gen.close()

        self.assertEqual(self.rag.usable_rag_count(), 1)
        self.assertEqual(self.rag.rag_document_count(), 1)

    def test_abort_before_done_rolls_back_new_only(self):
        import chromadb

        sample = SAMPLE_DIR / "virtual_guideline.pdf"
        other_hash = "aabbccdd" * 8
        client = chromadb.PersistentClient(path=str(self.rag.CHROMA_DIR))
        col = client.get_or_create_collection(name=CHROMA_COLLECTION, embedding_function=None)
        col.add(
            ids=["keep-me"],
            documents=["기존 정상"],
            embeddings=[[0.1] * 8],
            metadatas=[
                {
                    "source_file": "other.pdf",
                    "content_hash": other_hash,
                    "upload_job_id": "old",
                    "index_complete": True,
                }
            ],
        )
        self.rag.reset_chroma_client()
        self.assertEqual(self.rag.usable_rag_count(), 1)

        fake = DeterministicFakeEmbedding(size=8)
        calls = {"n": 0}

        def fake_split(docs):
            meta = dict(docs[0].metadata or {})
            return [
                Document(page_content="신규1", metadata=dict(meta)),
                Document(page_content="신규2", metadata=dict(meta)),
            ]

        def flaky_add(docs, ids=None):
            calls["n"] += 1
            if calls["n"] >= 1:
                # 첫 배치 저장 후 다음에서 실패하도록 — 한 배치에 모두 넣으면
                # add 자체에서 실패
                raise RuntimeError("mid store fail")

        with patch.object(self.rag, "_make_embeddings", return_value=fake), patch.object(
            self.rag, "split_rag_documents", side_effect=fake_split
        ), patch.object(self.rag, "_vectorstore_for_write") as vs_factory:
            store = MagicMock()
            store.add_documents.side_effect = flaky_add
            vs_factory.return_value = store
            events = _events(self.rag.index_pdfs([sample], ["virtual_guideline.pdf"]))

        self.assertTrue(any(e["type"] == "error" for e in events))
        self.assertEqual(self.rag.usable_rag_count(), 1)
        self.assertIn(other_hash, self.rag.existing_content_hashes())


if __name__ == "__main__":
    unittest.main()
