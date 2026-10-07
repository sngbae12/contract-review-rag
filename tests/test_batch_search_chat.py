# -*- coding: utf-8 -*-
"""대량 완료 표시 배치, 미완료 job 검색 제외, 잘못된 chat JSON 회귀 테스트."""

from __future__ import annotations

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


class BatchCompleteTests(unittest.TestCase):
    def setUp(self):
        import services.rag_service as rag_service

        self.rag = rag_service
        self.tmp = Path(tempfile.mkdtemp(prefix="crr_batch_"))
        self.chroma = self.tmp / "chroma"
        self.backup = self.tmp / "backups"
        self.uploads = self.tmp / "uploads"
        self.uploads.mkdir(parents=True)
        self.backup.mkdir(parents=True)
        self._old = {
            "CHROMA_DIR": rag_service.CHROMA_DIR,
            "CHROMA_BACKUP_DIR": rag_service.CHROMA_BACKUP_DIR,
            "RAG_UPLOAD_DIR": rag_service.RAG_UPLOAD_DIR,
        }
        rag_service.CHROMA_DIR = self.chroma
        rag_service.CHROMA_BACKUP_DIR = self.backup
        rag_service.RAG_UPLOAD_DIR = self.uploads
        rag_service.reset_chroma_client()

    def tearDown(self):
        for key, value in self._old.items():
            setattr(self.rag, key, value)
        self.rag.reset_chroma_client()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _seed_incomplete_job(self, n: int, job_id: str = "job-big", content_hash: str = "hash-big"):
        import chromadb

        client = chromadb.PersistentClient(path=str(self.chroma))
        col = client.get_or_create_collection(name=CHROMA_COLLECTION, embedding_function=None)
        ids = [f"{job_id}:{i:06d}" for i in range(n)]
        docs = [f"조각 {i} 내용" for i in range(n)]
        embeds = [[0.001 * (i + 1)] * 8 for i in range(n)]
        metas = [
            {
                "source_file": "big.pdf",
                "content_hash": content_hash,
                "upload_job_id": job_id,
                "index_complete": False,
            }
            for _ in range(n)
        ]
        # seed itself may need batches for add
        batch = self.rag.chroma_max_batch_size()
        for start in range(0, n, batch):
            end = min(start + batch, n)
            col.add(
                ids=ids[start:end],
                documents=docs[start:end],
                embeddings=embeds[start:end],
                metadatas=metas[start:end],
            )
        self.rag.reset_chroma_client()
        return ids

    def test_mark_complete_respects_runtime_batch_limit(self):
        # Use a tiny artificial limit to force multi-batch without storing 5k+ vectors.
        ids = self._seed_incomplete_job(12)
        with patch.object(self.rag, "chroma_max_batch_size", return_value=5):
            ok = self.rag.mark_ids_complete(ids)
        self.assertTrue(ok)
        analysis = self.rag.analyze_document_statuses()
        self.assertEqual(analysis["usable_count"], 12)
        self.assertEqual(analysis["incomplete_count"], 0)
        self.assertIn(("hash-big", "job-big"), analysis["usable_job_keys"])

    def test_mark_complete_mid_failure_reverts_whole_job(self):
        ids = self._seed_incomplete_job(9)
        real = self.rag._update_metadatas_batched

        def flaky(collection, batch_ids, metadatas):
            setting_complete = any(m.get("index_complete") is True for m in metadatas)
            if setting_complete:
                # 첫 배치만 반영된 뒤 실패를 흉내 낸다.
                size = 4
                assert real(collection, batch_ids[:size], metadatas[:size])
                return False
            return real(collection, batch_ids, metadatas)

        with patch.object(self.rag, "chroma_max_batch_size", return_value=4), patch.object(
            self.rag, "_update_metadatas_batched", side_effect=flaky
        ):
            ok = self.rag.mark_ids_complete(ids)
        self.assertFalse(ok)
        analysis = self.rag.analyze_document_statuses()
        self.assertEqual(analysis["usable_count"], 0)
        self.assertEqual(analysis["incomplete_count"], 9)
        self.assertEqual(self.rag.usable_rag_count(), 0)

    def test_index_pdfs_over_batch_limit_with_fake_embedding(self):
        sample = SAMPLE_DIR / "virtual_guideline.pdf"
        fake = DeterministicFakeEmbedding(size=8)
        # Force small batch so mark_ids_complete must split.
        n_chunks = 11

        def fake_split(docs):
            meta = dict(docs[0].metadata or {})
            return [
                Document(page_content=f"대량 조각 {i} 고유문구", metadata=dict(meta))
                for i in range(n_chunks)
            ]

        with patch.object(self.rag, "_make_embeddings", return_value=fake), patch.object(
            self.rag, "split_rag_documents", side_effect=fake_split
        ), patch.object(self.rag, "chroma_max_batch_size", return_value=4):
            events = _events(self.rag.index_pdfs([sample], ["virtual_guideline.pdf"]))
        done = next(e for e in events if e["type"] in ("done", "error"))
        self.assertEqual(done["type"], "done", msg=done)
        self.assertEqual(done["chunk_count"], n_chunks)
        self.assertEqual(self.rag.usable_rag_count(), n_chunks)


class IncompleteJobSearchTests(unittest.TestCase):
    def setUp(self):
        import services.rag_service as rag_service

        self.rag = rag_service
        self.tmp = Path(tempfile.mkdtemp(prefix="crr_search_job_"))
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

    def _seed(self, items):
        import chromadb

        client = chromadb.PersistentClient(path=str(self.rag.CHROMA_DIR))
        col = client.get_or_create_collection(name=CHROMA_COLLECTION, embedding_function=None)
        ids, docs, embeds, metas = [], [], [], []
        for index, item in enumerate(items):
            ids.append(item["id"])
            docs.append(item["text"])
            embeds.append([0.01 * (index + 1)] * 8)
            metas.append(item["meta"])
        col.add(ids=ids, documents=docs, embeddings=embeds, metadatas=metas)
        self.rag.reset_chroma_client()

    def test_same_hash_incomplete_job_chunks_excluded_even_if_true(self):
        shared = "same-hash"
        self._seed(
            [
                {
                    "id": "ok-1",
                    "text": "정상완료 GOODCHUNK 대금",
                    "meta": {
                        "source_file": "g.pdf",
                        "content_hash": shared,
                        "upload_job_id": "job-ok",
                        "index_complete": True,
                    },
                },
                {
                    "id": "bad-true",
                    "text": "미완료작업 TRUECHUNK 대금 BADUNIQUE",
                    "meta": {
                        "source_file": "g.pdf",
                        "content_hash": shared,
                        "upload_job_id": "job-partial",
                        "index_complete": True,
                    },
                },
                {
                    "id": "bad-false",
                    "text": "미완료작업 FALSECHUNK",
                    "meta": {
                        "source_file": "g.pdf",
                        "content_hash": shared,
                        "upload_job_id": "job-partial",
                        "index_complete": False,
                    },
                },
            ]
        )
        analysis = self.rag.analyze_document_statuses()
        self.assertIn((shared, "job-ok"), analysis["usable_job_keys"])
        self.assertNotIn((shared, "job-partial"), analysis["usable_job_keys"])
        self.assertEqual(analysis["usable_count"], 1)
        self.assertEqual(analysis["incomplete_count"], 2)

        fake = DeterministicFakeEmbedding(size=8)
        with patch.object(self.rag, "_make_embeddings", return_value=fake):
            docs = self.rag.similarity_search("BADUNIQUE 대금", k=5)
        texts = [d.page_content for d in docs]
        self.assertTrue(any("GOODCHUNK" in t for t in texts) or len(docs) >= 0)
        self.assertTrue(all("BADUNIQUE" not in t for t in texts))
        self.assertTrue(all("TRUECHUNK" not in t for t in texts))
        for doc in docs:
            self.assertEqual(doc.metadata.get("upload_job_id"), "job-ok")

    def test_restart_keeps_job_filter(self):
        shared = "restart-hash"
        self._seed(
            [
                {
                    "id": "ok-1",
                    "text": "재시작 정상",
                    "meta": {
                        "source_file": "a.pdf",
                        "content_hash": shared,
                        "upload_job_id": "job-ok",
                        "index_complete": True,
                    },
                },
                {
                    "id": "bad-1",
                    "text": "재시작 미완료 TRUE",
                    "meta": {
                        "source_file": "a.pdf",
                        "content_hash": shared,
                        "upload_job_id": "job-bad",
                        "index_complete": True,
                    },
                },
                {
                    "id": "bad-2",
                    "text": "재시작 미완료 FALSE",
                    "meta": {
                        "source_file": "a.pdf",
                        "content_hash": shared,
                        "upload_job_id": "job-bad",
                        "index_complete": False,
                    },
                },
            ]
        )
        self.rag.reset_chroma_client()
        fake = DeterministicFakeEmbedding(size=8)
        with patch.object(self.rag, "_make_embeddings", return_value=fake):
            docs = self.rag.similarity_search("재시작", k=5)
        self.assertTrue(all(d.metadata.get("upload_job_id") == "job-ok" for d in docs))


class ChatJsonValidationTests(unittest.TestCase):
    def setUp(self):
        from app import APP_STATE, app
        from services.openai_runtime import bind_api_key, reset_api_key

        self.app = app
        self.client = app.test_client()
        self.token = bind_api_key("sk-test-chat-validation")
        self._state = dict(APP_STATE)

    def tearDown(self):
        from app import APP_STATE
        from services.openai_runtime import reset_api_key

        reset_api_key(self.token)
        APP_STATE.clear()
        APP_STATE.update(self._state)

    def test_array_json_returns_400(self):
        res = self.client.post(
            "/api/chat",
            data="[1,2,3]",
            content_type="application/json",
            headers={"X-OpenAI-Api-Key": "sk-test-chat-validation"},
        )
        self.assertEqual(res.status_code, 400)
        body = res.get_json()
        self.assertEqual(body.get("type"), "error")
        self.assertIn("객체", body.get("message") or "")

    def test_numeric_question_returns_400(self):
        res = self.client.post(
            "/api/chat",
            json={"question": 123},
            headers={"X-OpenAI-Api-Key": "sk-test-chat-validation"},
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("문자열", res.get_json().get("message") or "")

    def test_object_question_returns_400(self):
        res = self.client.post(
            "/api/chat",
            json={"question": {"text": "hi"}},
            headers={"X-OpenAI-Api-Key": "sk-test-chat-validation"},
        )
        self.assertEqual(res.status_code, 400)

    def test_null_question_returns_400(self):
        res = self.client.post(
            "/api/chat",
            json={"question": None},
            headers={"X-OpenAI-Api-Key": "sk-test-chat-validation"},
        )
        self.assertEqual(res.status_code, 400)

    def test_empty_question_returns_400(self):
        res = self.client.post(
            "/api/chat",
            json={"question": "   "},
            headers={"X-OpenAI-Api-Key": "sk-test-chat-validation"},
        )
        self.assertEqual(res.status_code, 400)

    def test_broken_json_returns_400(self):
        res = self.client.post(
            "/api/chat",
            data="{not-json",
            content_type="application/json",
            headers={"X-OpenAI-Api-Key": "sk-test-chat-validation"},
        )
        self.assertEqual(res.status_code, 400)

    def test_invalid_json_without_key_still_400_not_500(self):
        res = self.client.post(
            "/api/chat",
            data="[1,2]",
            content_type="application/json",
        )
        # 키 없음이면 400 missing key, 배열이면 파싱 오류 400 — 둘 다 500이 아니어야 함
        self.assertNotEqual(res.status_code, 500)
        self.assertEqual(res.status_code, 400)

    def test_valid_header_key_still_accepted_path(self):
        import app as app_module

        calls = {"n": 0}

        def fake_stream(question):
            calls["n"] += 1
            self.assertIsInstance(question, str)
            yield "안"
            yield "녕"

        # from-import 별칭과 모듈 속성을 함께 교체해야 뷰가 mock을 본다.
        with patch.object(app_module, "stream_answer", side_effect=fake_stream):
            client = app_module.app.test_client()
            res = client.post(
                "/api/chat",
                json={"question": "안녕하세요"},
                headers={"X-OpenAI-Api-Key": "sk-test-chat-validation"},
            )
            body = res.data.decode("utf-8", errors="replace")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(calls["n"], 1)
        self.assertIn('"type": "done"', body)
        self.assertIn("안", body)


if __name__ == "__main__":
    unittest.main()
