# -*- coding: utf-8 -*-
"""테스트 대역 중심 회귀 테스트. 실제 OpenAI 호출은 하지 않는다."""

from __future__ import annotations

import io
import json
import os
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
from services.text_split import detect_clause, split_contract_text, split_rag_documents


def _events(gen):
    return list(gen)


class SplitTests(unittest.TestCase):
    def test_rag_splits_long_line(self):
        long_line = "가" * 900
        docs = split_rag_documents([Document(page_content=long_line, metadata={"source": "t.pdf"})])
        self.assertTrue(docs)
        self.assertTrue(all(len(d.page_content) <= 400 for d in docs))

    def test_contract_keeps_articles_and_splits_long(self):
        text = "제1조 목적. 짧은 조항이다.\n제2조 " + ("긴내용 " * 200)
        units = split_contract_text(text)
        self.assertGreaterEqual(len(units), 2)
        self.assertTrue(units[0].startswith("제1조"))
        self.assertTrue(any(u.startswith("제2조") or "긴내용" in u for u in units))
        self.assertTrue(all(len(u) <= 800 for u in units))
        self.assertEqual(detect_clause(units[0]), "제1조")


class SseParserTests(unittest.TestCase):
    """static/js/app.js 의 handleSseBuffer / consumeSSE 규칙과 같은 로직."""

    def handle(self, chunk, events):
        parts = chunk.split("\n\n")
        rest = parts.pop()
        for part in parts:
            line = next((row for row in part.split("\n") if row.startswith("data: ")), "")
            if not line:
                continue
            event = json.loads(line[6:])
            events.append(event)
            if event.get("type") in ("done", "error"):
                return "", event
        return rest, None

    def consume(self, pieces):
        buffer = ""
        events = []
        terminal = None
        for i, piece in enumerate(pieces):
            last = i == len(pieces) - 1
            if last:
                buffer += piece
                if buffer.strip():
                    _, terminal = self.handle(buffer + "\n\n", events)
                    if terminal:
                        return events, terminal
                raise RuntimeError("incomplete")
            buffer += piece
            rest, terminal = self.handle(buffer, events)
            buffer = rest
            if terminal:
                return events, terminal
        raise RuntimeError("incomplete")

    def test_done_in_last_buffer(self):
        events, terminal = self.consume(['data: {"type": "token", "text": "안녕"}\n\n', 'data: {"type": "done"}\n'])
        self.assertEqual(terminal["type"], "done")
        self.assertEqual(events[0]["text"], "안녕")

    def test_explicit_error(self):
        events, terminal = self.consume(['data: {"type": "error", "message": "실패"}\n\n'])
        self.assertEqual(terminal["type"], "error")
        self.assertEqual(events[0]["message"], "실패")

    def test_incomplete_raises(self):
        with self.assertRaises(RuntimeError):
            self.consume(['data: {"type": "token", "text": "중간"}\n\n'])


class FrontendSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.js = (ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")

    def test_filenames_use_text_message(self):
        self.assertIn("addTextMessage(\"user\", `RAG 파일 업로드: ${names}`)", self.js)
        self.assertIn("addTextMessage(\"user\", `계약서 업로드: ${file.name}`)", self.js)
        self.assertNotIn("addMessage(\"user\", `RAG", self.js)
        self.assertIn("INCOMPLETE_STREAM", self.js)
        self.assertIn("event.isComposing || event.keyCode === 229", self.js)
        self.assertIn("event.key === \"Enter\" && !event.shiftKey", self.js)
        self.assertIn("localStorage", (ROOT / "templates" / "index.html").read_text(encoding="utf-8"))

    def test_no_innerhtml_for_user_upload_names(self):
        self.assertNotIn("bubble.innerHTML = `RAG", self.js)
        self.assertIn("appendReviewItem", self.js)
        self.assertIn("textContent += evt.text", self.js)


class ReviewFlowTests(unittest.TestCase):
    def test_search_error_skips_llm(self):
        from services import review_agent

        llm_calls = {"n": 0}

        def boom_llm():
            llm_calls["n"] += 1
            raise AssertionError("LLM should not run")

        with patch.object(review_agent, "rag_document_count", return_value=3), patch.object(
            review_agent, "similarity_search", side_effect=RuntimeError("search down")
        ), patch.object(review_agent, "_llm", side_effect=boom_llm):
            events = _events(review_agent.review_sentences(["이 조항은 충분히 긴 계약 내용입니다."]))

        items = [e for e in events if e.get("type") == "item"]
        done = [e for e in events if e.get("type") == "done"]
        self.assertEqual(llm_calls["n"], 0)
        self.assertEqual(items[0]["item_status"], "search_error")
        self.assertFalse(items[0]["has_issue"])
        self.assertIn("실패", items[0]["reason"])
        self.assertEqual(done[0]["search_error_count"], 1)
        self.assertEqual(done[0]["reviewed_count"], 0)

    def test_empty_search_is_no_evidence(self):
        from services import review_agent

        with patch.object(review_agent, "rag_document_count", return_value=3), patch.object(
            review_agent, "similarity_search", return_value=[]
        ), patch.object(review_agent, "_llm", side_effect=AssertionError("LLM should not run")):
            events = _events(review_agent.review_sentences(["이 조항은 충분히 긴 계약 내용입니다."]))

        item = next(e for e in events if e.get("type") == "item")
        done = next(e for e in events if e.get("type") == "done")
        self.assertEqual(item["item_status"], "no_evidence")
        self.assertIn("근거", item["reason"])
        self.assertEqual(done["no_evidence_count"], 1)

    def test_successful_search_runs_analysis(self):
        from services import review_agent

        with patch.object(review_agent, "rag_document_count", return_value=3), patch.object(
            review_agent,
            "similarity_search",
            return_value=[Document(page_content="지급 기한", metadata={"source_file": "g.pdf", "page": 1})],
        ), patch.object(
            review_agent,
            "run_analysis",
            return_value={
                "item_status": "reviewed",
                "has_issue": True,
                "revised_text": "고친 문구",
                "reason": "기한이 없다",
            },
        ) as analysis:
            events = _events(review_agent.review_sentences(["대금을 지급한다. 이 문장은 충분히 깁니다."]))

        self.assertTrue(analysis.called)
        item = next(e for e in events if e.get("type") == "item")
        self.assertEqual(item["item_status"], "reviewed")
        self.assertTrue(item["has_issue"])
        self.assertEqual(item["revised"], "고친 문구")


class ChromaStoreTests(unittest.TestCase):
    def setUp(self):
        import services.rag_service as rag_service

        self.rag = rag_service
        self.tmp = Path(tempfile.mkdtemp(prefix="crr_chroma_"))
        self._old = rag_service.CHROMA_DIR
        rag_service.CHROMA_DIR = self.tmp
        rag_service.reset_chroma_client()

    def tearDown(self):
        self.rag.CHROMA_DIR = self._old
        self.rag.reset_chroma_client()

    def test_count_without_key_does_not_create_collection(self):
        from services.openai_runtime import current_api_key

        self.assertEqual(current_api_key(), "")
        self.assertEqual(self.rag.rag_document_count(), 0)
        self.assertFalse((self.tmp / "chroma.sqlite3").exists())
        self.assertIsNone(self.rag.get_existing_collection())

    def test_restart_count_reads_existing_db(self):
        import chromadb

        client = chromadb.PersistentClient(path=str(self.tmp))
        col = client.get_or_create_collection(name=CHROMA_COLLECTION, embedding_function=None)
        col.add(
            ids=["keep-1"],
            documents=["기존 가이드라인"],
            embeddings=[[0.1] * 8],
            metadatas=[{"source_file": "old.pdf", "content_hash": "abc"}],
        )
        self.rag.reset_chroma_client()
        self.assertEqual(self.rag.rag_document_count(), 1)
        self.assertFalse(hasattr(self.rag, "_vectorstore") and self.rag.__dict__.get("_vectorstore"))
        self.assertFalse(hasattr(self.rag, "_bound_key_fingerprint"))

    def test_duplicate_skip_keeps_count(self):
        import chromadb

        sample = SAMPLE_DIR / "virtual_guideline.pdf"
        digest = self.rag.file_sha256(sample)
        client = chromadb.PersistentClient(path=str(self.tmp))
        col = client.get_or_create_collection(name=CHROMA_COLLECTION, embedding_function=None)
        col.add(
            ids=["keep-1"],
            documents=["기존"],
            embeddings=[[0.2] * 8],
            metadatas=[{"source_file": "virtual_guideline.pdf", "content_hash": digest}],
        )
        self.rag.reset_chroma_client()
        events = _events(self.rag.index_pdfs([sample], ["virtual_guideline.pdf"]))
        done = next(e for e in events if e["type"] in ("done", "error"))
        self.assertEqual(done["type"], "done")
        self.assertEqual(done["chunk_count"], 0)
        self.assertEqual(self.rag.rag_document_count(), 1)
        self.assertIn("virtual_guideline.pdf", done.get("skipped_duplicates") or [])

    def test_partial_failure_rolls_back_only_new_ids(self):
        import chromadb

        client = chromadb.PersistentClient(path=str(self.tmp))
        col = client.get_or_create_collection(name=CHROMA_COLLECTION, embedding_function=None)
        col.add(
            ids=["keep-1"],
            documents=["기존 유지"],
            embeddings=[[0.3] * 8],
            metadatas=[{"source_file": "old.pdf", "content_hash": "old-hash"}],
        )
        self.rag.reset_chroma_client()

        fake = DeterministicFakeEmbedding(size=8)
        original_write = self.rag._vectorstore_for_write
        calls = {"n": 0}

        def wrapped_write():
            store = original_write()
            orig_add = store.add_documents

            def add(batch, ids=None):
                calls["n"] += 1
                if calls["n"] > 1:
                    raise RuntimeError("boom")
                return orig_add(batch, ids=ids)

            store.add_documents = add
            return store

        def fake_split(docs):
            meta = dict(docs[0].metadata or {})
            return [Document(page_content=f"가이드 조각 {i} " * 8, metadata=meta) for i in range(25)]

        with patch.object(self.rag, "_make_embeddings", return_value=fake), patch.object(
            self.rag, "_vectorstore_for_write", side_effect=wrapped_write
        ), patch.object(self.rag, "split_rag_documents", side_effect=fake_split):
            events = _events(self.rag.index_pdfs([SAMPLE_DIR / "virtual_guideline.pdf"], ["virtual_guideline.pdf"]))

        err = next(e for e in events if e["type"] == "error")
        self.assertIn("되돌렸", err["message"])
        self.rag.reset_chroma_client()
        self.assertEqual(self.rag.rag_document_count(), 1)
        remaining = self.rag.get_existing_collection().get(include=["metadatas"])
        self.assertEqual(remaining["ids"], ["keep-1"])


class AppStatusTests(unittest.TestCase):
    def setUp(self):
        import services.rag_service as rag_service
        from app import APP_STATE, app

        self.rag = rag_service
        self.APP_STATE = APP_STATE
        self.app = app
        self.tmp = Path(tempfile.mkdtemp(prefix="crr_status_"))
        self._old = rag_service.CHROMA_DIR
        rag_service.CHROMA_DIR = self.tmp
        rag_service.reset_chroma_client()
        self._state = dict(APP_STATE)
        APP_STATE.update(
            {
                "rag_ready": False,
                "rag_stage": "idle",
                "rag_chunk_count": 0,
                "contract_ready": False,
                "contract_stage": "idle",
                "contract_filename": "",
                "contract_units": [],
                "reviewing": False,
            }
        )

    def tearDown(self):
        self.rag.CHROMA_DIR = self._old
        self.rag.reset_chroma_client()
        self.APP_STATE.clear()
        self.APP_STATE.update(self._state)

    def test_status_without_api_key_sees_existing_db(self):
        import chromadb

        client = chromadb.PersistentClient(path=str(self.tmp))
        col = client.get_or_create_collection(name=CHROMA_COLLECTION, embedding_function=None)
        col.add(
            ids=["a"],
            documents=["가이드"],
            embeddings=[[0.4] * 8],
            metadatas=[{"source_file": "g.pdf"}],
        )
        self.rag.reset_chroma_client()
        client_app = self.app.test_client()
        res = client_app.get("/api/status")
        data = res.get_json()
        self.assertEqual(res.status_code, 200)
        self.assertEqual(data["rag_chunk_count"], 1)
        self.assertTrue(data["rag_ready"])
        self.assertEqual(data["rag_stage"], "ready")

    def test_broken_contract_keeps_previous_units(self):
        self.APP_STATE["contract_ready"] = True
        self.APP_STATE["contract_filename"] = "기존.pdf"
        self.APP_STATE["contract_units"] = [{"text": "기존 조항", "source_file": "기존.pdf"}]
        self.APP_STATE["contract_stage"] = "ready"
        broken = (b"%PDF-1.4\n1 0 obj<<>>endobj\nbroken", "broken.pdf")
        client_app = self.app.test_client()
        res = client_app.post(
            "/api/upload_contract",
            data={"file": (io.BytesIO(broken[0]), broken[1])},
            content_type="multipart/form-data",
        )
        body = res.data.decode("utf-8", errors="replace")
        self.assertIn("error", body)
        self.assertEqual(self.APP_STATE["contract_filename"], "기존.pdf")
        self.assertEqual(self.APP_STATE["contract_units"][0]["text"], "기존 조항")
        self.assertTrue(self.APP_STATE["contract_ready"])

    def test_valid_sample_contract_split(self):
        from services.review_agent import split_contract_pdf

        events = _events(split_contract_pdf(SAMPLE_DIR / "virtual_contract.pdf", "virtual_contract.pdf"))
        done = next(e for e in events if e["type"] == "done")
        self.assertGreaterEqual(done["unit_count"], 1)
        self.assertEqual(done["filename"], "virtual_contract.pdf")
        self.assertIn("text", done["units"][0])

    def test_missing_key_endpoints(self):
        client_app = self.app.test_client()
        rag = client_app.post("/api/upload_rag", data={})
        chat = client_app.post("/api/chat", json={"question": "안녕"})
        review = client_app.post("/api/review")
        self.assertEqual(rag.status_code, 400)
        self.assertEqual(chat.status_code, 400)
        self.assertEqual(review.status_code, 400)


if __name__ == "__main__":
    unittest.main()
