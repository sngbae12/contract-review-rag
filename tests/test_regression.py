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
from services.text_split import (
    detect_clause,
    split_contract_pages,
    split_contract_text,
    split_rag_documents,
)


def _events(gen):
    return list(gen)


def _consume_until_done(gen):
    """done/error를 받는 즉시 close하여 Flask SSE 조기 종료를 흉내 낸다."""
    events = []
    try:
        for event in gen:
            events.append(event)
            if event.get("type") in ("done", "error"):
                break
    finally:
        gen.close()
    return events


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
        self.assertTrue(units[0]["text"].startswith("제1조"))
        self.assertTrue(any(u["text"].startswith("제2조") or "긴내용" in u["text"] for u in units))
        self.assertTrue(all(len(u["text"]) <= 800 for u in units))
        self.assertEqual(detect_clause(units[0]["text"]), "제1조")
        self.assertTrue(all(u["clause"] for u in units if "긴내용" in u["text"]))

    def test_body_reference_is_not_new_article(self):
        text = (
            "제1조 (대금) 구매자는 제2조에 따라 대금을 지급한다.\n"
            "제2조 (기한) 지급 기한은 30일이다."
        )
        units = split_contract_text(text)
        self.assertEqual(len(units), 2)
        self.assertIn("제2조에 따라", units[0]["text"])
        self.assertTrue(units[0]["text"].startswith("제1조"))
        self.assertTrue(units[1]["text"].startswith("제2조"))
        self.assertEqual(units[0]["clause"], "제1조")
        self.assertEqual(units[1]["clause"], "제2조")
        joined = units[0]["text"] + "\n" + units[1]["text"]
        self.assertIn("구매자는 제2조에 따라", joined)
        self.assertIn("지급 기한은 30일", joined)

    def test_reference_준용_not_split(self):
        text = "제1조 목적. 제3조를 준용한다.\n제3조 범위. 본 조는 범위만 정한다."
        units = split_contract_text(text)
        self.assertEqual(len(units), 2)
        self.assertIn("제3조를 준용한다", units[0]["text"])

    def test_long_clause_keeps_clause_metadata(self):
        text = "제5조 (상세) " + ("상세내용 " * 250)
        units = split_contract_text(text)
        self.assertGreater(len(units), 1)
        self.assertTrue(units[0]["text"].startswith("제5조"))
        self.assertTrue(all(u["clause"] == "제5조" for u in units))
        self.assertFalse(any(u["text"].startswith("제5조") for u in units[1:]))

    def test_clause_carries_across_pages(self):
        pages = [
            ("제7조 (계속) 첫 페이지 본문입니다.", 1),
            ("이어지는 두 번째 페이지 본문입니다. 새 제목은 없습니다.", 2),
            ("제8조 (다음) 새 조항입니다.", 3),
        ]
        units = split_contract_pages(pages, source_file="c.pdf")
        self.assertEqual(units[0]["clause"], "제7조")
        self.assertEqual(units[0]["page"], 1)
        cont = next(u for u in units if "두 번째 페이지" in u["text"])
        self.assertEqual(cont["clause"], "제7조")
        self.assertEqual(cont["page"], 2)
        self.assertFalse(cont["text"].startswith("제7조"))
        last = next(u for u in units if u["text"].startswith("제8조"))
        self.assertEqual(last["clause"], "제8조")
        self.assertEqual(last["page"], 3)


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
        self.assertIn('addTextMessage("user", `RAG 파일 업로드: ${names}`)', self.js)
        self.assertIn('addTextMessage("user", `계약서 업로드: ${file.name}`)', self.js)
        self.assertNotIn('addMessage("user", `RAG', self.js)
        self.assertIn("INCOMPLETE_STREAM", self.js)
        self.assertIn("event.isComposing || event.keyCode === 229", self.js)
        self.assertIn('event.key === "Enter" && !event.shiftKey', self.js)
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

    def _seed(self, doc_id, text, content_hash, complete=True):
        import chromadb

        client = chromadb.PersistentClient(path=str(self.tmp))
        col = client.get_or_create_collection(name=CHROMA_COLLECTION, embedding_function=None)
        col.add(
            ids=[doc_id],
            documents=[text],
            embeddings=[[0.1] * 8],
            metadatas=[
                {
                    "source_file": "old.pdf",
                    "content_hash": content_hash,
                    "index_complete": complete,
                }
            ],
        )
        self.rag.reset_chroma_client()
        return col

    def _index_with_fake(self, sample, name, add_hook=None, split_n=None):
        fake = DeterministicFakeEmbedding(size=8)
        original_write = self.rag._vectorstore_for_write

        def wrapped_write():
            store = original_write()
            orig_add = store.add_documents

            def add(batch, ids=None):
                if add_hook:
                    return add_hook(orig_add, batch, ids)
                return orig_add(batch, ids=ids)

            store.add_documents = add
            return store

        patches = [
            patch.object(self.rag, "_make_embeddings", return_value=fake),
            patch.object(self.rag, "_vectorstore_for_write", side_effect=wrapped_write),
        ]
        if split_n is not None:
            def fake_split(docs):
                meta = dict(docs[0].metadata or {})
                meta["index_complete"] = False
                return [
                    Document(page_content=f"가이드 조각 {i} " * 8, metadata=dict(meta))
                    for i in range(split_n)
                ]

            patches.append(patch.object(self.rag, "split_rag_documents", side_effect=fake_split))

        with patches[0], patches[1]:
            if split_n is not None:
                with patches[2]:
                    return _events(self.rag.index_pdfs([sample], [name]))
            return _events(self.rag.index_pdfs([sample], [name]))

    def test_count_without_key_does_not_create_collection(self):
        from services.openai_runtime import current_api_key

        self.assertEqual(current_api_key(), "")
        self.assertEqual(self.rag.rag_document_count(), 0)
        self.assertFalse((self.tmp / "chroma.sqlite3").exists())
        self.assertIsNone(self.rag.get_existing_collection())

    def test_restart_count_reads_existing_db(self):
        self._seed("keep-1", "기존 가이드라인", "abc", complete=True)
        self.assertEqual(self.rag.rag_document_count(), 1)
        self.assertFalse(hasattr(self.rag, "_bound_key_fingerprint"))

    def test_duplicate_skip_only_for_complete(self):
        sample = SAMPLE_DIR / "virtual_guideline.pdf"
        digest = self.rag.file_sha256(sample)
        self._seed("keep-1", "기존", digest, complete=True)
        events = self._index_with_fake(sample, "virtual_guideline.pdf")
        done = next(e for e in events if e["type"] in ("done", "error"))
        self.assertEqual(done["type"], "done")
        self.assertEqual(done["chunk_count"], 0)
        self.assertEqual(self.rag.rag_document_count(), 1)
        self.assertIn("virtual_guideline.pdf", done.get("skipped_duplicates") or [])

    def test_incomplete_hash_allows_reupload(self):
        sample = SAMPLE_DIR / "virtual_guideline.pdf"
        digest = self.rag.file_sha256(sample)
        self._seed("leftover-1", "불완전", digest, complete=False)
        events = self._index_with_fake(sample, "virtual_guideline.pdf", split_n=3)
        done = next(e for e in events if e["type"] in ("done", "error"))
        self.assertEqual(done["type"], "done")
        self.assertEqual(done["chunk_count"], 3)
        self.assertGreaterEqual(self.rag.rag_document_count(), 4)

    def test_done_then_close_keeps_data(self):
        self._seed("keep-1", "기존 유지", "old-hash", complete=True)
        sample = SAMPLE_DIR / "virtual_guideline.pdf"
        fake = DeterministicFakeEmbedding(size=8)

        def fake_split(docs):
            meta = dict(docs[0].metadata or {})
            meta["index_complete"] = False
            return [Document(page_content="신규 가이드라인 조각", metadata=meta)]

        with patch.object(self.rag, "_make_embeddings", return_value=fake), patch.object(
            self.rag, "split_rag_documents", side_effect=fake_split
        ):
            events = _consume_until_done(self.rag.index_pdfs([sample], ["virtual_guideline.pdf"]))

        done = next(e for e in events if e["type"] == "done")
        self.assertEqual(done["chunk_count"], 1)
        self.rag.reset_chroma_client()
        self.assertEqual(self.rag.rag_document_count(), 2)
        data = self.rag.get_existing_collection().get(include=["metadatas"])
        completes = [m.get("index_complete") for m in data["metadatas"]]
        self.assertEqual(completes.count(True), 2)

    def test_cancel_before_done_rolls_back_new_only(self):
        self._seed("keep-1", "기존 유지", "old-hash", complete=True)
        sample = SAMPLE_DIR / "virtual_guideline.pdf"
        fake = DeterministicFakeEmbedding(size=8)

        def fake_split(docs):
            meta = dict(docs[0].metadata or {})
            return [Document(page_content=f"신규 {i}", metadata=dict(meta)) for i in range(3)]

        with patch.object(self.rag, "_make_embeddings", return_value=fake), patch.object(
            self.rag, "split_rag_documents", side_effect=fake_split
        ):
            gen = self.rag.index_pdfs([sample], ["virtual_guideline.pdf"])
            for event in gen:
                if event.get("type") == "progress" and event.get("stage") == "embedding" and event.get("current", 0) >= 3:
                    gen.close()
                    break

        self.rag.reset_chroma_client()
        self.assertEqual(self.rag.rag_document_count(), 1)
        self.assertEqual(self.rag.get_existing_collection().get(include=[])["ids"], ["keep-1"])

    def test_mid_batch_partial_store_rolls_back_all_new(self):
        """첫 배치 성공 후, 다음 배치에서 일부 upsert 뒤 예외가 나도 신규 전체를 지운다."""
        self._seed("keep-1", "기존 유지", "old-hash", complete=True)
        sample = SAMPLE_DIR / "virtual_guideline.pdf"
        calls = {"n": 0}

        def add_hook(orig_add, batch, ids):
            calls["n"] += 1
            if calls["n"] == 1:
                return orig_add(batch, ids=ids)
            # 실패한 배치: 일부만 저장한 뒤 예외
            partial = batch[:2]
            partial_ids = ids[:2]
            orig_add(partial, ids=partial_ids)
            raise RuntimeError("boom-mid-batch")

        events = self._index_with_fake(
            sample,
            "virtual_guideline.pdf",
            add_hook=add_hook,
            split_n=25,
        )
        err = next(e for e in events if e["type"] == "error")
        self.assertTrue(err.get("rollback_ok"))
        self.assertIn("되돌렸", err["message"])
        self.assertNotIn("모두 되돌렸습니다", err["message"])
        self.rag.reset_chroma_client()
        self.assertEqual(self.rag.rag_document_count(), 1)
        self.assertEqual(self.rag.get_existing_collection().get(include=[])["ids"], ["keep-1"])

    def test_failed_pdf_can_reupload_fully(self):
        sample = SAMPLE_DIR / "virtual_guideline.pdf"
        calls = {"n": 0}

        def add_hook(orig_add, batch, ids):
            calls["n"] += 1
            if calls["n"] == 1:
                return orig_add(batch, ids=ids)
            partial = batch[:2]
            orig_add(partial, ids=ids[:2])
            raise RuntimeError("boom")

        first = self._index_with_fake(sample, "virtual_guideline.pdf", add_hook=add_hook, split_n=25)
        self.assertEqual(next(e for e in first if e["type"] == "error")["type"], "error")
        self.assertEqual(self.rag.rag_document_count(), 0)

        second = self._index_with_fake(sample, "virtual_guideline.pdf", split_n=5)
        done = next(e for e in second if e["type"] in ("done", "error"))
        self.assertEqual(done["type"], "done")
        self.assertEqual(done["chunk_count"], 5)
        self.assertEqual(self.rag.rag_document_count(), 5)

    def test_rollback_failure_message_is_accurate(self):
        self._seed("keep-1", "기존 유지", "old-hash", complete=True)
        sample = SAMPLE_DIR / "virtual_guideline.pdf"

        def add_hook(orig_add, batch, ids):
            orig_add(batch, ids=ids)
            raise RuntimeError("boom")

        with patch.object(self.rag, "delete_ids", return_value={
            "requested": 3,
            "deleted": 1,
            "remaining": 2,
            "ok": False,
        }):
            events = self._index_with_fake(
                sample,
                "virtual_guideline.pdf",
                add_hook=add_hook,
                split_n=3,
            )
        err = next(e for e in events if e["type"] == "error")
        self.assertFalse(err.get("rollback_ok"))
        self.assertIn("남아 있을 수 있습니다", err["message"])
        self.assertNotIn("새로 추가하려던 조각을 되돌렸습니다", err["message"])
        self.assertIn("2개", err["message"])


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
            metadatas=[{"source_file": "g.pdf", "index_complete": True}],
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

    def test_flask_sse_close_after_done_keeps_chunks(self):
        """Flask 테스트 클라이언트로 done 직후 응답을 읽어도 데이터가 유지되는지 확인."""
        import chromadb
        from services.openai_runtime import bind_api_key, reset_api_key

        client = chromadb.PersistentClient(path=str(self.tmp))
        col = client.get_or_create_collection(name=CHROMA_COLLECTION, embedding_function=None)
        col.add(
            ids=["keep-1"],
            documents=["기존"],
            embeddings=[[0.5] * 8],
            metadatas=[{"source_file": "old.pdf", "content_hash": "old", "index_complete": True}],
        )
        self.rag.reset_chroma_client()

        sample = SAMPLE_DIR / "virtual_guideline.pdf"
        fake = DeterministicFakeEmbedding(size=8)

        def fake_split(docs):
            meta = dict(docs[0].metadata or {})
            return [Document(page_content="flask sse 신규 조각", metadata=dict(meta))]

        token = bind_api_key("sk-test-not-used-for-real")
        try:
            with patch.object(self.rag, "_make_embeddings", return_value=fake), patch.object(
                self.rag, "split_rag_documents", side_effect=fake_split
            ):
                client_app = self.app.test_client()
                with open(sample, "rb") as handle:
                    res = client_app.post(
                        "/api/upload_rag",
                        data={"files": (handle, "virtual_guideline.pdf"), "openai_api_key": "sk-test-not-used-for-real"},
                        content_type="multipart/form-data",
                        headers={"X-OpenAI-Api-Key": "sk-test-not-used-for-real"},
                    )
                    # 응답 본문을 끝까지 읽고 연결을 닫는다.
                    body = res.data.decode("utf-8", errors="replace")
        finally:
            reset_api_key(token)

        self.assertIn('"type": "done"', body)
        self.rag.reset_chroma_client()
        self.assertEqual(self.rag.rag_document_count(), 2)


if __name__ == "__main__":
    unittest.main()