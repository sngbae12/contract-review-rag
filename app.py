"""
LLM 기반 계약서 검토 웹 (Flask)
실행: python app.py
브라우저: http://127.0.0.1:8765
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from flask import Flask, Response, g, jsonify, render_template, request, stream_with_context
from werkzeug.utils import secure_filename

from config import CONTRACT_UPLOAD_DIR, MAX_CONTENT_LENGTH, RAG_UPLOAD_DIR, ensure_directories
from services.chat_service import stream_answer
from services.openai_runtime import (
    MissingApiKeyError,
    bind_api_key,
    current_api_key,
    reset_api_key,
    user_facing_openai_error,
)
from services.rag_service import index_pdfs, rag_document_count
from services.review_agent import review_sentences, split_contract_pdf

ensure_directories()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["JSON_AS_ASCII"] = False

APP_STATE: dict = {
    "rag_ready": False,
    "rag_stage": "idle",
    "rag_chunk_count": 0,
    "contract_ready": False,
    "contract_stage": "idle",
    "contract_filename": "",
    "contract_sentences": [],
    "reviewing": False,
}

MISSING_KEY_MESSAGE = "OpenAI API 키가 없습니다. 왼쪽 입력란에 키를 넣은 뒤 다시 시도해 주세요."


def _extract_api_key() -> str:
    """요청 헤더·폼·JSON에서만 키를 읽는다. URL 쿼리는 쓰지 않는다."""
    header = (request.headers.get("X-OpenAI-Api-Key") or "").strip()
    if header:
        return header
    if request.form:
        form_key = (request.form.get("openai_api_key") or "").strip()
        if form_key:
            return form_key
    if request.is_json:
        data = request.get_json(silent=True) or {}
        return str(data.get("openai_api_key") or "").strip()
    return ""


@app.before_request
def _bind_request_api_key():
    g.api_key_token = bind_api_key(_extract_api_key())


@app.teardown_request
def _clear_request_api_key(_exc):
    token = getattr(g, "api_key_token", None)
    g.api_key_token = None
    reset_api_key(token)


def _missing_key_response():
    return jsonify({"type": "error", "message": MISSING_KEY_MESSAGE}), 400


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _stream_events(events, kind: str | None = None):
    try:
        for event in events:
            etype = event.get("type")
            stage = event.get("stage")
            if kind == "rag":
                if etype == "progress":
                    APP_STATE["rag_stage"] = stage or "processing"
                elif etype == "done":
                    count = int(event.get("collection_count") or 0)
                    APP_STATE["rag_ready"] = count > 0
                    APP_STATE["rag_chunk_count"] = count
                    APP_STATE["rag_stage"] = "ready"
                elif etype == "error":
                    APP_STATE["rag_stage"] = "error"
            elif kind == "contract":
                if etype == "progress":
                    APP_STATE["contract_stage"] = stage or "processing"
                elif etype == "done" and "sentences" in event:
                    APP_STATE["contract_ready"] = True
                    APP_STATE["contract_filename"] = event.get("filename", "")
                    APP_STATE["contract_sentences"] = event.get("sentences", [])
                    APP_STATE["contract_stage"] = "ready"
                elif etype == "error":
                    APP_STATE["contract_stage"] = "error"
            elif kind == "review":
                APP_STATE["reviewing"] = etype not in ("done", "error")
            yield _sse(event)
    except MissingApiKeyError:
        if kind == "rag":
            APP_STATE["rag_stage"] = "error"
        elif kind == "contract":
            APP_STATE["contract_stage"] = "error"
        elif kind == "review":
            APP_STATE["reviewing"] = False
        yield _sse({"type": "error", "stage": "error", "message": MISSING_KEY_MESSAGE})
    except Exception as exc:
        app.logger.error("stream failed kind=%s", kind)
        if kind == "rag":
            APP_STATE["rag_stage"] = "error"
        elif kind == "contract":
            APP_STATE["contract_stage"] = "error"
        elif kind == "review":
            APP_STATE["reviewing"] = False
        yield _sse(
            {
                "type": "error",
                "stage": "error",
                "message": user_facing_openai_error(exc),
            }
        )
    finally:
        if kind == "review":
            APP_STATE["reviewing"] = False
        if kind == "rag" and APP_STATE.get("rag_stage") not in ("ready", "error"):
            APP_STATE["rag_stage"] = "idle"
        if kind == "contract" and APP_STATE.get("contract_stage") not in ("ready", "error"):
            APP_STATE["contract_stage"] = "idle"


def _sse_response(generate):
    return Response(
        generate,
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "close",
        },
    )


def _sse_with_key(key: str, events, kind: str | None = None):
    """스트리밍 동안에도 요청에서 받은 키를 메모리에 유지한다."""

    @stream_with_context
    def generate():
        token = bind_api_key(key)
        try:
            yield from _stream_events(events, kind=kind)
        finally:
            reset_api_key(token)

    return _sse_response(generate())


def _save_pdfs(file_storages, dest_dir: Path) -> list[Path]:
    saved: list[Path] = []
    dest_dir.mkdir(parents=True, exist_ok=True)
    for storage in file_storages:
        if not storage or not storage.filename:
            continue
        filename = storage.filename
        if not filename.lower().endswith(".pdf"):
            continue
        safe_name = secure_filename(filename) or "upload.pdf"
        unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
        path = dest_dir / unique_name
        storage.save(str(path))
        saved.append(path)
    return saved


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    chunk_count = APP_STATE.get("rag_chunk_count") or rag_document_count()
    APP_STATE["rag_chunk_count"] = chunk_count
    APP_STATE["rag_ready"] = chunk_count > 0 or bool(APP_STATE.get("rag_ready"))
    if chunk_count > 0 and APP_STATE.get("rag_stage") in ("idle", None):
        APP_STATE["rag_stage"] = "ready"
    return jsonify(
        {
            "rag_ready": bool(APP_STATE["rag_ready"]) and chunk_count > 0,
            "rag_chunk_count": chunk_count,
            "rag_stage": APP_STATE.get("rag_stage") or "idle",
            "contract_ready": bool(APP_STATE["contract_ready"]),
            "contract_filename": APP_STATE["contract_filename"],
            "contract_sentence_count": len(APP_STATE["contract_sentences"]),
            "contract_stage": APP_STATE.get("contract_stage") or "idle",
            "reviewing": bool(APP_STATE.get("reviewing")),
        }
    )


@app.route("/api/upload_rag", methods=["POST"])
def upload_rag():
    if not current_api_key():
        return _missing_key_response()
    files = request.files.getlist("files")
    saved = _save_pdfs(files, RAG_UPLOAD_DIR)
    if not saved:
        return jsonify({"type": "error", "message": "PDF 파일만 업로드할 수 있습니다."}), 400

    app.logger.info("rag upload accepted files=%s", len(saved))
    APP_STATE["rag_stage"] = "extracting"
    key = current_api_key()
    return _sse_with_key(key, index_pdfs(saved), kind="rag")


@app.route("/api/upload_contract", methods=["POST"])
def upload_contract():
    files = request.files.getlist("file")
    saved = _save_pdfs(files, CONTRACT_UPLOAD_DIR)
    if not saved:
        return jsonify({"type": "error", "message": "계약서 PDF를 선택해 주세요."}), 400

    app.logger.info("contract upload accepted")
    APP_STATE["contract_stage"] = "extracting"
    return _sse_with_key(current_api_key(), split_contract_pdf(saved[0]), kind="contract")


@app.route("/api/review", methods=["POST"])
def review_contract():
    if not current_api_key():
        return _missing_key_response()
    sentences = list(APP_STATE.get("contract_sentences") or [])
    APP_STATE["reviewing"] = True
    key = current_api_key()
    return _sse_with_key(key, review_sentences(sentences), kind="review")


@app.route("/api/chat", methods=["POST"])
def chat():
    if not current_api_key():
        return _missing_key_response()
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"type": "error", "message": "질문을 입력해 주세요."}), 400

    key = current_api_key()

    @stream_with_context
    def generate():
        token = bind_api_key(key)
        try:
            yield _sse({"type": "start"})
            for token_text in stream_answer(question):
                yield _sse({"type": "token", "text": token_text})
            yield _sse({"type": "done"})
        except MissingApiKeyError:
            yield _sse({"type": "error", "message": MISSING_KEY_MESSAGE})
        except Exception as exc:
            app.logger.error("chat stream failed")
            yield _sse({"type": "error", "message": user_facing_openai_error(exc)})
        finally:
            reset_api_key(token)

    return _sse_response(generate())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="127.0.0.1", port=8765, debug=False, threaded=True, use_reloader=False)
