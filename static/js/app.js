/**
 * 화면 동작
 * - 업로드/검토/답변 상태를 따로 관리한다.
 * - SSE는 done/error를 받으면 연결을 닫고 UI를 복구한다.
 * - 문서 처리 중에도 질문 초안은 입력할 수 있다.
 */

const chatLog = document.getElementById("chat-log");
const ragInput = document.getElementById("rag-input");
const contractInput = document.getElementById("contract-input");
const btnRag = document.getElementById("btn-rag");
const btnContract = document.getElementById("btn-contract");
const btnReview = document.getElementById("btn-review");
const chatForm = document.getElementById("chat-form");
const questionBox = document.getElementById("question");
const btnSend = document.getElementById("btn-send");
const ragStatus = document.getElementById("rag-status");
const contractStatus = document.getElementById("contract-status");
const composerHint = document.getElementById("composer-hint");

const ui = {
  ragUploading: false,
  contractUploading: false,
  reviewing: false,
  answering: false,
  ragStage: "idle",
  contractStage: "idle",
  ragReady: false,
  ragChunkCount: 0,
  contractReady: false,
  contractFilename: "",
};


function getApiKey() {
  const el = document.getElementById("openai-api-key");
  return el ? el.value.trim() : "";
}

function apiHeaders(extra) {
  const headers = extra ? { ...extra } : {};
  const key = getApiKey();
  if (key) headers["X-OpenAI-Api-Key"] = key;
  return headers;
}

function requireApiKey() {
  if (getApiKey()) return true;
  addMessage("bot", "OpenAI API 키가 없습니다. 왼쪽 입력란에 키를 넣은 뒤 다시 시도해 주세요.");
  return false;
}

function isDocBusy() {
  return ui.ragUploading || ui.contractUploading || ui.reviewing;
}

function applyUi() {
  const docBusy = isDocBusy();
  btnRag.disabled = docBusy || ui.answering;
  btnContract.disabled = docBusy || ui.answering;
  if (btnReview) btnReview.disabled = docBusy || ui.answering;

  questionBox.disabled = false;
  questionBox.readOnly = false;

  btnSend.disabled = ui.answering;

  if (!composerHint) return;
  if (ui.answering) {
    composerHint.hidden = false;
    composerHint.textContent = "답변을 생성하는 중입니다. 생성이 끝나면 다시 전송할 수 있습니다.";
  } else if (ui.ragUploading || ui.contractUploading) {
    composerHint.hidden = false;
    composerHint.textContent = "문서를 처리 중입니다. 질문 초안은 지금 작성할 수 있습니다.";
  } else {
    composerHint.hidden = true;
    composerHint.textContent = "";
  }
}

function stageLabel(kind, stage, ready, extra) {
  if (stage === "extracting") return "텍스트 추출 중";
  if (stage === "splitting") return "분할 중";
  if (stage === "embedding") return "임베딩 중";
  if (stage === "processing") return "처리 중";
  if (stage === "error") return ready ? `준비됨 · 마지막 처리 실패` : "처리 실패";
  if (kind === "rag") {
    if (ready) return extra ? `준비 완료 (${extra}조각)` : "준비 완료";
    return "대기 중";
  }
  if (ready) return extra || "준비 완료";
  return "미업로드";
}

function renderSidebarStatus() {
  ragStatus.textContent = stageLabel("rag", ui.ragStage, ui.ragReady, ui.ragChunkCount);
  contractStatus.textContent = stageLabel(
    "contract",
    ui.contractStage,
    ui.contractReady,
    ui.contractFilename
  );
  if (ui.contractReady) {
    btnReview.classList.remove("hidden");
  } else {
    btnReview.classList.add("hidden");
  }
}

function setFlag(name, value) {
  ui[name] = value;
  applyUi();
  renderSidebarStatus();
}

/** 서버가 보낸 SSE를 읽고, done/error에서 즉시 종료한다. */
async function consumeSSE(response, onEvent) {
  if (!response.ok || !response.body) {
    let message = `요청 실패 (${response.status})`;
    try {
      const data = await response.json();
      if (data.message) message = data.message;
    } catch (_err) {
      /* JSON이 아니면 기본 메시지 사용 */
    }
    throw new Error(message);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  const handleChunk = (chunk) => {
    const parts = chunk.split("\n\n");
    const rest = parts.pop();
    for (const part of parts) {
      const line = part.split("\n").find((item) => item.startsWith("data: "));
      if (!line) continue;
      let event;
      try {
        event = JSON.parse(line.slice(6));
      } catch (_err) {
        throw new Error("서버 응답 형식을 해석하지 못했습니다.");
      }
      try {
        onEvent(event);
      } catch (err) {
        console.error(err);
      }
      if (event.type === "done" || event.type === "error") {
        return { rest: "", terminal: event };
      }
    }
    return { rest, terminal: null };
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        buffer += decoder.decode();
        if (buffer.trim()) handleChunk(`${buffer}\n\n`);
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      const parsed = handleChunk(buffer);
      buffer = parsed.rest;
      if (parsed.terminal) return parsed.terminal;
    }
  } finally {
    try {
      await reader.cancel();
    } catch (_err) {
      /* 이미 닫힌 스트림 */
    }
  }
  return null;
}

function scrollToBottom() {
  chatLog.scrollTop = chatLog.scrollHeight;
}

function addMessage(role, html) {
  const wrap = document.createElement("article");
  wrap.className = `message ${role}`;
  wrap.innerHTML = `
    <div class="avatar ${role === "user" ? "user" : "bot"}">${role === "user" ? "나" : "AI"}</div>
    <div class="bubble">${html}</div>
  `;
  chatLog.appendChild(wrap);
  scrollToBottom();
  return wrap.querySelector(".bubble");
}

function progressHtml(title, message, percent) {
  const hasPercent = Number.isFinite(percent);
  const fillClass = hasPercent ? "progress-fill" : "progress-fill indeterminate";
  const width = hasPercent ? ` style="width:${percent}%"` : "";
  const label = hasPercent ? `${percent}%` : "진행 중";
  return `
    <div class="meta">${title}</div>
    <div>${message || ""}</div>
    <div class="progress-wrap">
      <div class="progress-label"><span>상태</span><span>${label}</span></div>
      <div class="progress-track"><div class="${fillClass}"${width}></div></div>
    </div>
  `;
}

function updateProgress(bubble, title, event) {
  const percent = Number.isFinite(event.percent) ? event.percent : undefined;
  bubble.innerHTML = progressHtml(title, event.message || "", percent);
}

function showWelcome() {
  addMessage(
    "bot",
    "안녕하세요. LLM 기반 계약서 검토 도우미입니다.\n\n" +
      "왼쪽에서 가이드라인/약관 PDF를 올린 뒤 계약서를 업로드하세요.\n" +
      "계약서 업로드가 끝나면 [계약서 검토] 버튼이 나타납니다.\n" +
      "OpenAI API 키를 왼쪽 입력란에 넣은 뒤 작업을 시작하세요.\n" +
      "아래 칸에는 일반적인 질문을 바로 할 수 있습니다."
  );
}

async function refreshStatus() {
  const res = await fetch("/api/status", { headers: apiHeaders() });
  if (!res.ok) throw new Error("상태 조회에 실패했습니다.");
  const data = await res.json();
  ui.ragReady = Boolean(data.rag_ready);
  ui.ragChunkCount = data.rag_chunk_count || 0;
  ui.ragStage = data.rag_stage || (ui.ragReady ? "ready" : "idle");
  ui.contractReady = Boolean(data.contract_ready);
  ui.contractFilename = data.contract_filename || "";
  ui.contractStage = data.contract_stage || (ui.contractReady ? "ready" : "idle");
  renderSidebarStatus();
}

btnRag.addEventListener("click", () => {
  if (btnRag.disabled) return;
  ragInput.click();
});
btnContract.addEventListener("click", () => {
  if (btnContract.disabled) return;
  contractInput.click();
});

ragInput.addEventListener("change", async () => {
  if (!ragInput.files.length) return;
  if (isDocBusy() || ui.answering) {
    ragInput.value = "";
    return;
  }
  if (!requireApiKey()) {
    ragInput.value = "";
    return;
  }
  const names = Array.from(ragInput.files).map((f) => f.name).join(", ");
  addMessage("user", `RAG 파일 업로드: ${names}`);
  const bubble = addMessage("bot", progressHtml("RAG 인덱싱", "업로드를 시작합니다..."));

  const form = new FormData();
  Array.from(ragInput.files).forEach((file) => form.append("files", file));
  ragInput.value = "";

  setFlag("ragUploading", true);
  ui.ragStage = "extracting";
  renderSidebarStatus();
  try {
    const response = await fetch("/api/upload_rag", { method: "POST", body: form, headers: apiHeaders() });
    const terminal = await consumeSSE(response, (event) => {
      if (event.stage) ui.ragStage = event.stage;
      if (event.type === "error") {
        bubble.innerHTML = event.message;
        ui.ragStage = "error";
        renderSidebarStatus();
        return;
      }
      if (event.type === "progress") {
        updateProgress(bubble, "RAG 인덱싱", event);
        renderSidebarStatus();
      }
      if (event.type === "done") {
        ui.ragReady = true;
        ui.ragStage = "ready";
        if (event.collection_count) ui.ragChunkCount = event.collection_count;
        bubble.innerHTML = progressHtml("RAG 인덱싱", event.message, 100);
      }
      scrollToBottom();
    });
    if (terminal && terminal.type === "error") {
      ui.ragStage = "error";
    }
    try {
      await refreshStatus();
    } catch (_err) {
      renderSidebarStatus();
    }
  } catch (err) {
    ui.ragStage = "error";
    bubble.textContent = `오류: ${err.message}`;
    renderSidebarStatus();
  } finally {
    setFlag("ragUploading", false);
  }
});

contractInput.addEventListener("change", async () => {
  if (!contractInput.files.length) return;
  if (isDocBusy() || ui.answering) {
    contractInput.value = "";
    return;
  }
  const file = contractInput.files[0];
  addMessage("user", `계약서 업로드: ${file.name}`);
  const bubble = addMessage("bot", progressHtml("계약서 처리", "업로드를 시작합니다..."));

  const form = new FormData();
  form.append("file", file);
  contractInput.value = "";

  setFlag("contractUploading", true);
  ui.contractStage = "extracting";
  renderSidebarStatus();
  try {
    const response = await fetch("/api/upload_contract", { method: "POST", body: form });
    await consumeSSE(response, (event) => {
      if (event.stage) ui.contractStage = event.stage;
      if (event.type === "error") {
        bubble.innerHTML = event.message;
        ui.contractStage = "error";
        renderSidebarStatus();
        return;
      }
      if (event.type === "progress") {
        updateProgress(bubble, "계약서 처리", event);
        renderSidebarStatus();
      }
      if (event.type === "done") {
        ui.contractReady = true;
        ui.contractStage = "ready";
        ui.contractFilename = event.filename || file.name;
        bubble.innerHTML = progressHtml("계약서 처리", event.message, 100);
        btnReview.classList.remove("hidden");
      }
      scrollToBottom();
    });
    try {
      await refreshStatus();
    } catch (_err) {
      renderSidebarStatus();
    }
  } catch (err) {
    ui.contractStage = "error";
    bubble.textContent = `오류: ${err.message}`;
    renderSidebarStatus();
  } finally {
    setFlag("contractUploading", false);
  }
});

btnReview.addEventListener("click", async () => {
  if (btnReview.disabled) return;
  if (!requireApiKey()) return;
  if (!ui.contractReady) {
    addMessage("bot", "계약서가 아직 준비되지 않았습니다. 먼저 계약서 PDF를 업로드해 주세요.");
    return;
  }
  if (!ui.ragReady) {
    addMessage("bot", "RAG 문서가 아직 준비되지 않았습니다. 먼저 가이드라인/약관 PDF를 업로드해 주세요.");
    return;
  }

  addMessage("user", "계약서 검토를 시작합니다.");
  const bubble = addMessage("bot", progressHtml("계약서 검토", "준비 중..."));
  const itemsHost = document.createElement("div");
  itemsHost.className = "review-list";
  bubble.appendChild(itemsHost);

  setFlag("reviewing", true);
  try {
    const response = await fetch("/api/review", { method: "POST", headers: apiHeaders() });
    await consumeSSE(response, (event) => {
      if (event.type === "error") {
        const note = document.createElement("div");
        note.textContent = event.message;
        bubble.appendChild(note);
        return;
      }
      if (event.type === "progress" || event.type === "item") {
        const label = bubble.querySelector(".progress-label span:last-child");
        const fill = bubble.querySelector(".progress-fill");
        const meta = bubble.querySelector(".meta");
        if (Number.isFinite(event.percent) && label) label.textContent = `${event.percent}%`;
        if (Number.isFinite(event.percent) && fill) {
          fill.classList.remove("indeterminate");
          fill.style.width = `${event.percent}%`;
        }
        if (meta) meta.textContent = event.message || "계약서 검토";
      }
      if (event.type === "item") {
        const item = document.createElement("div");
        item.className = "review-item";
        const original = escapeHtml(event.original || "");
        let html = `<div class="original"><span class="tag">[원문]</span>${original}</div>`;
        if (event.has_issue && event.revised) {
          html += `<div class="revised"><span class="tag">[수정문구]</span>${escapeHtml(event.revised)}</div>`;
          if (event.reason) {
            html += `<div class="reason">사유: ${escapeHtml(event.reason)}</div>`;
          }
        }
        item.innerHTML = html;
        itemsHost.appendChild(item);
      }
      if (event.type === "done") {
        const fill = bubble.querySelector(".progress-fill");
        const label = bubble.querySelector(".progress-label span:last-child");
        if (fill) {
          fill.classList.remove("indeterminate");
          fill.style.width = "100%";
        }
        if (label) label.textContent = "100%";
        const summary = document.createElement("div");
        summary.style.marginTop = "12px";
        summary.textContent = event.message;
        bubble.appendChild(summary);
      }
      scrollToBottom();
    });
  } catch (err) {
    const note = document.createElement("div");
    note.textContent = `오류: ${err.message}`;
    bubble.appendChild(note);
  } finally {
    setFlag("reviewing", false);
  }
});

function submitQuestion() {
  if (ui.answering) return;
  const question = questionBox.value.trim();
  if (!question) return;
  if (!requireApiKey()) return;

  questionBox.value = "";
  autoResize();
  addMessage("user", escapeHtml(question));
  const bubble = addMessage("bot", "");

  setFlag("answering", true);
  sendQuestion(question, bubble);
}

async function sendQuestion(question, bubble) {
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: apiHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ question }),
    });
    await consumeSSE(response, (evt) => {
      if (evt.type === "token") {
        bubble.textContent += evt.text;
        scrollToBottom();
      }
      if (evt.type === "error") {
        bubble.textContent = evt.message;
      }
    });
    if (!bubble.textContent) {
      bubble.textContent = "답변을 생성하지 못했습니다. 다시 시도해 주세요.";
    }
  } catch (err) {
    bubble.textContent = `오류: ${err.message}`;
  } finally {
    setFlag("answering", false);
    questionBox.focus();
  }
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  submitQuestion();
});

questionBox.addEventListener("keydown", (event) => {
  if (event.isComposing || event.keyCode === 229) return;
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    submitQuestion();
  }
});

questionBox.addEventListener("input", autoResize);

function autoResize() {
  questionBox.style.height = "auto";
  questionBox.style.height = `${Math.min(questionBox.scrollHeight, 160)}px`;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

showWelcome();
applyUi();
refreshStatus().catch(() => {
  renderSidebarStatus();
});
