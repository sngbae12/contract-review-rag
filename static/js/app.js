/**
 * 화면 동작
 * - 업로드/검토/답변 상태를 따로 관리한다.
 * - SSE는 done/error를 받으면 연결을 닫고, 완료 이벤트 없이 끝나면 오류로 처리한다.
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
const apiKeyInput = document.getElementById("openai-api-key");

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

const INCOMPLETE_STREAM = "연결이 완료 신호 없이 종료되었습니다. 작업이 중단되었을 수 있습니다.";

function getApiKey() {
  return apiKeyInput ? apiKeyInput.value.trim() : "";
}

function apiHeaders(extra) {
  const headers = extra ? { ...extra } : {};
  const key = getApiKey();
  if (key) headers["X-OpenAI-Api-Key"] = key;
  return headers;
}

function requireApiKey() {
  if (getApiKey()) return true;
  addTextMessage("bot", "OpenAI API 키가 없습니다. 왼쪽 입력란에 키를 넣은 뒤 다시 시도해 주세요.");
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
  if (stage === "error") return ready ? "준비됨 · 마지막 처리 실패" : "처리 실패";
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

function addBubble(role) {
  const wrap = document.createElement("article");
  wrap.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = `avatar ${role === "user" ? "user" : "bot"}`;
  avatar.textContent = role === "user" ? "나" : "AI";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  wrap.append(avatar, bubble);
  chatLog.appendChild(wrap);
  scrollToBottom();
  return bubble;
}

function addTextMessage(role, text) {
  const bubble = addBubble(role);
  bubble.textContent = text;
  return bubble;
}

function renderProgress(host, title, message, percent) {
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = title;

  const msg = document.createElement("div");
  msg.className = "progress-message";
  msg.textContent = message || "";

  const wrap = document.createElement("div");
  wrap.className = "progress-wrap";

  const labelRow = document.createElement("div");
  labelRow.className = "progress-label";
  const statusText = document.createElement("span");
  statusText.textContent = "상태";
  const percentText = document.createElement("span");
  percentText.className = "progress-percent";
  percentText.textContent = Number.isFinite(percent) ? `${percent}%` : "진행 중";
  labelRow.append(statusText, percentText);

  const track = document.createElement("div");
  track.className = "progress-track";
  const fill = document.createElement("div");
  fill.className = Number.isFinite(percent) ? "progress-fill" : "progress-fill indeterminate";
  if (Number.isFinite(percent)) fill.style.width = `${percent}%`;
  track.append(fill);
  wrap.append(labelRow, track);

  const keep = [...host.querySelectorAll(".review-list, .stream-note")];
  host.replaceChildren(meta, msg, wrap);
  keep.forEach((node) => host.appendChild(node));
}

function updateProgress(bubble, title, event) {
  const percent = Number.isFinite(event.percent) ? event.percent : undefined;
  const meta = bubble.querySelector(".meta");
  const msg = bubble.querySelector(".progress-message");
  const fill = bubble.querySelector(".progress-fill");
  const label = bubble.querySelector(".progress-percent");
  if (meta) meta.textContent = title;
  if (msg) msg.textContent = event.message || "";
  if (label) label.textContent = Number.isFinite(percent) ? `${percent}%` : "진행 중";
  if (fill && Number.isFinite(percent)) {
    fill.classList.remove("indeterminate");
    fill.style.width = `${percent}%`;
  }
  if (!meta || !msg || !fill) {
    renderProgress(bubble, title, event.message || "", percent);
  }
}

function appendStreamNote(bubble, text) {
  const note = document.createElement("div");
  note.className = "stream-note";
  note.textContent = text;
  bubble.appendChild(note);
  scrollToBottom();
  return note;
}

function parseSseBlock(part) {
  const line = part.split("\n").find((item) => item.startsWith("data: "));
  if (!line) return null;
  return JSON.parse(line.slice(6));
}

function handleSseBuffer(chunk, onEvent) {
  const parts = chunk.split("\n\n");
  const rest = parts.pop();
  for (const part of parts) {
    let event;
    try {
      event = parseSseBlock(part);
    } catch (_err) {
      throw new Error("서버 응답 형식을 해석하지 못했습니다.");
    }
    if (!event) continue;
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
}

/** 서버가 보낸 SSE를 읽고, done/error에서 종료한다. 완료 이벤트 없이 끝나면 오류다. */
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

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        buffer += decoder.decode();
        if (buffer.trim()) {
          const parsed = handleSseBuffer(`${buffer}\n\n`, onEvent);
          if (parsed.terminal) return parsed.terminal;
        }
        throw new Error(INCOMPLETE_STREAM);
      }
      buffer += decoder.decode(value, { stream: true });
      const parsed = handleSseBuffer(buffer, onEvent);
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
}

function scrollToBottom() {
  chatLog.scrollTop = chatLog.scrollHeight;
}

function showWelcome() {
  addTextMessage(
    "bot",
    "안녕하세요. LLM 기반 계약서 검토 도우미입니다.\n\n" +
      "왼쪽에서 가이드라인/약관 PDF를 올린 뒤 계약서를 업로드하세요.\n" +
      "계약서 업로드가 끝나면 [계약서 검토] 버튼이 나타납니다.\n" +
      "OpenAI API 키를 왼쪽 입력란에 넣은 뒤 작업을 시작하세요.\n" +
      "아래 칸에는 일반적인 질문을 바로 할 수 있습니다."
  );
}

async function refreshStatus() {
  const res = await fetch("/api/status");
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

if (apiKeyInput) {
  apiKeyInput.addEventListener("change", () => {
    refreshStatus().catch(() => renderSidebarStatus());
  });
}

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
  addTextMessage("user", `RAG 파일 업로드: ${names}`);
  const bubble = addBubble("bot");
  renderProgress(bubble, "RAG 인덱싱", "업로드를 시작합니다...");

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
        const msg = bubble.querySelector(".progress-message");
        if (msg) msg.textContent = event.message || "처리에 실패했습니다.";
        else bubble.textContent = event.message || "처리에 실패했습니다.";
        ui.ragStage = "error";
        renderSidebarStatus();
        return;
      }
      if (event.type === "progress") {
        updateProgress(bubble, "RAG 인덱싱", event);
        renderSidebarStatus();
      }
      if (event.type === "done") {
        ui.ragReady = Number(event.collection_count || ui.ragChunkCount) > 0;
        ui.ragStage = ui.ragReady ? "ready" : "idle";
        if (event.collection_count) ui.ragChunkCount = event.collection_count;
        updateProgress(bubble, "RAG 인덱싱", { message: event.message, percent: 100 });
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
    appendStreamNote(bubble, `오류: ${err.message}`);
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
  addTextMessage("user", `계약서 업로드: ${file.name}`);
  const bubble = addBubble("bot");
  renderProgress(bubble, "계약서 처리", "업로드를 시작합니다...");

  const form = new FormData();
  form.append("file", file);
  contractInput.value = "";

  setFlag("contractUploading", true);
  ui.contractStage = "extracting";
  renderSidebarStatus();
  try {
    const response = await fetch("/api/upload_contract", { method: "POST", body: form });
    const terminal = await consumeSSE(response, (event) => {
      if (event.stage) ui.contractStage = event.stage;
      if (event.type === "error") {
        const msg = bubble.querySelector(".progress-message");
        if (msg) msg.textContent = event.message || "처리에 실패했습니다.";
        else bubble.textContent = event.message || "처리에 실패했습니다.";
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
        updateProgress(bubble, "계약서 처리", { message: event.message, percent: 100 });
        btnReview.classList.remove("hidden");
      }
      scrollToBottom();
    });
    if (terminal && terminal.type === "error") {
      ui.contractStage = "error";
    }
    try {
      await refreshStatus();
    } catch (_err) {
      renderSidebarStatus();
    }
  } catch (err) {
    ui.contractStage = "error";
    appendStreamNote(bubble, `오류: ${err.message}`);
    renderSidebarStatus();
  } finally {
    setFlag("contractUploading", false);
  }
});

function locationLabel(event) {
  const parts = [];
  if (event.source_file) parts.push(event.source_file);
  if (event.page) parts.push(`p.${event.page}`);
  if (event.clause) parts.push(event.clause);
  return parts.join(" · ");
}

function appendReviewItem(host, event) {
  const item = document.createElement("div");
  const status = event.item_status || (event.has_issue ? "reviewed" : "reviewed");
  item.className = `review-item ${status}`;

  const loc = locationLabel(event);
  if (loc) {
    const locEl = document.createElement("div");
    locEl.className = "review-location";
    locEl.textContent = loc;
    item.appendChild(locEl);
  }

  const original = document.createElement("div");
  original.className = "original";
  const originalTag = document.createElement("span");
  originalTag.className = "tag";
  originalTag.textContent = "[원문]";
  original.append(originalTag, document.createTextNode(event.original || ""));
  item.appendChild(original);

  if (status === "search_error") {
    const fail = document.createElement("div");
    fail.className = "review-fail";
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = "[검토 실패]";
    fail.append(tag, document.createTextNode(event.reason || "검색 오류로 검토하지 못했습니다. 다시 시도해 주세요."));
    item.appendChild(fail);
  } else if (status === "no_evidence") {
    const miss = document.createElement("div");
    miss.className = "review-missing";
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = "[근거 부족]";
    miss.append(tag, document.createTextNode(event.reason || "관련 가이드라인을 찾지 못했습니다."));
    item.appendChild(miss);
  } else if (status === "skipped_short") {
    const skip = document.createElement("div");
    skip.className = "reason";
    skip.textContent = event.reason || "짧은 조각이라 검토하지 않았습니다.";
    item.appendChild(skip);
  } else if (event.has_issue && event.revised) {
    const revised = document.createElement("div");
    revised.className = "revised";
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = "[수정문구]";
    revised.append(tag, document.createTextNode(event.revised));
    item.appendChild(revised);
    if (event.reason) {
      const reason = document.createElement("div");
      reason.className = "reason";
      reason.textContent = `사유: ${event.reason}`;
      item.appendChild(reason);
    }
  }

  host.appendChild(item);
}

btnReview.addEventListener("click", async () => {
  if (btnReview.disabled) return;
  if (!requireApiKey()) return;
  if (!ui.contractReady) {
    addTextMessage("bot", "계약서가 아직 준비되지 않았습니다. 먼저 계약서 PDF를 업로드해 주세요.");
    return;
  }
  if (!ui.ragReady) {
    addTextMessage("bot", "RAG 문서가 아직 준비되지 않았습니다. 먼저 가이드라인/약관 PDF를 업로드해 주세요.");
    return;
  }

  addTextMessage("user", "계약서 검토를 시작합니다.");
  const bubble = addBubble("bot");
  renderProgress(bubble, "계약서 검토", "준비 중...");
  const itemsHost = document.createElement("div");
  itemsHost.className = "review-list";
  bubble.appendChild(itemsHost);

  setFlag("reviewing", true);
  try {
    const response = await fetch("/api/review", { method: "POST", headers: apiHeaders() });
    const terminal = await consumeSSE(response, (event) => {
      if (event.type === "error") {
        appendStreamNote(bubble, event.message || "검토에 실패했습니다.");
        return;
      }
      if (event.type === "progress" || event.type === "item") {
        updateProgress(bubble, event.message || "계약서 검토", event);
      }
      if (event.type === "item") {
        appendReviewItem(itemsHost, event);
      }
      if (event.type === "done") {
        updateProgress(bubble, "계약서 검토", { message: event.message, percent: 100 });
      }
      scrollToBottom();
    });
    if (terminal && terminal.type === "error") {
      /* 항목은 유지하고 안내만 추가된 상태 */
    }
  } catch (err) {
    appendStreamNote(bubble, `오류: ${err.message}`);
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
  addTextMessage("user", question);
  const bubble = addTextMessage("bot", "");

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
    const terminal = await consumeSSE(response, (evt) => {
      if (evt.type === "token") {
        bubble.textContent += evt.text;
        scrollToBottom();
      }
      if (evt.type === "error") {
        bubble.textContent = evt.message || "답변 생성에 실패했습니다.";
      }
    });
    if (terminal && terminal.type === "done" && !bubble.textContent) {
      bubble.textContent = "답변을 생성하지 못했습니다. 다시 시도해 주세요.";
    }
  } catch (err) {
    if (bubble.textContent) {
      bubble.textContent += `\n\n오류: ${err.message}`;
    } else {
      bubble.textContent = `오류: ${err.message}`;
    }
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

showWelcome();
applyUi();
refreshStatus().catch(() => {
  renderSidebarStatus();
});
