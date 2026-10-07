/**
 * DOM에 의존하지 않는 SSE 파서.
 * 브라우저와 Node 테스트에서 같은 구현을 쓴다.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.ContractReviewSse = factory();
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  const INCOMPLETE_STREAM =
    "연결이 완료 신호 없이 종료되었습니다. 작업이 중단되었을 수 있습니다.";

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

  async function consumeSSE(response, onEvent) {
    if (!response.ok || !response.body) {
      let message = "요청 실패 (" + response.status + ")";
      try {
        const data = await response.json();
        if (data.message) message = data.message;
      } catch (_err) {
        /* ignore */
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
            const parsed = handleSseBuffer(buffer + "\n\n", onEvent);
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
        /* already closed */
      }
    }
  }

  function shouldSubmitOnEnter(event) {
    if (event.isComposing || event.keyCode === 229) return false;
    return event.key === "Enter" && !event.shiftKey;
  }

  function unlockUiFlags(flags) {
    return Object.assign({}, flags, {
      ragUploading: false,
      contractUploading: false,
      reviewing: false,
      answering: false,
    });
  }

  return {
    INCOMPLETE_STREAM: INCOMPLETE_STREAM,
    parseSseBlock: parseSseBlock,
    handleSseBuffer: handleSseBuffer,
    consumeSSE: consumeSSE,
    shouldSubmitOnEnter: shouldSubmitOnEnter,
    unlockUiFlags: unlockUiFlags,
  };
});
