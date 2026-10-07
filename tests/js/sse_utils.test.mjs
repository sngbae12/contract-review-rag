import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { describe, it } from "node:test";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const Sse = require(path.join(__dirname, "../../static/js/sse_utils.js"));

function consumePieces(pieces) {
  let buffer = "";
  const events = [];
  for (let i = 0; i < pieces.length; i += 1) {
    const last = i === pieces.length - 1;
    if (last) {
      buffer += pieces[i];
      if (buffer.trim()) {
        const parsed = Sse.handleSseBuffer(`${buffer}\n\n`, (event) => events.push(event));
        if (parsed.terminal) return { events, terminal: parsed.terminal };
      }
      throw new Error(Sse.INCOMPLETE_STREAM);
    }
    buffer += pieces[i];
    const parsed = Sse.handleSseBuffer(buffer, (event) => events.push(event));
    buffer = parsed.rest;
    if (parsed.terminal) return { events, terminal: parsed.terminal };
  }
  throw new Error(Sse.INCOMPLETE_STREAM);
}

describe("sse_utils handleSseBuffer / terminal events", () => {
  it("returns done from the last buffer", () => {
    const { events, terminal } = consumePieces([
      'data: {"type":"token","text":"안녕"}\n\n',
      'data: {"type":"done"}\n',
    ]);
    assert.equal(terminal.type, "done");
    assert.equal(events[0].text, "안녕");
  });

  it("returns explicit error", () => {
    const { terminal } = consumePieces(['data: {"type":"error","message":"실패"}\n\n']);
    assert.equal(terminal.type, "error");
    assert.equal(terminal.message, "실패");
  });

  it("treats stream end without done as incomplete", () => {
    assert.throws(
      () => consumePieces(['data: {"type":"token","text":"중간"}\n\n']),
      (err) => err.message === Sse.INCOMPLETE_STREAM
    );
  });
});

describe("sse_utils consumeSSE with ReadableStream", () => {
  it("resolves on done and unlocks after error path", async () => {
    const chunks = [
      'data: {"type":"token","text":"가"}\n\n',
      'data: {"type":"done"}\n\n',
    ];
    let index = 0;
    const stream = new ReadableStream({
      pull(controller) {
        if (index >= chunks.length) {
          controller.close();
          return;
        }
        controller.enqueue(new TextEncoder().encode(chunks[index]));
        index += 1;
      },
    });
    const events = [];
    const terminal = await Sse.consumeSSE(
      { ok: true, status: 200, body: stream },
      (event) => events.push(event)
    );
    assert.equal(terminal.type, "done");
    assert.equal(events[0].text, "가");
  });

  it("rejects when the stream ends without a terminal event", async () => {
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode('data: {"type":"token","text":"끊김"}\n\n'));
        controller.close();
      },
    });
    await assert.rejects(
      () => Sse.consumeSSE({ ok: true, status: 200, body: stream }, () => {}),
      (err) => err.message === Sse.INCOMPLETE_STREAM
    );
  });
});

describe("keyboard and button unlock helpers", () => {
  it("keeps IME composing Enter from submitting", () => {
    assert.equal(Sse.shouldSubmitOnEnter({ key: "Enter", shiftKey: false, isComposing: true }), false);
    assert.equal(Sse.shouldSubmitOnEnter({ key: "Enter", shiftKey: false, keyCode: 229 }), false);
    assert.equal(Sse.shouldSubmitOnEnter({ key: "Enter", shiftKey: true, isComposing: false }), false);
    assert.equal(Sse.shouldSubmitOnEnter({ key: "Enter", shiftKey: false, isComposing: false }), true);
  });

  it("clears busy flags after unlock", () => {
    const unlocked = Sse.unlockUiFlags({
      ragUploading: true,
      contractUploading: true,
      reviewing: true,
      answering: true,
      ragReady: true,
    });
    assert.equal(unlocked.ragUploading, false);
    assert.equal(unlocked.contractUploading, false);
    assert.equal(unlocked.reviewing, false);
    assert.equal(unlocked.answering, false);
    assert.equal(unlocked.ragReady, true);
  });
});
