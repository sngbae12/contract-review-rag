import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { describe, it } from "node:test";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const Status = require(path.join(__dirname, "../../static/js/status_utils.js"));

describe("status_utils formatRagStatus", () => {
  it("shows unverified count when stage is error and not ready", () => {
    const text = Status.formatRagStatus({
      stage: "error",
      ready: false,
      usableCount: 0,
      incompleteCount: 0,
      unverifiedCount: 1,
    });
    assert.match(text, /미확인 1/);
    assert.match(text, /재업로드/);
    assert.doesNotMatch(text, /^처리 실패$/);
  });

  it("keeps unverified warning when ready", () => {
    const text = Status.formatRagStatus({
      stage: "ready",
      ready: true,
      usableCount: 5,
      incompleteCount: 0,
      unverifiedCount: 2,
    });
    assert.match(text, /준비 완료 \(5조각\)/);
    assert.match(text, /미확인 2/);
    assert.match(text, /재업로드 권장/);
  });

  it("shows incomplete with usable chunks", () => {
    const text = Status.formatRagStatus({
      stage: "ready",
      ready: true,
      usableCount: 3,
      incompleteCount: 4,
      unverifiedCount: 1,
    });
    assert.match(text, /검토 가능|준비 완료 \(3조각\)/);
    assert.match(text, /미확인 1/);
    assert.match(text, /미완료 4/);
  });

  it("keeps progress labels without hiding as ready", () => {
    assert.equal(
      Status.formatRagStatus({
        stage: "embedding",
        ready: false,
        usableCount: 0,
        incompleteCount: 1,
        unverifiedCount: 0,
      }),
      "임베딩 중"
    );
  });

  it("shows last failure while ready and warnings", () => {
    const text = Status.formatRagStatus({
      stage: "error",
      ready: true,
      usableCount: 2,
      incompleteCount: 1,
      unverifiedCount: 0,
    });
    assert.match(text, /준비 완료 \(2조각\)/);
    assert.match(text, /미완료 1/);
    assert.match(text, /마지막 처리 실패/);
  });
});
