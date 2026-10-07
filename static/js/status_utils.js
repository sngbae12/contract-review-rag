/**
 * RAG/계약서 상태 문구. 브라우저와 Node 테스트에서 같은 구현을 쓴다.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.ContractReviewStatus = factory();
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  function formatRagStatus(input) {
    const stage = input.stage || "idle";
    const ready = Boolean(input.ready);
    const usable = Number(input.usableCount || 0);
    const incomplete = Number(input.incompleteCount || 0);
    const unverified = Number(input.unverifiedCount || 0);

    const warnings = [];
    if (unverified > 0) warnings.push(`미확인 ${unverified}`);
    if (incomplete > 0) warnings.push(`미완료 ${incomplete}`);
    const warnSuffix = warnings.length
      ? ` · ${warnings.join(" · ")} · 재업로드 권장`
      : "";

    if (stage === "extracting") return "텍스트 추출 중";
    if (stage === "splitting") return "분할 중";
    if (stage === "embedding") return "임베딩 중";
    if (stage === "processing") return "처리 중";

    if (stage === "error") {
      if (ready) {
        return `준비 완료 (${usable}조각)${warnSuffix} · 마지막 처리 실패`;
      }
      if (warnings.length) {
        return `${warnings.join(" · ")} · 재업로드 필요`;
      }
      return "처리 실패";
    }

    if (ready) {
      return `준비 완료 (${usable}조각)${warnSuffix}`;
    }
    if (warnings.length) {
      return `${warnings.join(" · ")} · 재업로드 필요`;
    }
    return "대기 중";
  }

  function formatContractStatus(input) {
    const stage = input.stage || "idle";
    const ready = Boolean(input.ready);
    const filename = input.filename || "";
    if (stage === "extracting" || stage === "processing") return "처리 중";
    if (stage === "error") return ready ? "준비됨 · 마지막 처리 실패" : "처리 실패";
    if (ready) return filename || "준비 완료";
    return "미업로드";
  }

  return {
    formatRagStatus,
    formatContractStatus,
  };
});
