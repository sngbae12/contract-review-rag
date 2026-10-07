"""
계약서 검토 에이전트 (LangGraph)
흐름: 검토 단위 → RAG 유사 문서 검색 → (검색 성공 시에만) LLM 위배/보완 판단
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Literal, TypedDict

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from tqdm import tqdm

from config import LLM_MODEL, MIN_REVIEW_LENGTH
from services.openai_runtime import require_api_key, user_facing_openai_error
from services.rag_service import rag_document_count, similarity_search
from services.text_split import detect_clause, page_number_from_metadata, split_contract_text

RetrieveStatus = Literal["ok", "empty", "error"]
ItemStatus = Literal["reviewed", "search_error", "no_evidence", "skipped_short"]


class ReviewDecision(BaseModel):
    """LLM이 반드시 이 형식으로만 답하도록 고정한다."""

    has_issue: bool = Field(description="가이드라인/약관 위배 또는 보완이 필요하면 true")
    revised_sentence: str = Field(description="수정이 필요하면 고친 문구, 없으면 원문과 동일")
    reason: str = Field(description="판단 이유를 한두 문장으로")


class ReviewState(TypedDict):
    """LangGraph가 노드 사이에 주고받는 상태."""

    text: str
    context: str
    retrieve_status: RetrieveStatus
    item_status: ItemStatus
    has_issue: bool
    revised_text: str
    reason: str


def _llm() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=require_api_key())


def split_contract_pdf(
    pdf_path: Path,
    original_filename: str | None = None,
) -> Generator[dict, None, None]:
    """
    계약서 PDF를 조항·문단 단위로 나눈다. 긴 조항만 추가 분할한다.
    성공하기 전에는 기존 계약서 상태를 덮어쓰지 않는다.
    """
    display_name = original_filename or pdf_path.name
    try:
        yield {
            "type": "progress",
            "stage": "extracting",
            "message": "텍스트 추출 중입니다...",
        }

        try:
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
        except Exception:
            yield {
                "type": "error",
                "stage": "error",
                "message": f"{display_name} 에서 텍스트를 읽지 못했습니다. 파일이 손상되었거나 PDF가 아닐 수 있습니다.",
            }
            return

        if not documents:
            yield {
                "type": "error",
                "stage": "error",
                "message": "계약서 PDF에서 텍스트를 읽지 못했습니다.",
            }
            return

        yield {
            "type": "progress",
            "stage": "splitting",
            "message": f"{display_name} 추출 완료 (페이지 {len(documents)}). 검토 단위로 나눕니다...",
        }

        units: list[dict] = []
        for document in documents:
            page = page_number_from_metadata(document.metadata)
            for piece in split_contract_text(document.page_content or ""):
                units.append(
                    {
                        "text": piece,
                        "source_file": display_name,
                        "page": page,
                        "clause": detect_clause(piece),
                    }
                )

        if not units:
            yield {
                "type": "error",
                "stage": "error",
                "message": "계약서에서 검토할 텍스트를 찾지 못했습니다.",
            }
            return

        yield {
            "type": "done",
            "stage": "ready",
            "percent": 100,
            "message": (
                f"계약서 업로드가 완료되었습니다.\n"
                f"- 파일: {display_name}\n"
                f"- 페이지: {len(documents)}쪽\n"
                f"- 검토 단위: {len(units)}개\n"
                f"왼쪽의 [계약서 검토] 버튼을 누르면 단위별 검토를 시작합니다."
            ),
            "filename": display_name,
            "page_count": len(documents),
            "unit_count": len(units),
            "units": units,
        }
    except Exception:
        yield {
            "type": "error",
            "stage": "error",
            "message": "계약서 처리 중 오류가 발생했습니다. 다시 시도해 주세요.",
        }


def _retrieve_node(state: ReviewState) -> dict:
    """노드 1: 현재 검토 단위와 비슷한 가이드라인/약관을 찾는다."""
    query = state["text"]
    try:
        docs = similarity_search(query)
    except Exception:
        return {
            "retrieve_status": "error",
            "item_status": "search_error",
            "context": "",
            "has_issue": False,
            "revised_text": query,
            "reason": "가이드라인 검색에 실패했습니다. API 키와 네트워크를 확인한 뒤 다시 검토해 주세요.",
        }

    if not docs:
        return {
            "retrieve_status": "empty",
            "item_status": "no_evidence",
            "context": "",
            "has_issue": False,
            "revised_text": query,
            "reason": "관련 가이드라인 조각을 찾지 못해 근거가 부족합니다. 업로드한 문서를 근거로 판단하지 않았습니다.",
        }

    lines = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page")
        clause = doc.metadata.get("clause") or ""
        location = str(source)
        if page:
            location += f", p.{page}"
        if clause:
            location += f", {clause}"
        lines.append(f"[{i}] ({location}) {doc.page_content.strip()}")
    return {
        "retrieve_status": "ok",
        "item_status": "reviewed",
        "context": "\n".join(lines),
    }


def _route_after_retrieve(state: ReviewState) -> str:
    if state.get("retrieve_status") == "ok":
        return "analyze"
    return "end"


def run_analysis(text: str, context: str) -> dict:
    """검색에 성공한 항목만 LLM으로 수정 문구를 만든다."""
    structured_llm = _llm().with_structured_output(ReviewDecision)
    prompt = f"""당신은 한국어 계약서 검토 전문가입니다.
아래 계약서 내용을 가이드라인/약관과 비교해 위배하거나 보완이 필요한지 판단하세요.

[가이드라인/약관에서 검색된 관련 내용]
{context}

[계약서 내용]
{text}

판단 규칙:
- 검색된 가이드라인·약관에 어긋나거나, 한쪽에게 부당하게 불리하거나, 의미가 뒤집힌 조항이면 has_issue=true
- 수정할 때는 원문의 번호·문체·형식은 유지하고, 문제가 되는 표현만 고친다
- 이상이 없으면 has_issue=false 이고 revised_sentence는 원문과 동일하게 둔다
- 검색된 내용에 없는 가이드라인을 있는 것처럼 인용하지 않는다
"""
    decision = structured_llm.invoke(prompt)
    revised = decision.revised_sentence.strip() or text
    has_issue = bool(decision.has_issue) and revised != text
    return {
        "item_status": "reviewed",
        "has_issue": has_issue,
        "revised_text": revised,
        "reason": decision.reason,
    }


def _analyze_node(state: ReviewState) -> dict:
    return run_analysis(state["text"], state.get("context") or "")


def build_review_graph():
    """검색 성공 시에만 분석으로 이어지는 LangGraph를 컴파일한다."""
    graph = StateGraph(ReviewState)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("analyze", _analyze_node)
    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {"analyze": "analyze", "end": END},
    )
    graph.add_edge("analyze", END)
    return graph.compile()


REVIEW_GRAPH = build_review_graph()


def _unit_text(unit) -> str:
    if isinstance(unit, dict):
        return str(unit.get("text") or "").strip()
    return str(unit).strip()


def review_sentences(units: list) -> Generator[dict, None, None]:
    """
    계약서 검토 단위를 하나씩 검토하고, 끝날 때마다 이벤트를 보낸다.
    검색 오류 항목은 LLM 분석을 하지 않는다.
    """
    if rag_document_count() == 0:
        yield {
            "type": "error",
            "message": "RAG 문서가 없습니다. 먼저 왼쪽에서 가이드라인/약관 PDF를 업로드하세요.",
        }
        return

    if not units:
        yield {"type": "error", "message": "검토할 계약서 내용이 없습니다. 계약서를 다시 업로드하세요."}
        return

    total = len(units)
    yield {
        "type": "progress",
        "percent": 0,
        "message": f"계약서 검토를 시작합니다. 총 {total}개 단위입니다.",
        "current": 0,
        "total": total,
    }

    issue_count = 0
    reviewed_count = 0
    search_error_count = 0
    no_evidence_count = 0
    skipped_count = 0

    try:
        with tqdm(total=total, desc="계약서 검토", ncols=80) as pbar:
            for index, unit in enumerate(units, start=1):
                text = _unit_text(unit)
                source_file = unit.get("source_file", "") if isinstance(unit, dict) else ""
                page = unit.get("page") if isinstance(unit, dict) else None
                clause = unit.get("clause", "") if isinstance(unit, dict) else ""

                if len(text) < MIN_REVIEW_LENGTH:
                    item_status: ItemStatus = "skipped_short"
                    result = {
                        "item_status": item_status,
                        "has_issue": False,
                        "revised_text": text,
                        "reason": "검토 대상이 아닌 짧은 조각입니다.",
                    }
                    skipped_count += 1
                else:
                    result = REVIEW_GRAPH.invoke(
                        {
                            "text": text,
                            "context": "",
                            "retrieve_status": "ok",
                            "item_status": "reviewed",
                            "has_issue": False,
                            "revised_text": text,
                            "reason": "",
                        }
                    )
                    item_status = result.get("item_status") or "reviewed"
                    if item_status == "search_error":
                        search_error_count += 1
                    elif item_status == "no_evidence":
                        no_evidence_count += 1
                    else:
                        reviewed_count += 1
                        if result.get("has_issue"):
                            issue_count += 1

                pbar.update(1)
                percent = int(index / total * 100)
                yield {
                    "type": "item",
                    "percent": percent,
                    "current": index,
                    "total": total,
                    "message": f"검토 중... {index}/{total}",
                    "original": text,
                    "revised": result.get("revised_text") or text,
                    "has_issue": bool(result.get("has_issue")) and item_status == "reviewed",
                    "reason": result.get("reason") or "",
                    "item_status": item_status,
                    "source_file": source_file,
                    "page": page,
                    "clause": clause,
                }
    except Exception as exc:
        yield {
            "type": "error",
            "stage": "error",
            "message": user_facing_openai_error(exc),
        }
        return

    yield {
        "type": "done",
        "percent": 100,
        "message": (
            f"계약서 검토가 끝났습니다.\n"
            f"- 전체 단위: {total}개\n"
            f"- 검토 성공: {reviewed_count}개\n"
            f"- 수정이 필요한 항목: {issue_count}개\n"
            f"- 근거 부족: {no_evidence_count}개\n"
            f"- 검토 실패(검색 오류): {search_error_count}개\n"
            f"- 짧은 조각 생략: {skipped_count}개"
        ),
        "total": total,
        "issue_count": issue_count,
        "reviewed_count": reviewed_count,
        "no_evidence_count": no_evidence_count,
        "search_error_count": search_error_count,
        "skipped_count": skipped_count,
    }
