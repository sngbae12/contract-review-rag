"""
계약서 검토 에이전트 (LangGraph)
흐름: 계약서 조각 → RAG 유사 문서 검색 → LLM 위배/보완 판단 → 수정 문구 생성
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, TypedDict

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from tqdm import tqdm

from config import (
    CONTRACT_CHUNK_OVERLAP,
    CONTRACT_CHUNK_SIZE,
    CONTRACT_SEPARATORS,
    LLM_MODEL,
    MIN_REVIEW_LENGTH,
)
from services.openai_runtime import require_api_key, user_facing_openai_error
from services.rag_service import get_retriever, rag_document_count


class ReviewDecision(BaseModel):
    """LLM이 반드시 이 형식으로만 답하도록 고정한다."""

    has_issue: bool = Field(description="가이드라인/약관 위배 또는 보완이 필요하면 true")
    revised_sentence: str = Field(description="수정이 필요하면 고친 문장, 없으면 원문과 동일")
    reason: str = Field(description="판단 이유를 한두 문장으로")


class ReviewState(TypedDict):
    """LangGraph가 노드 사이에 주고받는 상태."""

    sentence: str
    context: str
    has_issue: bool
    revised_sentence: str
    reason: str


def _llm() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=require_api_key())


def split_contract_pdf(pdf_path: Path) -> Generator[dict, None, None]:
    """
    계약서 PDF를 30글자로 분할한다. 오버랩은 없다.
    성공하기 전에는 기존 계약서 상태를 덮어쓰지 않는다.
    """
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
                "message": f"{pdf_path.name} 에서 텍스트를 읽지 못했습니다. 파일이 손상되었거나 PDF가 아닐 수 있습니다.",
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
            "message": f"{pdf_path.name} 추출 완료 (페이지 {len(documents)}). 문장을 분할합니다...",
        }

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONTRACT_CHUNK_SIZE,
            chunk_overlap=CONTRACT_CHUNK_OVERLAP,
            separators=CONTRACT_SEPARATORS,
            length_function=len,
        )
        chunks = splitter.split_documents(documents)
        sentences = [c.page_content.strip() for c in chunks if c.page_content and c.page_content.strip()]

        if not sentences:
            yield {
                "type": "error",
                "stage": "error",
                "message": "계약서에서 검토할 문장을 찾지 못했습니다.",
            }
            return

        yield {
            "type": "done",
            "stage": "ready",
            "percent": 100,
            "message": (
                f"계약서 업로드가 완료되었습니다.\n"
                f"- 파일: {pdf_path.name}\n"
                f"- 페이지: {len(documents)}쪽\n"
                f"- 검토 단위: {len(sentences)}개\n"
                f"왼쪽의 [계약서 검토] 버튼을 누르면 문장별 검토를 시작합니다."
            ),
            "filename": pdf_path.name,
            "page_count": len(documents),
            "sentence_count": len(sentences),
            "sentences": sentences,
        }
    except Exception:
        yield {
            "type": "error",
            "stage": "error",
            "message": "계약서 처리 중 오류가 발생했습니다. 다시 시도해 주세요.",
        }


def _retrieve_node(state: ReviewState) -> dict:
    """노드 1: 현재 계약서 문장과 비슷한 가이드라인/약관을 찾는다."""
    query = state["sentence"]
    try:
        docs = get_retriever().invoke(query)
    except Exception:
        return {"context": "(유사 문서를 찾지 못했습니다.)"}

    if not docs:
        return {"context": "(관련된 가이드라인/약관 조각을 찾지 못했습니다.)"}

    lines = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_file", "unknown")
        lines.append(f"[{i}] ({source}) {doc.page_content.strip()}")
    return {"context": "\n".join(lines)}


def _analyze_node(state: ReviewState) -> dict:
    """노드 2: RAG 문서를 보고 위배/보완 여부를 판단하고 필요하면 수정한다."""
    sentence = state["sentence"]
    context = state.get("context") or ""

    # 짧은 조각은 LLM 호출 없이 원문만 통과
    if len(sentence) < MIN_REVIEW_LENGTH:
        return {
            "has_issue": False,
            "revised_sentence": sentence,
            "reason": "검토 대상이 아닌 짧은 조각입니다.",
        }

    structured_llm = _llm().with_structured_output(ReviewDecision)
    prompt = f"""당신은 한국어 계약서 검토 전문가입니다.
아래 계약서 문장을 가이드라인/약관과 비교해 위배하거나 보완이 필요한지 판단하세요.

[가이드라인/약관에서 검색된 관련 내용]
{context}

[계약서 문장]
{sentence}

판단 규칙:
- 가이드라인·약관에 어긋나거나, 한쪽에게 부당하게 불리하거나, 의미가 뒤집힌 조항이면 has_issue=true
- 수정할 때는 원문의 번호·문체·형식은 유지하고, 문제가 되는 표현만 고친다
- 이상이 없으면 has_issue=false 이고 revised_sentence는 원문과 동일하게 둔다
- 관련 가이드라인이 거의 없어도, 문장 자체에 명백한 오류(부정/긍정 뒤바뀜 등)가 있으면 수정한다
"""
    decision = structured_llm.invoke(prompt)
    revised = decision.revised_sentence.strip() or sentence
    # 모델이 원문과 똑같이 돌려주면 이슈가 없는 것으로 본다.
    has_issue = bool(decision.has_issue) and revised != sentence
    return {
        "has_issue": has_issue,
        "revised_sentence": revised,
        "reason": decision.reason,
    }


def build_review_graph():
    """검색 → 분석 순서로 이어지는 LangGraph 에이전트를 컴파일한다."""
    graph = StateGraph(ReviewState)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("analyze", _analyze_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "analyze")
    graph.add_edge("analyze", END)
    return graph.compile()


# 앱 기동 시 한 번만 컴파일해 재사용
REVIEW_GRAPH = build_review_graph()


def review_sentences(sentences: list[str]) -> Generator[dict, None, None]:
    """
    계약서 조각을 하나씩 검토하고, 끝날 때마다 이벤트를 보낸다.
    한꺼번에 결과를 모아서 보내지 않는다. (화면에서 진행 여부를 확인하기 위함)
    """
    if rag_document_count() == 0:
        yield {
            "type": "error",
            "message": "RAG 문서가 없습니다. 먼저 왼쪽에서 가이드라인/약관 PDF를 업로드하세요.",
        }
        return

    if not sentences:
        yield {"type": "error", "message": "검토할 계약서 문장이 없습니다. 계약서를 다시 업로드하세요."}
        return

    total = len(sentences)
    yield {
        "type": "progress",
        "percent": 0,
        "message": f"계약서 검토를 시작합니다. 총 {total}개 문장입니다.",
        "current": 0,
        "total": total,
    }

    issue_count = 0
    try:
        with tqdm(total=total, desc="계약서 문장 검토", ncols=80) as pbar:
            for index, sentence in enumerate(sentences, start=1):
                result = REVIEW_GRAPH.invoke(
                    {
                        "sentence": sentence,
                        "context": "",
                        "has_issue": False,
                        "revised_sentence": sentence,
                        "reason": "",
                    }
                )
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
                    "original": sentence,
                    "revised": result.get("revised_sentence") or sentence,
                    "has_issue": bool(result.get("has_issue")),
                    "reason": result.get("reason") or "",
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
            f"계약서 검토가 완료되었습니다.\n"
            f"- 검토 문장: {total}개\n"
            f"- 수정이 필요한 문장: {issue_count}개"
        ),
        "total": total,
        "issue_count": issue_count,
    }
