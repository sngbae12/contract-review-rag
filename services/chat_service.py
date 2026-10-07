"""
일반 질문 처리
- RAG를 쓰지 않고 gpt-4o-mini에 바로 요청한다.
"""

from __future__ import annotations

from collections.abc import Generator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import LLM_MODEL
from services.openai_runtime import require_api_key

SYSTEM_PROMPT = (
    "당신은 계약서와 법률 실무를 돕는 한국어 비서입니다. "
    "사용자의 일반 질문에 간결하고 정확하게 답하세요. "
    "법적 조언이 필요할 때는 일반 정보임을 밝히고, 최종 판단은 전문가와 상의하라고 안내하세요."
)


def stream_answer(question: str) -> Generator[str, None, None]:
    """토큰 단위로 답을 흘려보내 채팅창에 바로 찍히게 한다."""
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3, api_key=require_api_key())
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]
    for chunk in llm.stream(messages):
        text = chunk.content
        if text:
            yield text
