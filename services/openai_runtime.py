"""
요청 또는 해당 스트리밍이 처리되는 동안만 OpenAI API 키를 메모리에 둔다.
전역 객체·파일·환경변수·로그·URL·브라우저 저장소에 쓰지 않는다.
"""

from __future__ import annotations

from contextvars import ContextVar, Token

_API_KEY: ContextVar[str] = ContextVar("openai_api_key", default="")


class MissingApiKeyError(ValueError):
    """화면에서 API 키가 오지 않았을 때 사용한다."""


def bind_api_key(key: str) -> Token[str]:
    """현재 작업 스레드에만 키를 연결한다."""
    return _API_KEY.set((key or "").strip())


def reset_api_key(token: Token[str] | None) -> None:
    if token is None:
        return
    try:
        _API_KEY.reset(token)
    except RuntimeError:
        return


def current_api_key() -> str:
    return (_API_KEY.get() or "").strip()


def require_api_key() -> str:
    key = current_api_key()
    if not key:
        raise MissingApiKeyError(
            "OpenAI API 키가 없습니다. 화면 왼쪽에서 키를 입력한 뒤 다시 시도해 주세요."
        )
    return key


def user_facing_openai_error(exc: Exception) -> str:
    """예외 문구에 키가 섞일 수 있어 고정 안내만 돌린다."""
    text = f"{type(exc).__name__} {exc}".lower()
    if any(token in text for token in ("api key", "authentication", "unauthorized", "401", "invalid_api_key")):
        return "OpenAI API 키가 없거나 올바르지 않습니다. 키를 확인한 뒤 다시 시도해 주세요."
    if any(token in text for token in ("rate limit", "429", "quota")):
        return "OpenAI 요청 한도를 초과했습니다. 잠시 후 다시 시도해 주세요."
    return "OpenAI 호출에 실패했습니다. 키와 네트워크를 확인한 뒤 다시 시도해 주세요."
