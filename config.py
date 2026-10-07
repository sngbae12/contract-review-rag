"""
프로젝트 전역 설정
- OpenAI API 키는 화면 입력란에서 요청 단위로만 받는다. 이 파일에 키를 두지 않는다.
- 경로와 모델 이름은 코드에 적힌 값을 사용한다.
"""

from pathlib import Path

# 프로젝트 루트 (이 파일이 있는 폴더)
BASE_DIR = Path(__file__).resolve().parent

# 업로드 PDF와 Chroma DB를 둘 폴더
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
CHROMA_BACKUP_DIR = BASE_DIR / "data" / "chroma_db_backups"

# RAG(가이드라인/약관)용 업로드 폴더와 계약서 업로드 폴더를 분리
RAG_UPLOAD_DIR = UPLOAD_DIR / "rag"
CONTRACT_UPLOAD_DIR = UPLOAD_DIR / "contract"

# 가상 예시 문서
SAMPLE_DIR = BASE_DIR / "data" / "sample"

# LLM / 임베딩 모델 (OpenAI API)
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# RAG 문서 분할: 검색용. 문단·줄·문장 경계를 우선하고, 마지막에 글자 단위로 자른다.
# 이전 30글자 설정은 조항 의미가 끊기고, 구분자가 없으면 긴 줄이 그대로 남았다.
RAG_CHUNK_SIZE = 400
RAG_CHUNK_OVERLAP = 50
RAG_SEPARATORS = ["\n\n", "\n", ". ", ".", " ", ""]

# 계약서 분할: 제N조·문단을 검토 단위로 유지하고, 긴 조항만 추가 분할한다.
CONTRACT_CHUNK_SIZE = 800
CONTRACT_CHUNK_OVERLAP = 0
CONTRACT_SEPARATORS = ["\n\n", "\n", ". ", ".", " ", ""]

# 유사 문서 검색 개수
RETRIEVE_K = 5

# 너무 짧은 조각(쪽번호, 빈 줄 등)은 LLM 검토를 건너뜀
MIN_REVIEW_LENGTH = 8

# 업로드 용량 제한 (50MB)
MAX_CONTENT_LENGTH = 50 * 1024 * 1024

# Chroma 컬렉션 이름
CHROMA_COLLECTION = "contract_guidelines"


def ensure_directories() -> None:
    """실행 시 필요한 폴더를 미리 만들어 둔다."""
    RAG_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CONTRACT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
