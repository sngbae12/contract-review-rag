@echo off
chcp 65001 >nul
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)

"%PY%" -c "import sys" >nul 2>&1
if errorlevel 1 (
  echo Python을 찾지 못했습니다. Python 3.11 이상을 설치한 뒤 다시 실행하세요.
  pause
  exit /b 1
)

"%PY%" -c "import flask, langchain, langchain_openai, langchain_chroma, langgraph, pypdf" >nul 2>&1
if errorlevel 1 (
  echo 필요한 Python 패키지가 없습니다.
  echo.
  echo 다음을 실행하세요:
  echo   python -m venv .venv
  echo   .venv\Scripts\python -m pip install -r requirements.txt
  echo.
  pause
  exit /b 1
)

echo OpenAI API 키는 브라우저 왼쪽 입력란에 넣습니다. 파일에는 저장되지 않습니다.
echo 브라우저에서 http://127.0.0.1:8765 로 접속하세요.
echo.
"%PY%" app.py
if errorlevel 1 (
  echo.
  echo 서버가 오류로 종료되었습니다. 위 메시지를 확인하세요.
)
echo.
pause
