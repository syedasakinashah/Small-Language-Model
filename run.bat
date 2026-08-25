@echo off
REM Start Miss RUBI. Double-click this file, or run it from a terminal.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)

echo Starting Miss RUBI at http://localhost:8501
".venv\Scripts\python.exe" -m streamlit run app/app.py
pause
