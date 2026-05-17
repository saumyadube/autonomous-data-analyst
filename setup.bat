@echo off
echo ============================================
echo   LLM-Powered Autonomous Data Analyst
echo   Setup Script for Windows
echo   Saumaya Dube - Rama University
echo ============================================
echo.

REM Check Python
python --version 2>nul
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please download Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo.
echo [2/4] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [3/4] Installing all required packages...
pip install -r requirements.txt

echo.
echo [4/4] Setup complete!
echo.
echo ============================================
echo   HOW TO RUN THE APP:
echo   1. Open .env file and add your Groq API key
echo      Get FREE key at: https://console.groq.com
echo   2. Run:  streamlit run app.py
echo   3. Open: http://localhost:8501
echo ============================================
echo.
pause
