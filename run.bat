@echo off
title Movie Recommendation System
cd /d "%~dp0"

echo ========================================================
echo Starting Movie Recommendation System...
echo ========================================================

REM Check if .venv exists
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found in "%~dp0.venv"!
    echo Please check your setup.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment (.venv)...
call .venv\Scripts\activate.bat

REM Run application
echo Starting run.py...
python run.py

REM Keep terminal open if application exits
echo.
echo Application stopped.
pause
