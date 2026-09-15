@echo off
setlocal
cd /d "%~dp0"
set "PYDIR=%~dp0.venv\Scripts"

if not exist "%PYDIR%\python.exe" (
    echo.
    echo [ERROR] Virtual environment not found. Create it with:
    echo   py -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

rem --- dependency sanity check so failures are visible instead of silent ---
"%PYDIR%\python.exe" -c "import PyQt6, faster_whisper, soundcard, openai, numpy, yaml" 2>nul
if errorlevel 1 (
    echo.
    echo [ERROR] Dependencies missing. Install them with:
    echo   "%PYDIR%\python.exe" -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

rem --- launch the overlay without a console window ---
start "" "%PYDIR%\pythonw.exe" run.py
exit /b 0
