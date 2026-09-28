@echo off
REM Start the NanoGym dashboard server on http://127.0.0.1:7669
cd /d "%~dp0"

REM Prefer the project's virtual environment if present, else fall back to global py
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=py"
)

echo Starting NanoGym dashboard on http://127.0.0.1:7669
"%PY%" scripts\serve.py --allow-control

REM Keep the window open if the server exits or crashes so errors are visible
echo.
echo Server stopped.
pause
