@echo off

if not exist ".venv" (
    python -m venv .venv || goto :error
)

call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt || goto :error

echo done
pause
exit /b 0

:error
echo install failed
pause
exit /b 1