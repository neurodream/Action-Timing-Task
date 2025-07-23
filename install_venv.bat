@echo off
REM create venv next to repo if missing
if not exist ".venv" (
    python -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo done
