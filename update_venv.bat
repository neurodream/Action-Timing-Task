@echo off
REM update existing .venv to match requirements.txt
REM run from repo root

if not exist ".venv" (
    echo ".venv not found – run install_venv.bat first"
    exit /b 1
)

call .venv\Scripts\activate

python -m pip install --upgrade pip
pip install --upgrade -r requirements.txt

echo ---
echo "Done. Out‑of‑date packages:"
pip list --outdated

deactivate

