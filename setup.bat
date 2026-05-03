@echo off
:: setup.bat — TradeVision one-shot environment setup for Windows (Command Prompt)
::
:: Creates a project-local .venv\ and installs all dependencies into it.
:: Never touches your global Python installation.
::
:: Usage:
::   setup.bat

setlocal EnableDelayedExpansion

set "VENV_DIR=.venv"
set "REQUIREMENTS=requirements.txt"

echo.
echo [setup] Checking Python version ...

:: Find python or python3
where python >nul 2>&1
if %errorlevel% == 0 (
    set "PYTHON_BIN=python"
) else (
    where python3 >nul 2>&1
    if %errorlevel% == 0 (
        set "PYTHON_BIN=python3"
    ) else (
        echo [ERROR] Python not found in PATH.
        echo         Install Python 3.10+ from https://www.python.org/downloads/
        echo         Make sure to check "Add Python to PATH" during installation.
        exit /b 1
    )
)

:: Check version >= 3.10
for /f "tokens=2" %%v in ('%PYTHON_BIN% --version 2^>^&1') do set "PY_VER=%%v"
echo [setup] Found Python %PY_VER%

:: ── Create virtual environment ─────────────────────────────────────────────────
if exist "%VENV_DIR%\Scripts\python.exe" (
    echo [setup] Virtual environment already exists at %VENV_DIR%\ -- skipping creation.
    echo         To rebuild: rmdir /s /q %VENV_DIR% ^& setup.bat
) else (
    echo [setup] Creating virtual environment at .\%VENV_DIR%\ ...
    %PYTHON_BIN% -m venv %VENV_DIR%
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
    echo [setup] Virtual environment created.
)

:: ── Upgrade pip ────────────────────────────────────────────────────────────────
echo [setup] Upgrading pip ...
%VENV_DIR%\Scripts\python.exe -m pip install --upgrade pip --quiet

:: ── Install dependencies ───────────────────────────────────────────────────────
if not exist "%REQUIREMENTS%" (
    echo [ERROR] %REQUIREMENTS% not found. Are you in the project root?
    exit /b 1
)

echo [setup] Installing dependencies from %REQUIREMENTS% ...
%VENV_DIR%\Scripts\pip.exe install -r %REQUIREMENTS%
if %errorlevel% neq 0 (
    echo [ERROR] Dependency installation failed.
    exit /b 1
)
echo [setup] All dependencies installed.

:: ── Print next steps ───────────────────────────────────────────────────────────
echo.
echo ======================================================
echo    TradeVision setup complete!
echo ======================================================
echo.
echo   Activate the venv before each session:
echo     .venv\Scripts\activate
echo.
echo   Next steps:
echo     1. streamlit run app.py
echo     2. Optional: python -m tradevision.model.trainer
echo     3. Optional rebuild workflow: see README.md
echo.
echo   Or use the Makefile (requires make for Windows):
echo     make run ^| make train ^| make test ^| make fetch-data
echo.

endlocal
