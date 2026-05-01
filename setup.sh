#!/usr/bin/env bash
# setup.sh — One-shot TradeVision environment setup
#
# Creates a project-local .venv/ and installs all dependencies into it.
# Never touches your global Python installation.
#
# Usage:
#   bash setup.sh

set -euo pipefail

VENV_DIR=".venv"
REQUIREMENTS="requirements.txt"
ENV_EXAMPLE=".env.example"
ENV_FILE=".env"

# ── Colour helpers ─────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
RESET='\033[0m'

info()    { echo -e "${CYAN}[setup]${RESET} $*"; }
success() { echo -e "${GREEN}[setup]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[setup]${RESET} $*"; }
error()   { echo -e "${RED}[setup]${RESET} $*" >&2; exit 1; }

# ── Check Python ───────────────────────────────────────────────────────────────
PYTHON_BIN=""
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then
        version=$("$candidate" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [[ "$major" -ge 3 && "$minor" -ge 10 ]]; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [[ -z "$PYTHON_BIN" ]]; then
    error "Python 3.10+ is required but was not found in PATH.\nInstall it from https://www.python.org/downloads/"
fi

info "Using Python: $($PYTHON_BIN --version)"

# ── Create virtual environment ─────────────────────────────────────────────────
if [[ -d "$VENV_DIR" ]]; then
    warn "Virtual environment already exists at $VENV_DIR/ — skipping creation."
    warn "To rebuild from scratch, run:  rm -rf $VENV_DIR && bash setup.sh"
else
    info "Creating virtual environment at ./$VENV_DIR/ …"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    success "Virtual environment created."
fi

# ── Activate ───────────────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
info "Activated: $(which python) ($(python --version))"

# ── Upgrade pip ────────────────────────────────────────────────────────────────
info "Upgrading pip …"
pip install --upgrade pip --quiet

# ── Install dependencies ───────────────────────────────────────────────────────
if [[ ! -f "$REQUIREMENTS" ]]; then
    error "$REQUIREMENTS not found. Are you in the project root?"
fi

info "Installing dependencies from $REQUIREMENTS …"
pip install -r "$REQUIREMENTS"
success "All dependencies installed."

# ── Copy .env if missing ───────────────────────────────────────────────────────
if [[ ! -f "$ENV_FILE" ]]; then
    if [[ -f "$ENV_EXAMPLE" ]]; then
        cp "$ENV_EXAMPLE" "$ENV_FILE"
        warn ".env created from .env.example — please fill in your Kaggle credentials."
    fi
else
    info ".env already exists — skipping copy."
fi

# ── Print next steps ───────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${RESET}"
echo -e "${GREEN}║          TradeVision setup complete! ✅              ║${RESET}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  ${CYAN}Activate the venv before each session:${RESET}"
echo -e "    macOS / Linux:  source .venv/bin/activate"
echo -e "    Windows CMD:    .venv\\Scripts\\activate"
echo -e "    Windows PS:     .venv\\Scripts\\Activate.ps1"
echo ""
echo -e "  ${CYAN}Next steps:${RESET}"
echo -e "    1. Edit .env with your Kaggle credentials"
echo -e "    2. Set KAGGLE_DATASET_SLUG in config.py"
echo -e "    3. python data/download_dataset.py"
echo -e "    4. python -m tradevision.model.trainer"
echo -e "    5. streamlit run app.py"
echo ""
echo -e "  ${CYAN}Or use the Makefile shortcuts (macOS/Linux):${RESET}"
echo -e "    make download | make train | make run | make test"
echo ""
