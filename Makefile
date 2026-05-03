# TradeVision — Makefile
#
# All commands run inside the project-local .venv/ automatically.
# You do NOT need to manually activate the venv to use these targets.
#
# Usage:
#   make setup      → Create .venv and install dependencies
#   make fetch-data → Optionally download dataset from Roboflow
#   make train      → Train the model
#   make run        → Launch the Streamlit app
#   make test       → Run pytest suite
#   make clean      → Remove .venv/, logs/, __pycache__

SHELL := /bin/bash
PYTHON := .venv/bin/python
PIP    := .venv/bin/pip
PYTEST := .venv/bin/pytest
STREAMLIT := .venv/bin/streamlit

.DEFAULT_GOAL := help

# ── Colours ────────────────────────────────────────────────────────────────────
CYAN  := \033[0;36m
GREEN := \033[0;32m
RESET := \033[0m

# ── Help ───────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@printf "\n"
	@printf "  $(CYAN)TradeVision — Available Commands$(RESET)\n"
	@printf "\n"
	@printf "  $(GREEN)make setup$(RESET)      Create .venv + install all dependencies\n"
	@printf "  $(GREEN)make fetch-data$(RESET) Optional: download dataset from Roboflow\n"
	@printf "  $(GREEN)make train$(RESET)      Train the EfficientNetB0 model\n"
	@printf "  $(GREEN)make run$(RESET)        Launch the Streamlit app\n"
	@printf "  $(GREEN)make test$(RESET)       Run pytest test suite\n"
	@printf "  $(GREEN)make clean$(RESET)      Remove .venv/, logs/, and __pycache__\n"
	@printf "\n"

# ── Setup / install ────────────────────────────────────────────────────────────
.PHONY: setup
setup:
	@echo -e "$(CYAN)[make]$(RESET) Running setup.sh …"
	@bash setup.sh

# Check that the venv exists before running any other target
.venv/bin/python:
	@echo "Virtual environment not found. Run:  make setup"
	@exit 1

# ── Optional dataset download ──────────────────────────────────────────────────
.PHONY: fetch-data
fetch-data: .venv/bin/python
	@echo -e "$(CYAN)[make]$(RESET) Downloading dataset from Roboflow …"
	$(PYTHON) data/download_dataset.py

# ── Training ───────────────────────────────────────────────────────────────────
.PHONY: train
train: .venv/bin/python
	@echo -e "$(CYAN)[make]$(RESET) Starting model training …"
	$(PYTHON) -m tradevision.model.trainer

# ── Run app ────────────────────────────────────────────────────────────────────
.PHONY: run
run: .venv/bin/python
	@echo -e "$(CYAN)[make]$(RESET) Launching Streamlit app …"
	$(STREAMLIT) run app.py

# ── Tests ──────────────────────────────────────────────────────────────────────
.PHONY: test
test: .venv/bin/python
	@echo -e "$(CYAN)[make]$(RESET) Running test suite …"
	$(PYTEST) tests/ -v

# ── Clean ──────────────────────────────────────────────────────────────────────
.PHONY: clean
clean:
	@echo -e "$(CYAN)[make]$(RESET) Cleaning project …"
	rm -rf .venv/
	rm -rf logs/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo -e "$(GREEN)[make]$(RESET) Clean complete."
