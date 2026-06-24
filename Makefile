# GigaMemory — Developer Makefile
# Requires: uv (https://docs.astral.sh/uv/getting-started/installation/)
#
# Usage:
#   make install       — set up CUDA environment (default, cu126)
#   make install-cpu   — set up CPU-only environment
#   make lint          — run ruff linter
#   make format        — apply black formatting
#   make test          — run pytest in stub mode (no GPU/LLM needed)
#   make smoke         — quick pipeline smoke-test in stub mode
#   make docker-up     — start with docker compose (CUDA)
#   make docker-up-cpu — start with docker compose (CPU-only)

UV              := uv
PYTHON_VERSION  ?= 3.11
CUDA_VERSION    ?= cu126

.PHONY: all install install-cpu install-dev \
        lint lint-fix format format-check type-check \
        test test-cov smoke \
        docker-build docker-build-cpu docker-up docker-up-cpu docker-down \
        clean help

all: help

# ─── Environment setup ────────────────────────────────────────────────────────

install: ## Set up CUDA environment (cu126) with uv
	$(UV) sync --extra cuda --extra dev
	@echo ""
	@echo "CUDA environment ready."
	@echo "Activate: source .venv/bin/activate  (Linux/Mac)"
	@echo "          .venv\\Scripts\\activate     (Windows)"

install-cpu: ## Set up CPU-only environment with uv (Linux/macOS)
	$(UV) sync --extra dev
	$(UV) pip install torch torchvision torchaudio \
		--extra-index-url https://download.pytorch.org/whl/cpu
	@echo ""
	@echo "CPU environment ready."
	@echo "Activate: source .venv/bin/activate  (Linux/Mac)"
	@echo "          .venv\\Scripts\\activate     (Windows)"

install-dev: ## Install dev tools only (no torch — for CI lint/type jobs)
	$(UV) sync --extra dev
	@echo "Dev-only environment ready."

hooks: ## Install pre-commit hooks (run once after cloning)
	$(UV) run pre-commit install
	@echo "Pre-commit hooks installed."

# ─── Code quality ─────────────────────────────────────────────────────────────

lint: ## Run ruff linter (errors only)
	$(UV) run ruff check .

lint-fix: ## Run ruff linter and auto-fix
	$(UV) run ruff check . --fix

format: ## Apply black formatting
	$(UV) run black .

format-check: ## Check formatting without changes (used in CI)
	$(UV) run black --check .

type-check: ## Run mypy static type checker
	$(UV) run mypy DST_memory/dst_memory/ --ignore-missing-imports

# ─── Tests ────────────────────────────────────────────────────────────────────

test: ## Run pytest unit tests (stub mode — no GPU/LLM required)
	$(UV) run pytest tests/ -v

test-cov: ## Run tests with coverage report
	$(UV) run pytest tests/ -v \
		--cov=DST_memory/dst_memory \
		--cov-report=term-missing \
		--cov-report=html:htmlcov

smoke: ## Quick pipeline smoke-test in stub mode (no GPU/LLM)
	$(UV) run python DST_memory/run.py \
		--llm-mode stub \
		--slot-use-stub \
		--memory-gate-use-stub \
		--no-final-llm \
		--memory-strategy full_graph_json \
		pipeline test \
		--dataset-path data/format_example.jsonl \
		--output-path DST_memory/smoke_output.json

# ─── Docker ───────────────────────────────────────────────────────────────────

docker-build: ## Build Docker image with CUDA support (default)
	docker build --build-arg TORCH_EXTRA=cuda -t gigamemory:latest .

docker-build-cpu: ## Build Docker image CPU-only (for CI/testing)
	docker build --build-arg TORCH_EXTRA=cpu -t gigamemory:cpu .

docker-up: ## Start pipeline with docker compose (CUDA)
	docker compose up --build

docker-up-cpu: ## Start pipeline with docker compose (CPU-only)
	TORCH_EXTRA=cpu docker compose up --build

docker-down: ## Stop and remove containers
	docker compose down

# ─── Cleanup ──────────────────────────────────────────────────────────────────

clean: ## Remove build caches and test artifacts
	find . -type d -name __pycache__ -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f DST_memory/smoke_output.json DST_memory/smoke_*.json
	@echo "Clean done."

# ─── Help ─────────────────────────────────────────────────────────────────────

help: ## Show available targets
	@echo ""
	@echo "GigaMemory — available make targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
