# GigaMemory — Developer Makefile
# Requires: uv (https://docs.astral.sh/uv/getting-started/installation/)
#
# Usage:
#   make install       — install everything (CUDA + API + dev tools) — recommended for development
#   make install-local — install for local pipeline run only (CUDA, no API server deps)
#   make install-api   — install for API server only (CUDA + fastapi/uvicorn)
#   make install-cpu   — CPU-only environment (no GPU)
#   make lint          — run ruff linter
#   make format        — apply black formatting
#   make test          — run pytest in stub mode (no GPU/LLM needed)
#   make smoke         — quick pipeline smoke-test in stub mode
#   make vllm          — start vLLM inference server (slot model, Linux/WSL)
#   make serve         — start FastAPI REST server (port 8000)
#   make start         — start vLLM + FastAPI together (Linux/WSL)
#   make bot           — start the Telegram bot (needs API + TELEGRAM_BOT_TOKEN)
#   make docker-up      — start API only (bring your own vLLM)
#   make docker-up-vllm — start vLLM + API together via Docker
#   make docker-up-bot  — start API + Telegram bot together
#   make docker-up-all  — start vLLM + API + bot together (full demo)
#   make docker-up-cpu  — CPU-only API (no GPU, no vLLM)
#   make docker-down    — stop and remove containers

UV              := uv
PYTHON_VERSION  ?= 3.11
CUDA_VERSION    ?= cu126
VLLM_MODEL      ?= /mnt/d/Users/IvanK/Desktop/GigaMemory/models/Qwen3.5-4B-AWQ
VLLM_SERVED_NAME ?= models/Qwen3.5-4B-AWQ
VLLM_PORT       ?= 8001
VLLM_GPU_UTIL   ?= 0.65
VLLM_MAX_LEN    ?= 8192
VLLM_MAX_SEQS   ?= 4
API_PORT        ?= 8000

.PHONY: all install install-local install-cpu install-dev install-api install-bot \
        lint lint-fix format format-check type-check \
        test test-cov smoke \
        serve vllm start bot \
        docker-build docker-build-cpu docker-up docker-up-cpu docker-up-bot docker-up-all docker-up-vllm docker-down \
        clean help

all: help

# ─── Environment setup ────────────────────────────────────────────────────────

install: ## Install everything: CUDA + API + Telegram bot + dev tools (recommended for development)
	$(UV) sync --extra cuda --extra api --extra bot --extra dev
	@echo ""
	@echo "Full environment ready (CUDA + API + bot + dev)."
	@echo "Activate: source .venv/bin/activate  (Linux/Mac)"
	@echo "          .venv\\Scripts\\activate     (Windows)"

install-local: ## Install for local pipeline run only (CUDA, no API server, no dev tools)
	$(UV) sync --extra cuda
	@echo ""
	@echo "Local environment ready (CUDA only)."
	@echo "Activate: source .venv/bin/activate  (Linux/Mac)"
	@echo "          .venv\\Scripts\\activate     (Windows)"

install-api: ## Install for API server (CUDA + fastapi/uvicorn)
	$(UV) sync --extra cuda --extra api
	@echo ""
	@echo "API environment ready."
	@echo "Set OPENROUTER_API_KEY, then: make serve"

install-bot: ## Install Telegram bot deps only (no torch/GPU — thin API client)
	$(UV) sync --extra bot
	@echo ""
	@echo "Bot environment ready."
	@echo "Set TELEGRAM_BOT_TOKEN (and start the API), then: make bot"

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

serve: ## Start FastAPI REST server on port $(API_PORT) (requires vLLM running separately)
	$(UV) run uvicorn api:app \
		--app-dir DST_memory \
		--host 0.0.0.0 \
		--port $(API_PORT)

vllm: ## Start vLLM inference server (slot model, Linux/WSL only)
	@echo "Starting vLLM: model=$(VLLM_MODEL) port=$(VLLM_PORT) quant=compressed-tensors(auto)"
	vllm serve $(VLLM_MODEL) \
		--served-model-name $(VLLM_SERVED_NAME) \
		--port $(VLLM_PORT) \
		--dtype auto \
		--reasoning-parser qwen3 \
		--gpu-memory-utilization $(VLLM_GPU_UTIL) \
		--max-model-len $(VLLM_MAX_LEN) \
		--max-num-seqs $(VLLM_MAX_SEQS) \
		--limit-mm-per-prompt '{"image":0,"video":0}' \
		--trust-remote-code \
		--disable-log-stats

start: ## Start BOTH vLLM server and FastAPI in parallel (Linux/WSL only)
	@echo "Starting vLLM on port $(VLLM_PORT) and API on port $(API_PORT)..."
	@$(MAKE) vllm & \
	 sleep 30 && $(MAKE) serve

bot: ## Start the Telegram bot (requires API running + TELEGRAM_BOT_TOKEN)
	$(UV) run python -m telegram_bot

smoke: ## Quick pipeline smoke-test in stub mode (no GPU/LLM)
	$(UV) run python DST_memory/run.py \
		--llm-mode stub \
		--slot-use-stub \
		--memory-gate-use-stub \
		--no-final-llm \
		--memory-strategy full_graph_json \
		pipeline test \
		--dataset-path tests/fixtures/format_example.jsonl \
		--output-path DST_memory/smoke_output.json

# ─── Docker ───────────────────────────────────────────────────────────────────

docker-build: ## Build Docker image with CUDA support (default)
	docker build --build-arg TORCH_EXTRA=cuda -t gigamemory:latest .

docker-build-cpu: ## Build Docker image CPU-only (for CI/testing)
	docker build --build-arg TORCH_EXTRA=cpu -t gigamemory:cpu .

docker-up: ## Start API service only — vLLM managed externally (set SLOT_LLM_API_URL)
	docker compose up --build

docker-up-vllm: ## Start vLLM inference server + API together (requires NVIDIA GPU + MODEL_DIR)
	docker compose --profile vllm up --build

docker-up-cpu: ## Start API in CPU-only mode (no GPU, no vLLM; stub slot model)
	TORCH_EXTRA=cpu docker compose up --build

docker-up-bot: ## Start API + Telegram bot together (set TELEGRAM_BOT_TOKEN in .env)
	docker compose --profile bot up --build

docker-up-all: ## Start vLLM + API + Telegram bot together (full demo, needs GPU)
	docker compose --profile vllm --profile bot up --build

docker-down: ## Stop and remove all containers
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
