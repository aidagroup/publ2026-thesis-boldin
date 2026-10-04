.PHONY: dev server lint test lock hooks

# Local authoring env (macOS): RL stack + tooling, NO GPU sim (ManiSkill is Linux-only).
dev:
	uv sync --extra train --extra dev

# Full training env (Linux + CUDA). For first-time server setup prefer: bash scripts/setup_server.sh
server:
	uv sync --extra sim --extra train --extra dev

lint:
	uv run ruff check .

test:
	uv run pytest -q

lock:
	uv lock

# Enable the versioned git hooks (.githooks/commit-msg rewrites subjects to `<branch> (<type>): <msg>`).
hooks:
	git config core.hooksPath .githooks
