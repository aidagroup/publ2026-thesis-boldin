.PHONY: dev server lint test lock bundle

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

# Source snapshot for hand-upload to the lab JupyterHub, which has no GitHub
# access (docs/jupyterhub-runbook.md). Includes vendor/ — gitignored, so
# `git archive` alone would drop it — giving a single self-contained upload.
#   make bundle            # current branch
#   make bundle REF=main   # any ref
REF ?= HEAD
bundle:
	@rm -rf .bundle-stage && mkdir -p .bundle-stage
	@git archive --format=tar --prefix=callosum/ $(REF) | tar x -C .bundle-stage
	@if [ -d vendor ]; then cp -R vendor .bundle-stage/callosum/vendor; \
		echo "   included vendor/ ($$(du -sh vendor | cut -f1))"; \
	else echo "   note: no vendor/ — PhysX must be staged separately on the server"; fi
	@tar czf callosum.tar.gz -C .bundle-stage callosum
	@rm -rf .bundle-stage
	@echo "   callosum.tar.gz  $$(du -h callosum.tar.gz | cut -f1)  (ref: $(REF))"
