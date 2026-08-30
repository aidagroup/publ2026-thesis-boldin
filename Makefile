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
# access (docs/jupyterhub-runbook.md).
#   make bundle              # sources only (~200 KB) — the normal case
#   make bundle VENDOR=1     # + vendor/ (~80 MB), needed only when the server's
#                            #   scratch was wiped and PhysX must be re-staged
#   make bundle REF=main     # any ref
#
# vendor/ is gitignored, so `git archive` drops it; it is copied in explicitly
# and only on request — PhysX never changes, so re-uploading it with every code
# edit is pure waste.
REF ?= HEAD
bundle:
	@rm -rf .bundle-stage && mkdir -p .bundle-stage
	@git archive --format=tar --prefix=callosum/ $(REF) | tar x -C .bundle-stage
ifdef VENDOR
	@if [ -d vendor ]; then cp -R vendor .bundle-stage/callosum/vendor; \
		echo "   included vendor/ ($$(du -sh vendor | cut -f1))"; \
	else echo "   VENDOR=1 requested but vendor/ does not exist"; exit 1; fi
else
	@echo "   sources only (add VENDOR=1 if the server needs PhysX re-staged)"
endif
	@tar czf callosum.tar.gz -C .bundle-stage callosum
	@rm -rf .bundle-stage
	@echo "   callosum.tar.gz  $$(du -h callosum.tar.gz | cut -f1)  (ref: $(REF))"
