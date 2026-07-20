.PHONY: index fresh start help

help:
	@echo "Orchestral Machine - Index Management"
	@echo ""
	@echo "Commands:"
	@echo "  make index  - Force update project index"
	@echo "  make fresh  - Check and update index if older than 30 minutes"
	@echo "  make start  - Prepare session (run before starting work)"

index:
	python3 -m src.tools.indexer

fresh:
	./scripts/ensure_fresh_index.sh

start:
	./scripts/start_session.sh
