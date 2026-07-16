#!/usr/bin/env bash
# Reindex all documents after switching embedding provider/model.
#
# Usage:
#   bash scripts/reindex.sh
#   bash scripts/reindex.sh --store ./other_rag_data
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

exec uv run python3 -m src.cli reindex "$@"
