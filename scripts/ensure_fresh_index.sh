#!/bin/bash

# ensure_fresh_index.sh
# Checks if project index is fresh and updates if needed

set -e

INDEX_FILE="project_index.json"
MAX_AGE_MINUTES=30

cd "$(dirname "$0")/.."

if [ ! -f "$INDEX_FILE" ]; then
    echo "⚠️  Index not found, creating..."
    python3 -m src.tools.indexer
    exit 0
fi

# Check index age (compatible with both macOS and Linux)
if stat -f %m "$INDEX_FILE" &>/dev/null; then
    # macOS
    INDEX_AGE=$(($(date +%s) - $(stat -f %m "$INDEX_FILE")))
else
    # Linux
    INDEX_AGE=$(($(date +%s) - $(stat -c %Y "$INDEX_FILE")))
fi

AGE_MINUTES=$((INDEX_AGE / 60))

if [ $AGE_MINUTES -gt $MAX_AGE_MINUTES ]; then
    echo "⚠️  Index is ${AGE_MINUTES} minutes old, updating..."
    python3 -m src.tools.indexer
else
    echo "✅ Index is fresh (${AGE_MINUTES} minutes old)"
fi
