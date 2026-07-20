#!/bin/bash

# start_session.sh
# Run this script at the start of any AI agent session
# It ensures the project index is fresh before work begins

set -e

echo "=== Session Start: Ensuring Fresh Index ==="

cd "$(dirname "$0")/.."

INDEX_FILE="project_index.json"
MAX_AGE_MINUTES=30

if [ ! -f "$INDEX_FILE" ]; then
    echo "⚠️  Index not found, creating..."
    python3 -m src.tools.indexer
else
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
fi

echo ""
echo "=== Ready to work ==="
echo "📖 Read project_index.json to understand the project structure"
echo "   - 30 Python files"
echo "   - 63 classes, 168 functions"
echo "   - Dependencies and symbol search included"
