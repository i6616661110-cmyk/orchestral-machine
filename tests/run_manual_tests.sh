#!/usr/bin/env bash

# Run all manual (heavy/paid) verification tests sequentially

echo "=========================================="
echo "Starting Manual/Paid Test Suite"
echo "=========================================="

echo "1. Testing LLM Connection..."
python3 tests/verify_llm_connection.py
if [ $? -ne 0 ]; then
    echo "❌ LLM Connection Test Failed"
    exit 1
fi

echo -e "\n2. Testing Streaming Execution..."
python3 tests/test_streaming.py
if [ $? -ne 0 ]; then
    echo "❌ Streaming Test Failed"
    exit 1
fi

echo -e "\n3. Testing Full Manual System Run..."
python3 tests/manual_test_run.py
if [ $? -ne 0 ]; then
    echo "❌ Full System Run Failed"
    exit 1
fi

echo -e "\n=========================================="
echo "✅ All Manual Tests Completed Successfully"
echo "=========================================="
