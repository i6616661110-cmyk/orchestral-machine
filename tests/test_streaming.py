#!/usr/bin/env python3
"""Verification script for streaming execution engine.

This script tests the streaming functionality of the execution engine,
demonstrating real-time state updates during graph execution.

Usage:
    python test_streaming.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from src.execution_engine import run_task_generator


def main():
    """Run the streaming verification test."""
    # Load environment
    load_dotenv()
    
    task = "Calculate 5th Fibonacci"
    task_id = "test_stream_001"
    
    print("=" * 60)
    print("STREAMING EXECUTION TEST")
    print("=" * 60)
    print(f"Task: {task}")
    print(f"Task ID: {task_id}")
    print("-" * 60)
    
    state_count = 0
    
    for event in run_task_generator(task, task_id):
        event_type = event.get("type")
        
        if event_type == "STATE":
            state_count += 1
            phase = event.get("phase", "unknown")
            role = event.get("role", "unknown")
            
            # Get progress info if available
            payload = event.get("payload", {})
            dynamics = payload.get("activity", {}).get("dynamics", {})
            progress = dynamics.get("progress", 0.0)
            
            print(f"[STATE #{state_count}] Phase: {phase.upper():12} | "
                  f"Role: {role or 'N/A':10} | Progress: {progress:.1%}")
            
        elif event_type == "RESULT":
            status = event.get("payload", {}).get("status", "unknown")
            print("-" * 60)
            print(f"[RESULT] Status: {status}")
            print(f"Total state updates: {state_count}")
            
        elif event_type == "ERROR":
            print("-" * 60)
            print(f"[ERROR] {event.get('error')}")
            
    print("=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
