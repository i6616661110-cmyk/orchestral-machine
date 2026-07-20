import sys
import os
import asyncio
from dotenv import load_dotenv

# Load env vars first
load_dotenv()

# Fix path to include src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.execution_engine import run_task_generator

def main():
    task_text = "Calculate the 10th Fibonacci number and explain the math."
    task_id = "test-system-run-001"
    
    print(f"🚀 Starting Manual System Test")
    print(f"Task: {task_text}")
    print(f"ID: {task_id}")
    print("-" * 50)

    try:
        # Run the generator
        for event in run_task_generator(task_text, task_id):
            e_type = event.get("type")
            
            if e_type == "STATE":
                phase = event.get("phase")
                role = event.get("role")
                payload = event.get("payload", {})
                activity = payload.get("activity", {}).get("message", "")
                
                print(f"[{phase}] {role}: {activity[:100]}...")
                
            elif e_type == "RESULT":
                print("-" * 50)
                print("✅ RESULT RECEIVED")
                print(event.get("payload"))
                
            elif e_type == "ERROR":
                print("-" * 50)
                print(f"❌ ERROR: {event.get('error')}")
                
    except Exception as e:
        print(f"🔥 FATAL EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
