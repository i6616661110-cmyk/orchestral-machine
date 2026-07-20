import os
import sys
import importlib
import pytest
from unittest.mock import patch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
# Adjust path to handle running from root or tests dir
if "src" not in sys.path[-1]:
    sys.path.append(os.path.abspath("src"))

def test_requirements_pinning():
    """Verify all lines in requirements.txt have pinned versions (==)."""
    req_path = os.path.join(os.path.dirname(__file__), "../requirements.txt")
    with open(req_path, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        assert "==" in line, f"Dependency not pinned: {line}"
    print("✅ requirements.txt pinning verified")

def test_telegram_token_validation():
    """Verify TELEGRAM_BOT_TOKEN validation logic."""
    import config
    
    # Case 1: Valid Token
    with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "123456:ABC-DEF_123"}):
        importlib.reload(config)
        assert config.TELEGRAM_BOT_TOKEN == "123456:ABC-DEF_123"
    print("✅ Valid TELEGRAM_BOT_TOKEN passed")
        
    # Case 2: Invalid Token Format
    with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "invalid_token_format"}):
        try:
            importlib.reload(config)
            assert False, "Should have raised ValueError for invalid token"
        except ValueError as e:
            assert "Invalid TELEGRAM_BOT_TOKEN format" in str(e)
    print("✅ Invalid TELEGRAM_BOT_TOKEN rejected")

    # Case 3: No Token (Should ensure None and no error as it's optional in type hint, 
    # but potentially checked if logical usage requires it. 
    # The _validate function checks 'if TELEGRAM_BOT_TOKEN:' so None is fine.)
    with patch.dict(os.environ):
        if "TELEGRAM_BOT_TOKEN" in os.environ:
             del os.environ["TELEGRAM_BOT_TOKEN"]
        importlib.reload(config)
        assert config.TELEGRAM_BOT_TOKEN is None
    print("✅ Missing TELEGRAM_BOT_TOKEN handled (None)")

def test_allowed_users_parsing():
    """Verify ALLOWED_TELEGRAM_USERS parsing from env var."""
    import config
    
    # Case 1: Valid List
    with patch.dict(os.environ, {"ALLOWED_TELEGRAM_USERS": "111, 222, 333"}):
        importlib.reload(config)
        assert config.ALLOWED_TELEGRAM_USERS == [111, 222, 333]
    print("✅ Valid ALLOWED_TELEGRAM_USERS parsed")
    
    # Case 2: Empty
    with patch.dict(os.environ, {"ALLOWED_TELEGRAM_USERS": ""}):
        importlib.reload(config)
        assert config.ALLOWED_TELEGRAM_USERS == []
    print("✅ Empty ALLOWED_TELEGRAM_USERS handled")

    # Case 3: Missing
    with patch.dict(os.environ):
        if "ALLOWED_TELEGRAM_USERS" in os.environ:
            del os.environ["ALLOWED_TELEGRAM_USERS"]
        importlib.reload(config)
        assert config.ALLOWED_TELEGRAM_USERS == []
    print("✅ Missing ALLOWED_TELEGRAM_USERS handled")
    
    # Case 4: Garbage/Invalid handling (Fail safe)
    with patch.dict(os.environ, {"ALLOWED_TELEGRAM_USERS": "123, abc, 456"}):
        importlib.reload(config)
        # The logic: 
        # try: ... except ValueError: pass (result is empty list)
        assert config.ALLOWED_TELEGRAM_USERS == [] 
        # Alternatively if the code was more robust to skip invalid items, it might differ.
        # But the current implementation wraps the list comp in a single try/except block.
        # So one error fails the whole list to [].
    print("✅ Invalid ALLOWED_TELEGRAM_USERS fail-safe verified")

if __name__ == "__main__":
    try:
        test_requirements_pinning()
        test_telegram_token_validation()
        test_allowed_users_parsing()
        print("\n🎉 ALL SECURITY CONFIG TESTS PASSED!")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
