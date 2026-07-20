import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
# Adjust path to handle running from root or tests dir
if "src" not in sys.path[-1]:
    sys.path.append(os.path.abspath("src"))

from fastapi import FastAPI
from fastapi.testclient import TestClient

# Import the router
from src.api.control_endpoints import router

class TestAPISecurity(unittest.TestCase):
    
    def setUp(self):
        # Create a fresh app for testing
        self.app = FastAPI()
        self.app.include_router(router)
        self.client = TestClient(self.app)
        
    @patch("src.api.control_endpoints.operator_halt")
    @patch("src.api.control_endpoints._get_current_state")
    def test_control_endpoint_security(self, mock_get_state, mock_halt):
        """Test security on /api/control."""
        
        mock_get_state.return_value = {}
        mock_halt.return_value = {"status": "HALTED"}
        
        # Test 1: Missing API Key -> 422 (FastAPI default for missing header)
        print("\nTesting Missing API Key...")
        response = self.client.post("/api/control", json={"command": "HALT"})
        # FastAPI returns 422 for missing required header/query params by default
        self.assertEqual(response.status_code, 422) 
        print("✅ Missing Key rejected (422)")

        # Test 2: Invalid API Key -> 401
        print("Testing Invalid API Key...")
        with patch.dict(os.environ, {"CONTROL_API_KEY": "secret-key"}):
            response = self.client.post(
                "/api/control", 
                json={"command": "HALT"},
                headers={"X-API-Key": "wrong-key"}
            )
            self.assertEqual(response.status_code, 401)
            self.assertIn("Invalid API Key", response.json()["detail"])
        print("✅ Invalid Key rejected (401)")

        # Test 3: Valid API Key -> 200
        print("Testing Valid API Key...")
        with patch.dict(os.environ, {"CONTROL_API_KEY": "secret-key"}):
            response = self.client.post(
                "/api/control", 
                json={"command": "HALT"},
                headers={"X-API-Key": "secret-key"}
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "HALTED"})
        print("✅ Valid Key accepted (200)")

    def test_public_endpoints(self):
        """Test that public endpoints are still accessible without key."""
        print("Testing Public Endpoints...")
        
        # /api/status should be public
        with patch("src.api.control_endpoints.get_system_status") as mock_status, \
             patch("src.api.control_endpoints._get_current_state") as mock_get_state:
            mock_status.return_value = {"status": "OK"}
            mock_get_state.return_value = {}
            
            response = self.client.get("/api/status")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "OK"})
            print("✅ /status is public")

if __name__ == "__main__":
    unittest.main()
