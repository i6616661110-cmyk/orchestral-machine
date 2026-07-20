
import unittest
import os
import sys
import json

# Ensure src is in pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph import hard_reset_node, hir_halt_node, verifier_reset_node

class TestGraphSystemNodes(unittest.TestCase):
    def setUp(self):
        self.state = {
            "task_id": "test_task_123",
            "task": "do something",
            "loop_iteration": 0,
            "audit_log": [],
            "event_log": []
        }

    def test_hard_reset_node(self):
        result = hard_reset_node(self.state)
        self.assertEqual(result["loop_iteration"], 1)
        self.assertEqual(result["status"], "rewrite_confirmed")
        
        # Verify AuditEntry via helper
        audit_log = result["audit_log"]
        self.assertEqual(len(audit_log), 1)
        entry = audit_log[0]
        self.assertEqual(entry["node"], "GRAPH_CONTROLLER")
        self.assertEqual(entry["model_id"], "system")
        self.assertEqual(entry["status"], "hard_reset_executed")
        self.assertIn("Atomic hard reset", entry["prompt_summary"])
        self.assertTrue(len(entry["seed"]) == 16)

    def test_hir_halt_node(self):
        result = hir_halt_node(self.state)
        self.assertEqual(result["status"], "HIR")
        self.assertTrue(result["system_flags"]["halted"])
        
        # Verify AuditEntry via helper
        audit_log = result["audit_log"]
        self.assertEqual(len(audit_log), 1)
        entry = audit_log[0]
        self.assertEqual(entry["node"], "GRAPH_CONTROLLER")
        self.assertEqual(entry["status"], "HIR")
        self.assertIn("System halted", entry["prompt_summary"])

    def test_verifier_reset_node(self):
        result = verifier_reset_node(self.state)
        self.assertEqual(result["correction_attempt"], 0)
        self.assertTrue(result["verifier_reset_used"])
        
        # Verify AuditEntry via helper
        audit_log = result["audit_log"]
        self.assertEqual(len(audit_log), 1)
        entry = audit_log[0]
        self.assertEqual(entry["node"], "GRAPH_CONTROLLER")
        self.assertEqual(entry["status"], "verifier_reset_applied")

if __name__ == "__main__":
    unittest.main()
