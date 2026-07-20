"""Unit tests for src/enforcement.py — constitutional enforcement system."""

import pytest

from src.enforcement import (
    ConstitutionalViolation,
    emit_audit_for_violation,
    enforce_constitutional_rules,
    extract_role_from_node_name,
    validate_state_mutation,
)


# ───────────────────────────────────────────────────────────────────────────
# extract_role_from_node_name
# ───────────────────────────────────────────────────────────────────────────


class TestExtractRoleFromNodeName:
    def test_corrector_c1(self):
        assert extract_role_from_node_name("CORRECTOR_C1") == "corrector"

    def test_corrector_c2(self):
        assert extract_role_from_node_name("CORRECTOR_C2") == "corrector"

    def test_archivist_a1(self):
        assert extract_role_from_node_name("Archivist_A1") == "archivist"

    def test_archivist_a2(self):
        assert extract_role_from_node_name("ARCHIVIST_A2") == "archivist"

    def test_architect(self):
        assert extract_role_from_node_name("ARCHITECT") == "architect"

    def test_coder(self):
        assert extract_role_from_node_name("CODER") == "coder"


# ───────────────────────────────────────────────────────────────────────────
# validate_state_mutation
# ───────────────────────────────────────────────────────────────────────────


class TestValidateStateMutation:
    # ------------------------------------------------------------------
    # Check 0: Immutable task field
    # ------------------------------------------------------------------

    def test_architect_cannot_mutate_task(self):
        with pytest.raises(ConstitutionalViolation, match="immutable 'task' field"):
            validate_state_mutation({"task": "new task"}, "ARCHITECT")

    def test_coder_cannot_mutate_task(self):
        with pytest.raises(ConstitutionalViolation, match="immutable 'task' field"):
            validate_state_mutation({"task": "new task"}, "CODER")

    def test_verifier_cannot_mutate_task(self):
        with pytest.raises(ConstitutionalViolation, match="immutable 'task' field"):
            validate_state_mutation({"task": "override"}, "VERIFIER")

    # ------------------------------------------------------------------
    # Check 1: Protected approval keys
    # ------------------------------------------------------------------

    def test_verifier_can_set_approved_summary(self):
        assert validate_state_mutation({"approved_summary": True}, "VERIFIER") is True

    def test_validator_can_set_approved_meta(self):
        assert validate_state_mutation({"approved_meta": True}, "VALIDATOR") is True

    def test_architect_can_set_approved_hard_reset(self):
        assert (
            validate_state_mutation({"approved_hard_reset": True}, "ARCHITECT") is True
        )

    def test_coder_cannot_set_approved_summary(self):
        with pytest.raises(ConstitutionalViolation, match="protected keys"):
            validate_state_mutation({"approved_summary": True}, "CODER")

    def test_corrector_cannot_set_approved_meta(self):
        with pytest.raises(ConstitutionalViolation, match="protected keys"):
            validate_state_mutation({"approved_meta": True}, "CORRECTOR_C1")

    def test_tester_cannot_set_approved_hard_reset(self):
        with pytest.raises(ConstitutionalViolation, match="protected keys"):
            validate_state_mutation({"approved_hard_reset": True}, "TESTER")

    def test_archivist_cannot_set_approved_summary(self):
        with pytest.raises(ConstitutionalViolation, match="protected keys"):
            validate_state_mutation({"approved_summary": True}, "ARCHIVIST_A1")

    # ------------------------------------------------------------------
    # Check 2: Role-exclusive keys
    # ------------------------------------------------------------------

    def test_architect_can_set_plan(self):
        assert validate_state_mutation({"plan": {}}, "ARCHITECT") is True

    def test_coder_cannot_set_plan(self):
        with pytest.raises(
            ConstitutionalViolation, match="exclusive to role 'architect'"
        ):
            validate_state_mutation({"plan": {}}, "CODER")

    def test_corrector_can_set_applied_fixes(self):
        assert (
            validate_state_mutation(
                {"applied_fixes": [{"file": "test.py", "diff": "..."}]}, "CORRECTOR_C1"
            )
            is True
        )

    def test_reviewer_cannot_set_applied_fixes(self):
        with pytest.raises(
            ConstitutionalViolation, match="exclusive to role 'corrector'"
        ):
            validate_state_mutation({"applied_fixes": []}, "REVIEWER")

    def test_tester_can_set_test_results(self):
        assert (
            validate_state_mutation({"test_results": {"passed": True}}, "TESTER")
            is True
        )

    def test_coder_cannot_set_test_results(self):
        with pytest.raises(ConstitutionalViolation, match="exclusive to role 'tester'"):
            validate_state_mutation({"test_results": {}}, "CODER")

    def test_archivist_can_set_archived_summary_ref(self):
        assert (
            validate_state_mutation({"archived_summary_ref": "ref-123"}, "ARCHIVIST_A1")
            is True
        )

    def test_coder_cannot_set_archived_summary_ref(self):
        with pytest.raises(
            ConstitutionalViolation, match="exclusive to role 'archivist'"
        ):
            validate_state_mutation({"archived_summary_ref": "ref"}, "CODER")

    def test_reviewer_can_set_issues(self):
        assert validate_state_mutation({"issues": []}, "REVIEWER") is True

    def test_verifier_can_set_verifier_feedback(self):
        assert validate_state_mutation({"verifier_feedback": {}}, "VERIFIER") is True

    # ------------------------------------------------------------------
    # Valid mutations (non-exclusive, non-protected)
    # ------------------------------------------------------------------

    def test_coder_can_set_status_code_notes(self):
        assert (
            validate_state_mutation(
                {"status": "generated", "code": {"main.py": "pass"}, "notes": "done"},
                "CODER",
            )
            is True
        )

    def test_corrector_can_set_status_and_code(self):
        assert (
            validate_state_mutation(
                {"status": "fixed", "code": {"main.py": "fixed"}},
                "CORRECTOR_C1",
            )
            is True
        )

    # ------------------------------------------------------------------
    # Absorbed from verify_constitution.py
    # ------------------------------------------------------------------

    def test_corrector_can_modify_code_and_applied_fixes(self):
        state_update = {
            "code": {"test.py": "print('hello')"},
            "applied_fixes": [{"file": "test.py", "diff": "..."}],
        }
        assert validate_state_mutation(state_update, "CORRECTOR_C1") is True

    def test_coder_can_modify_code(self):
        assert validate_state_mutation({"code": {"main.py": "pass"}}, "CODER") is True

    def test_coder_blocked_from_plan(self):
        with pytest.raises(ConstitutionalViolation):
            validate_state_mutation({"plan": {}}, "CODER")


# ───────────────────────────────────────────────────────────────────────────
# emit_audit_for_violation
# ───────────────────────────────────────────────────────────────────────────


class TestEmitAuditForViolation:
    def test_returns_required_keys(self):
        violation = ConstitutionalViolation("test violation")
        result = emit_audit_for_violation("TEST_NODE", violation)
        assert "timestamp" in result
        assert "severity" in result
        assert "type" in result
        assert "node" in result
        assert "violation" in result

    def test_severity_is_critical(self):
        violation = ConstitutionalViolation("test")
        result = emit_audit_for_violation("NODE", violation)
        assert result["severity"] == "CRITICAL"

    def test_type_is_constitutional_violation(self):
        violation = ConstitutionalViolation("test")
        result = emit_audit_for_violation("NODE", violation)
        assert result["type"] == "CONSTITUTIONAL_VIOLATION"

    def test_node_matches_input(self):
        violation = ConstitutionalViolation("test")
        result = emit_audit_for_violation("MY_NODE", violation)
        assert result["node"] == "MY_NODE"

    def test_violation_matches_exception_message(self):
        msg = "attempted to set protected keys"
        violation = ConstitutionalViolation(msg)
        result = emit_audit_for_violation("NODE", violation)
        assert result["violation"] == msg


# ───────────────────────────────────────────────────────────────────────────
# enforce_constitutional_rules decorator
# ───────────────────────────────────────────────────────────────────────────


class TestEnforceConstitutionalRules:
    def test_valid_mutation_passes_through(self):
        @enforce_constitutional_rules
        def good_node(state: dict) -> dict:
            return {"status": "OK", "notes": "all good"}

        result = good_node({})
        assert result["status"] == "OK"
        assert result["notes"] == "all good"

    def test_task_in_output_returns_hir(self):
        @enforce_constitutional_rules
        def bad_node(state: dict) -> dict:
            return {"task": "overwritten", "status": "OK"}

        result = bad_node({})
        assert result["status"] == "HIR"
        assert "HIR_reason" in result
        assert "violated_at" in result

    def test_unauthorized_approval_key_returns_hir(self):
        @enforce_constitutional_rules
        def executor_node(state: dict) -> dict:
            return {"approved_summary": True, "status": "OK"}

        result = executor_node({})
        assert result["status"] == "HIR"
        assert "Constitutional violation" in result["HIR_reason"]

    def test_hir_dict_has_correct_structure(self):
        @enforce_constitutional_rules
        def violating_node(state: dict) -> dict:
            return {"task": "bad"}

        result = violating_node({})
        assert set(result.keys()) == {"status", "HIR_reason", "violated_at"}

    def test_violated_at_uses_uppercase_function_name(self):
        @enforce_constitutional_rules
        def my_custom_node(state: dict) -> dict:
            return {"task": "bad"}

        result = my_custom_node({})
        assert result["violated_at"] == "MY_CUSTOM_NODE"

    def test_exclusive_key_violation_returns_hir(self):
        @enforce_constitutional_rules
        def coder_node(state: dict) -> dict:
            return {"plan": {"steps": []}, "status": "OK"}

        result = coder_node({})
        assert result["status"] == "HIR"
        assert "violated_at" in result
