"""Unit tests for src/routing.py — routing functions and feedback nodes."""

import pytest
from langgraph.graph import END

from src.config import (
    ARCHIVAL_ENABLED,
    MAX_ARCHIVAL_ATTEMPTS,
    MAX_CORRECTION_ATTEMPTS,
    MAX_LOOP_ITERATIONS,
    MAX_META_ATTEMPTS,
    MAX_VALIDATION_ATTEMPTS,
)
from src.routing import (
    a2_feedback_to_a1_node,
    route_after_a2_feedback,
    route_after_architect,
    route_after_archivist_a1,
    route_after_archivist_a2,
    route_after_archivist_queue_assembly,
    route_after_coder,
    route_after_corrector,
    route_after_entry_status_capture,
    route_after_hard_reset,
    route_after_reviewer,
    route_after_tester,
    route_after_validator,
    route_after_validator_feedback_a1,
    route_after_validator_feedback_a2,
    route_after_verifier,
    route_after_verifier_reset,
    validator_feedback_to_a1_node,
    validator_feedback_to_a2_node,
    verifier_reset_node,
)


def make_state(**overrides):
    """Build a minimal valid state dict with sensible defaults."""
    base = {
        "status": "",
        "correction_attempt": 0,
        "archival_attempt": 0,
        "meta_attempt": 0,
        "validation_attempt": 0,
        "loop_iteration": 0,
        "verifier_reset_used": False,
        "event_log": [],
        "audit_log": [],
        "task_id": "test-task-id",
    }
    base.update(overrides)
    return base


# ───────────────────────────────────────────────────────────────────────────
# route_after_architect
# ───────────────────────────────────────────────────────────────────────────


class TestRouteAfterArchitect:
    def test_plan_ready_goes_to_coder(self):
        assert route_after_architect(make_state(status="plan_ready")) == "CODER"

    def test_hir_goes_to_hir_halt(self):
        assert route_after_architect(make_state(status="HIR")) == "HIR_HALT"

    def test_any_non_hir_goes_to_coder(self):
        assert route_after_architect(make_state(status="garbage")) == "CODER"


# ───────────────────────────────────────────────────────────────────────────
# route_after_coder
# ───────────────────────────────────────────────────────────────────────────


class TestRouteAfterCoder:
    def test_generated_goes_to_reviewer(self):
        assert route_after_coder(make_state(status="generated")) == "REVIEWER"

    def test_hir_goes_to_hir_halt(self):
        assert route_after_coder(make_state(status="HIR")) == "HIR_HALT"

    def test_any_non_hir_goes_to_reviewer(self):
        assert route_after_coder(make_state(status="garbage")) == "REVIEWER"


# ───────────────────────────────────────────────────────────────────────────
# route_after_reviewer
# ───────────────────────────────────────────────────────────────────────────


class TestRouteAfterReviewer:
    def test_ok_goes_to_tester(self):
        assert route_after_reviewer(make_state(status="OK")) == "TESTER"

    def test_error_l1_goes_to_corrector(self):
        assert route_after_reviewer(make_state(status="error_L1")) == "CORRECTOR"

    def test_error_l2_goes_to_corrector(self):
        assert route_after_reviewer(make_state(status="error_L2")) == "CORRECTOR"

    def test_error_l3_goes_to_verifier(self):
        assert route_after_reviewer(make_state(status="error_L3")) == "VERIFIER"

    def test_hir_goes_to_hir_halt(self):
        assert route_after_reviewer(make_state(status="HIR")) == "HIR_HALT"

    def test_unexpected_goes_to_hir_halt(self):
        assert route_after_reviewer(make_state(status="garbage")) == "HIR_HALT"


# ───────────────────────────────────────────────────────────────────────────
# route_after_tester
# ───────────────────────────────────────────────────────────────────────────


class TestRouteAfterTester:
    def test_pass_goes_to_entry_status_capture(self):
        assert route_after_tester(make_state(status="PASS")) == "ENTRY_STATUS_CAPTURE"

    def test_fail_goes_to_corrector(self):
        assert route_after_tester(make_state(status="FAIL")) == "CORRECTOR"

    def test_hir_goes_to_hir_halt(self):
        assert route_after_tester(make_state(status="HIR")) == "HIR_HALT"

    def test_unexpected_goes_to_hir_halt(self):
        assert route_after_tester(make_state(status="garbage")) == "HIR_HALT"


# ───────────────────────────────────────────────────────────────────────────
# route_after_corrector
# ───────────────────────────────────────────────────────────────────────────


class TestRouteAfterCorrector:
    def test_fixed_goes_to_reviewer(self):
        assert route_after_corrector(make_state(status="fixed")) == "REVIEWER"

    def test_needs_rewrite_goes_to_verifier(self):
        assert route_after_corrector(make_state(status="NEEDS_REWRITE")) == "VERIFIER"

    def test_no_change_below_max_goes_to_reviewer(self):
        assert (
            route_after_corrector(make_state(status="no_change", correction_attempt=0))
            == "REVIEWER"
        )

    def test_no_change_at_max_goes_to_reviewer(self):
        assert (
            route_after_corrector(
                make_state(
                    status="no_change", correction_attempt=MAX_CORRECTION_ATTEMPTS
                )
            )
            == "REVIEWER"
        )

    def test_no_change_above_max_goes_to_verifier(self):
        assert (
            route_after_corrector(
                make_state(
                    status="no_change", correction_attempt=MAX_CORRECTION_ATTEMPTS + 1
                )
            )
            == "VERIFIER"
        )

    def test_hir_goes_to_hir_halt(self):
        assert route_after_corrector(make_state(status="HIR")) == "HIR_HALT"

    def test_unexpected_goes_to_hir_halt(self):
        assert route_after_corrector(make_state(status="garbage")) == "HIR_HALT"


# ───────────────────────────────────────────────────────────────────────────
# route_after_verifier
# ───────────────────────────────────────────────────────────────────────────


class TestRouteAfterVerifier:
    def test_execution_failure_first_time_resets(self):
        assert (
            route_after_verifier(
                make_state(status="execution_failure", verifier_reset_used=False)
            )
            == "VERIFIER_RESET_THEN_CORRECTOR"
        )

    def test_execution_failure_already_reset_goes_hir(self):
        assert (
            route_after_verifier(
                make_state(status="execution_failure", verifier_reset_used=True)
            )
            == "HIR_HALT"
        )

    def test_confirmed_l3_goes_to_entry_status_capture(self):
        assert (
            route_after_verifier(make_state(status="confirmed_L3"))
            == "ENTRY_STATUS_CAPTURE"
        )

    def test_confirmed_needs_rewrite_goes_to_entry_status_capture(self):
        assert (
            route_after_verifier(make_state(status="confirmed_needs_rewrite"))
            == "ENTRY_STATUS_CAPTURE"
        )

    def test_disagree_l3_goes_to_corrector(self):
        assert route_after_verifier(make_state(status="disagree_L3")) == "CORRECTOR"

    def test_hir_goes_to_hir_halt(self):
        assert route_after_verifier(make_state(status="HIR")) == "HIR_HALT"

    def test_unexpected_goes_to_hir_halt(self):
        assert route_after_verifier(make_state(status="garbage")) == "HIR_HALT"


# ───────────────────────────────────────────────────────────────────────────
# route_after_entry_status_capture
# ───────────────────────────────────────────────────────────────────────────


class TestRouteAfterEntryStatusCapture:
    def test_always_goes_to_archivist_queue_assembly(self):
        assert (
            route_after_entry_status_capture(make_state()) == "ARCHIVIST_QUEUE_ASSEMBLY"
        )


# ───────────────────────────────────────────────────────────────────────────
# route_after_archivist_queue_assembly
# ───────────────────────────────────────────────────────────────────────────


class TestRouteAfterArchivistQueueAssembly:
    def test_archival_disabled_goes_to_end(self):
        assert route_after_archivist_queue_assembly(make_state()) == END

    def test_archival_enabled_goes_to_archivist_a1(self, monkeypatch):
        monkeypatch.setattr("src.routing.ARCHIVAL_ENABLED", True)
        assert route_after_archivist_queue_assembly(make_state()) == "ARCHIVIST_A1"


# ───────────────────────────────────────────────────────────────────────────
# route_after_archivist_a1
# ───────────────────────────────────────────────────────────────────────────


class TestRouteAfterArchivistA1:
    def test_archived_goes_to_validator(self):
        assert route_after_archivist_a1(make_state(status="archived")) == "VALIDATOR"

    def test_archival_error_below_max_retries(self):
        assert (
            route_after_archivist_a1(
                make_state(status="archival_error", archival_attempt=0)
            )
            == "ARCHIVIST_A1"
        )

    def test_archival_error_at_max_minus_one_retries(self):
        assert (
            route_after_archivist_a1(
                make_state(
                    status="archival_error",
                    archival_attempt=MAX_ARCHIVAL_ATTEMPTS - 1,
                )
            )
            == "ARCHIVIST_A1"
        )

    def test_archival_error_at_max_goes_hir(self):
        assert (
            route_after_archivist_a1(
                make_state(
                    status="archival_error",
                    archival_attempt=MAX_ARCHIVAL_ATTEMPTS,
                )
            )
            == "HIR_HALT"
        )

    def test_hir_goes_to_hir_halt(self):
        assert route_after_archivist_a1(make_state(status="HIR")) == "HIR_HALT"

    def test_unexpected_goes_to_hir_halt(self):
        assert route_after_archivist_a1(make_state(status="garbage")) == "HIR_HALT"


# ───────────────────────────────────────────────────────────────────────────
# route_after_archivist_a2
# ───────────────────────────────────────────────────────────────────────────


class TestRouteAfterArchivistA2:
    def test_archived_meta_summary_goes_to_validator(self):
        assert (
            route_after_archivist_a2(make_state(status="archived_meta_summary"))
            == "VALIDATOR"
        )

    def test_meta_summary_error_below_max_retries(self):
        assert (
            route_after_archivist_a2(
                make_state(status="meta_summary_error", meta_attempt=0)
            )
            == "ARCHIVIST_A2"
        )

    def test_meta_summary_error_at_max_minus_one_retries(self):
        assert (
            route_after_archivist_a2(
                make_state(
                    status="meta_summary_error",
                    meta_attempt=MAX_META_ATTEMPTS - 1,
                )
            )
            == "ARCHIVIST_A2"
        )

    def test_meta_summary_error_at_max_goes_hir(self):
        assert (
            route_after_archivist_a2(
                make_state(status="meta_summary_error", meta_attempt=MAX_META_ATTEMPTS)
            )
            == "HIR_HALT"
        )

    def test_archived_summary_corrupt_goes_to_a2_feedback(self):
        assert (
            route_after_archivist_a2(make_state(status="archived_summary_corrupt"))
            == "A2_FEEDBACK_TO_A1"
        )

    def test_hir_goes_to_hir_halt(self):
        assert route_after_archivist_a2(make_state(status="HIR")) == "HIR_HALT"

    def test_unexpected_goes_to_hir_halt(self):
        assert route_after_archivist_a2(make_state(status="garbage")) == "HIR_HALT"


# ───────────────────────────────────────────────────────────────────────────
# route_after_a2_feedback
# ───────────────────────────────────────────────────────────────────────────


class TestRouteAfterA2Feedback:
    def test_below_max_goes_to_archivist_a1(self):
        assert route_after_a2_feedback(make_state(archival_attempt=0)) == "ARCHIVIST_A1"

    def test_at_max_minus_one_goes_to_archivist_a1(self):
        assert (
            route_after_a2_feedback(
                make_state(archival_attempt=MAX_ARCHIVAL_ATTEMPTS - 1)
            )
            == "ARCHIVIST_A1"
        )

    def test_at_max_goes_to_hir_halt(self):
        assert (
            route_after_a2_feedback(make_state(archival_attempt=MAX_ARCHIVAL_ATTEMPTS))
            == "HIR_HALT"
        )

    def test_above_max_goes_to_hir_halt(self):
        assert (
            route_after_a2_feedback(
                make_state(archival_attempt=MAX_ARCHIVAL_ATTEMPTS + 1)
            )
            == "HIR_HALT"
        )


# ───────────────────────────────────────────────────────────────────────────
# route_after_validator — the most complex routing function
# ───────────────────────────────────────────────────────────────────────────


class TestRouteAfterValidator:
    # Case 1: HIR → HIR_HALT
    def test_hir_goes_to_hir_halt(self):
        assert route_after_validator(make_state(status="HIR")) == "HIR_HALT"

    # Case 2: COMPLETED → END
    def test_completed_goes_to_end(self):
        assert route_after_validator(make_state(status="COMPLETED")) == END

    # Case 3: approved_hard_reset with low loop_iteration → HARD_RESET
    def test_approved_hard_reset_low_loop(self):
        assert (
            route_after_validator(
                make_state(status="approved_hard_reset", loop_iteration=0)
            )
            == "HARD_RESET"
        )

    # Case 4: approved_hard_reset at loop limit → HIR_HALT
    def test_approved_hard_reset_at_loop_limit(self):
        assert (
            route_after_validator(
                make_state(
                    status="approved_hard_reset",
                    loop_iteration=MAX_LOOP_ITERATIONS - 1,
                )
            )
            == "HIR_HALT"
        )

    def test_approved_hard_reset_above_loop_limit(self):
        assert (
            route_after_validator(
                make_state(
                    status="approved_hard_reset",
                    loop_iteration=MAX_LOOP_ITERATIONS,
                )
            )
            == "HIR_HALT"
        )

    # Case 5: unapproved_archived_summary below max → VALIDATOR_FEEDBACK_TO_A1
    def test_unapproved_archived_summary_below_max(self):
        assert (
            route_after_validator(
                make_state(status="unapproved_archived_summary", validation_attempt=0)
            )
            == "VALIDATOR_FEEDBACK_TO_A1"
        )

    # Case 6: unapproved_archived_summary at max → HIR_HALT
    def test_unapproved_archived_summary_at_max(self):
        assert (
            route_after_validator(
                make_state(
                    status="unapproved_archived_summary",
                    validation_attempt=MAX_VALIDATION_ATTEMPTS,
                )
            )
            == "HIR_HALT"
        )

    # Case 7: unapproved_meta_summary below max → VALIDATOR_FEEDBACK_TO_A2
    def test_unapproved_meta_summary_below_max(self):
        assert (
            route_after_validator(
                make_state(status="unapproved_meta_summary", validation_attempt=0)
            )
            == "VALIDATOR_FEEDBACK_TO_A2"
        )

    # Case 8: unapproved_meta_summary at max → HIR_HALT
    def test_unapproved_meta_summary_at_max(self):
        assert (
            route_after_validator(
                make_state(
                    status="unapproved_meta_summary",
                    validation_attempt=MAX_VALIDATION_ATTEMPTS,
                )
            )
            == "HIR_HALT"
        )

    # Case 9: ready_for_next, PASS, no meta → ARCHIVIST_A2
    def test_ready_for_next_pass_no_meta(self):
        assert (
            route_after_validator(
                make_state(
                    status="ready_for_next",
                    entry_status="PASS",
                    archived_meta_summary_ref=None,
                )
            )
            == "ARCHIVIST_A2"
        )

    # Case 10: ready_for_next, PASS, has meta → END
    def test_ready_for_next_pass_has_meta(self):
        assert (
            route_after_validator(
                make_state(
                    status="ready_for_next",
                    entry_status="PASS",
                    archived_meta_summary_ref="some-ref",
                )
            )
            == END
        )

    # Case 11: ready_for_next, FAIL, no meta → CORRECTOR
    def test_ready_for_next_fail_no_meta(self):
        assert (
            route_after_validator(
                make_state(
                    status="ready_for_next",
                    entry_status="FAIL",
                    archived_meta_summary_ref=None,
                )
            )
            == "CORRECTOR"
        )

    # Case 12: ready_for_next, FAIL, has meta → CORRECTOR
    def test_ready_for_next_fail_has_meta(self):
        assert (
            route_after_validator(
                make_state(
                    status="ready_for_next",
                    entry_status="FAIL",
                    archived_meta_summary_ref="some-ref",
                )
            )
            == "CORRECTOR"
        )

    # Case 13: ready_for_next, error_L1, no meta → CORRECTOR
    def test_ready_for_next_error_l1_no_meta(self):
        assert (
            route_after_validator(
                make_state(
                    status="ready_for_next",
                    entry_status="error_L1",
                    archived_meta_summary_ref=None,
                )
            )
            == "CORRECTOR"
        )

    # Case 14: ready_for_next, error_L2, no meta → CORRECTOR
    def test_ready_for_next_error_l2_no_meta(self):
        assert (
            route_after_validator(
                make_state(
                    status="ready_for_next",
                    entry_status="error_L2",
                    archived_meta_summary_ref=None,
                )
            )
            == "CORRECTOR"
        )

    # Case 15: ready_for_next, confirmed_L3, no meta → ARCHIVIST_A2
    def test_ready_for_next_confirmed_l3_no_meta(self):
        assert (
            route_after_validator(
                make_state(
                    status="ready_for_next",
                    entry_status="confirmed_L3",
                    archived_meta_summary_ref=None,
                )
            )
            == "ARCHIVIST_A2"
        )

    # Case 16: ready_for_next, confirmed_needs_rewrite, no meta → ARCHIVIST_A2
    def test_ready_for_next_confirmed_needs_rewrite_no_meta(self):
        assert (
            route_after_validator(
                make_state(
                    status="ready_for_next",
                    entry_status="confirmed_needs_rewrite",
                    archived_meta_summary_ref=None,
                )
            )
            == "ARCHIVIST_A2"
        )

    # Case 17: ready_for_next, confirmed_L3, has meta → HARD_RESET
    def test_ready_for_next_confirmed_l3_has_meta(self):
        assert (
            route_after_validator(
                make_state(
                    status="ready_for_next",
                    entry_status="confirmed_L3",
                    archived_meta_summary_ref="some-ref",
                )
            )
            == "HARD_RESET"
        )

    # Case 18: ready_for_next, confirmed_needs_rewrite, has meta → HARD_RESET
    def test_ready_for_next_confirmed_needs_rewrite_has_meta(self):
        assert (
            route_after_validator(
                make_state(
                    status="ready_for_next",
                    entry_status="confirmed_needs_rewrite",
                    archived_meta_summary_ref="some-ref",
                )
            )
            == "HARD_RESET"
        )

    # Case 19: ready_for_next, unexpected entry_status, no meta, high loop → HIR_HALT
    def test_ready_for_next_unexpected_entry_status_high_loop(self):
        assert (
            route_after_validator(
                make_state(
                    status="ready_for_next",
                    entry_status="something_weird",
                    archived_meta_summary_ref=None,
                    loop_iteration=2,
                )
            )
            == "HIR_HALT"
        )

    # Case 20: unexpected top-level status → HIR_HALT
    def test_unexpected_status_goes_to_hir_halt(self):
        assert route_after_validator(make_state(status="garbage")) == "HIR_HALT"

    # Case 21: ready_for_next, PASS, no meta, high archival_attempt → ARCHIVIST_A2
    def test_ready_for_next_pass_no_meta_high_archival(self):
        assert (
            route_after_validator(
                make_state(
                    status="ready_for_next",
                    entry_status="PASS",
                    archived_meta_summary_ref=None,
                    archival_attempt=MAX_ARCHIVAL_ATTEMPTS + 1,
                )
            )
            == "ARCHIVIST_A2"
        )

    # Additional edge case: ready_for_next, unexpected entry_status, no meta, low loop → HIR_HALT
    def test_ready_for_next_unexpected_entry_status_low_loop(self):
        assert (
            route_after_validator(
                make_state(
                    status="ready_for_next",
                    entry_status="something_weird",
                    archived_meta_summary_ref=None,
                    loop_iteration=0,
                )
            )
            == "HIR_HALT"
        )


# ───────────────────────────────────────────────────────────────────────────
# route_after_validator_feedback_a1
# ───────────────────────────────────────────────────────────────────────────


class TestRouteAfterValidatorFeedbackA1:
    def test_always_goes_to_archivist_a1(self):
        assert route_after_validator_feedback_a1(make_state()) == "ARCHIVIST_A1"


# ───────────────────────────────────────────────────────────────────────────
# route_after_validator_feedback_a2
# ───────────────────────────────────────────────────────────────────────────


class TestRouteAfterValidatorFeedbackA2:
    def test_always_goes_to_archivist_a2(self):
        assert route_after_validator_feedback_a2(make_state()) == "ARCHIVIST_A2"


# ───────────────────────────────────────────────────────────────────────────
# route_after_hard_reset
# ───────────────────────────────────────────────────────────────────────────


class TestRouteAfterHardReset:
    def test_always_goes_to_architect(self):
        assert route_after_hard_reset(make_state()) == "ARCHITECT"


# ───────────────────────────────────────────────────────────────────────────
# route_after_verifier_reset
# ───────────────────────────────────────────────────────────────────────────


class TestRouteAfterVerifierReset:
    def test_always_goes_to_corrector(self):
        assert route_after_verifier_reset(make_state()) == "CORRECTOR"


# ───────────────────────────────────────────────────────────────────────────
# verifier_reset_node (system node)
# ───────────────────────────────────────────────────────────────────────────


class TestVerifierResetNode:
    def test_sets_correction_attempt_to_zero(self):
        result = verifier_reset_node(make_state(correction_attempt=3))
        assert result["correction_attempt"] == 0

    def test_sets_verifier_reset_used_to_true(self):
        result = verifier_reset_node(make_state(verifier_reset_used=False))
        assert result["verifier_reset_used"] is True

    def test_appends_event_log(self):
        result = verifier_reset_node(make_state())
        assert len(result["event_log"]) == 1
        assert result["event_log"][0]["event"] == "verifier_correction_reset"

    def test_appends_audit_log(self):
        result = verifier_reset_node(make_state())
        assert len(result["audit_log"]) == 1
        entry = result["audit_log"][0]
        assert entry["node"] == "GRAPH_CONTROLLER"
        assert entry["model_id"] == "system"
        assert entry["status"] == "verifier_reset_applied"

    def test_preserves_existing_event_log(self):
        existing = [{"event": "prior", "detail": "old", "timestamp": "t0"}]
        result = verifier_reset_node(make_state(event_log=existing))
        assert len(result["event_log"]) == 2
        assert result["event_log"][0]["event"] == "prior"

    def test_updated_at_is_set(self):
        result = verifier_reset_node(make_state())
        assert "updated_at" in result
        assert isinstance(result["updated_at"], str)


# ───────────────────────────────────────────────────────────────────────────
# a2_feedback_to_a1_node (system node)
# ───────────────────────────────────────────────────────────────────────────


class TestA2FeedbackToA1Node:
    def test_sets_archivist_feedback_from_top_level_instructions(self):
        instructions = ["fix this", "fix that"]
        result = a2_feedback_to_a1_node(
            make_state(correction_instructions=instructions)
        )
        assert result["archivist_feedback"] == instructions

    def test_extracts_instructions_from_meta_summary(self):
        instructions = ["rewrite section 3", "fix timestamps"]
        result = a2_feedback_to_a1_node(
            make_state(meta_summary={"correction_instructions": instructions})
        )
        assert result["archivist_feedback"] == instructions

    def test_top_level_instructions_take_precedence_over_meta_summary(self):
        top_level = ["top-level fix"]
        nested = ["nested fix"]
        result = a2_feedback_to_a1_node(
            make_state(
                correction_instructions=top_level,
                meta_summary={"correction_instructions": nested},
            )
        )
        assert result["archivist_feedback"] == top_level

    def test_default_feedback_when_no_instructions(self):
        result = a2_feedback_to_a1_node(make_state())
        assert len(result["archivist_feedback"]) == 1
        assert "corrupt" in result["archivist_feedback"][0].lower()

    def test_clears_archived_meta_summary_ref(self):
        result = a2_feedback_to_a1_node(make_state())
        assert result["archived_meta_summary_ref"] is None

    def test_appends_event_log(self):
        result = a2_feedback_to_a1_node(make_state())
        assert len(result["event_log"]) == 1
        assert result["event_log"][0]["event"] == "a2_feedback_to_a1"

    def test_updated_at_is_set(self):
        result = a2_feedback_to_a1_node(make_state())
        assert "updated_at" in result


# ───────────────────────────────────────────────────────────────────────────
# validator_feedback_to_a1_node (system node)
# ───────────────────────────────────────────────────────────────────────────


class TestValidatorFeedbackToA1Node:
    def test_increments_validation_attempt(self):
        result = validator_feedback_to_a1_node(make_state(validation_attempt=1))
        assert result["validation_attempt"] == 2

    def test_sets_archivist_feedback_from_instructions(self):
        instructions = ["correction 1"]
        result = validator_feedback_to_a1_node(
            make_state(correction_instructions=instructions)
        )
        assert result["archivist_feedback"] == instructions

    def test_default_feedback_when_no_instructions(self):
        result = validator_feedback_to_a1_node(make_state())
        assert len(result["archivist_feedback"]) == 1
        assert "rejected" in result["archivist_feedback"][0].lower()

    def test_appends_event_log(self):
        result = validator_feedback_to_a1_node(make_state())
        assert len(result["event_log"]) == 1
        assert result["event_log"][0]["event"] == "validator_feedback_to_a1"

    def test_updated_at_is_set(self):
        result = validator_feedback_to_a1_node(make_state())
        assert "updated_at" in result


# ───────────────────────────────────────────────────────────────────────────
# validator_feedback_to_a2_node (system node)
# ───────────────────────────────────────────────────────────────────────────


class TestValidatorFeedbackToA2Node:
    def test_increments_validation_attempt(self):
        result = validator_feedback_to_a2_node(make_state(validation_attempt=2))
        assert result["validation_attempt"] == 3

    def test_increments_meta_attempt(self):
        result = validator_feedback_to_a2_node(make_state(meta_attempt=1))
        assert result["meta_attempt"] == 2

    def test_sets_archivist_feedback_from_instructions(self):
        instructions = ["fix meta"]
        result = validator_feedback_to_a2_node(
            make_state(correction_instructions=instructions)
        )
        assert result["archivist_feedback"] == instructions

    def test_default_feedback_when_no_instructions(self):
        result = validator_feedback_to_a2_node(make_state())
        assert len(result["archivist_feedback"]) == 1
        assert "rejected" in result["archivist_feedback"][0].lower()

    def test_appends_event_log(self):
        result = validator_feedback_to_a2_node(make_state())
        assert len(result["event_log"]) == 1
        assert result["event_log"][0]["event"] == "validator_feedback_to_a2"

    def test_updated_at_is_set(self):
        result = validator_feedback_to_a2_node(make_state())
        assert "updated_at" in result


# ───────────────────────────────────────────────────────────────────────────
# Parametrized edge cases
# ───────────────────────────────────────────────────────────────────────────


class TestParametrizedEdgeCases:
    @pytest.mark.parametrize("status", ["error_L1", "error_L2"])
    def test_reviewer_error_levels_go_to_corrector(self, status):
        assert route_after_reviewer(make_state(status=status)) == "CORRECTOR"

    @pytest.mark.parametrize("status", ["confirmed_L3", "confirmed_needs_rewrite"])
    def test_verifier_confirmed_statuses_go_to_entry_capture(self, status):
        assert route_after_verifier(make_state(status=status)) == "ENTRY_STATUS_CAPTURE"

    @pytest.mark.parametrize("entry_status", ["FAIL", "error_L1", "error_L2"])
    def test_validator_ready_error_statuses_no_meta_go_to_corrector(self, entry_status):
        assert (
            route_after_validator(
                make_state(
                    status="ready_for_next",
                    entry_status=entry_status,
                    archived_meta_summary_ref=None,
                )
            )
            == "CORRECTOR"
        )

    @pytest.mark.parametrize(
        "entry_status", ["confirmed_L3", "confirmed_needs_rewrite"]
    )
    def test_validator_ready_confirmed_no_meta_go_to_a2(self, entry_status):
        assert (
            route_after_validator(
                make_state(
                    status="ready_for_next",
                    entry_status=entry_status,
                    archived_meta_summary_ref=None,
                )
            )
            == "ARCHIVIST_A2"
        )

    @pytest.mark.parametrize(
        "entry_status", ["confirmed_L3", "confirmed_needs_rewrite"]
    )
    def test_validator_ready_confirmed_has_meta_go_to_hard_reset(self, entry_status):
        assert (
            route_after_validator(
                make_state(
                    status="ready_for_next",
                    entry_status=entry_status,
                    archived_meta_summary_ref="ref",
                )
            )
            == "HARD_RESET"
        )
