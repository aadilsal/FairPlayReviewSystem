from API.utils.lbw_decision import (
    normalize_lbw_decision,
    resolve_final_lbw_decision,
    sanitize_prediction_decisions,
)


def test_normalize_lbw_decision_accepts_binary_values_only():
    assert normalize_lbw_decision("OUT") == "OUT"
    assert normalize_lbw_decision("not out") == "NOT OUT"
    assert normalize_lbw_decision("NOT_OUT") == "NOT OUT"
    assert normalize_lbw_decision("NO_DECISION") is None


def test_inconclusive_model_decision_falls_back_to_original_decision():
    decision, review_outcome, original = resolve_final_lbw_decision(
        model_decision="NO_DECISION",
        original_decision="OUT",
    )

    assert decision == "OUT"
    assert original == "OUT"
    assert review_outcome == "inconclusive"


def test_inconclusive_model_decision_uses_default_when_original_missing():
    decision, review_outcome, original = resolve_final_lbw_decision(
        model_decision="UNKNOWN",
        original_decision=None,
    )

    assert decision == "NOT OUT"
    assert original is None
    assert review_outcome == "inconclusive"


def test_sanitize_prediction_decisions_rewrites_nested_decision_fields():
    payload = {
        "decision": "NO_DECISION",
        "lbw": {"decision": "NO_DECISION"},
        "lbw_overlay": {"decision": "NO_DECISION"},
    }

    sanitized = sanitize_prediction_decisions(payload, "OUT")

    assert sanitized["decision"] == "OUT"
    assert sanitized["lbw"]["decision"] == "OUT"
    assert sanitized["lbw_overlay"]["decision"] == "OUT"