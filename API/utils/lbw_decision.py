from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple


LbwDecision = Literal["OUT", "NOT OUT"]
_ALLOWED_DECISIONS = {"OUT", "NOT OUT"}


def normalize_lbw_decision(value: Any) -> Optional[LbwDecision]:
    """Convert arbitrary decision text to API-safe LBW vocabulary."""
    if not isinstance(value, str):
        return None

    normalized = value.strip().upper().replace("_", " ")
    normalized = " ".join(normalized.split())

    if normalized == "NOTOUT":
        normalized = "NOT OUT"

    if normalized in _ALLOWED_DECISIONS:
        return normalized  # type: ignore[return-value]
    return None


def resolve_final_lbw_decision(
    model_decision: Any,
    original_decision: Any,
    fallback: LbwDecision = "NOT OUT",
) -> Tuple[LbwDecision, Optional[str], Optional[LbwDecision]]:
    """
    Resolve model output to strict binary LBW decision.

    Returns (final_decision, review_outcome, normalized_original_decision).
    `review_outcome` is "inconclusive" when model output was not binary.
    """
    normalized_original = normalize_lbw_decision(original_decision)
    normalized_model = normalize_lbw_decision(model_decision)

    if normalized_model:
        return normalized_model, None, normalized_original

    # Product rule: any inconclusive model output is surfaced as NOT OUT.
    return fallback, "inconclusive", normalized_original


def sanitize_prediction_decisions(payload: Dict[str, Any], final_decision: LbwDecision) -> Dict[str, Any]:
    """Replace nested `decision` fields with API-safe binary decision text."""

    def _sanitize(node: Any) -> Any:
        if isinstance(node, dict):
            cleaned: Dict[str, Any] = {}
            for key, value in node.items():
                if key == "decision":
                    cleaned[key] = final_decision
                else:
                    cleaned[key] = _sanitize(value)
            return cleaned
        if isinstance(node, list):
            return [_sanitize(item) for item in node]
        return node

    return _sanitize(payload)