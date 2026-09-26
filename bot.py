"""Public challenge composition adapter over Vera's frozen deterministic engine."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping

from src.context_store import ContextStore
from src.decision import DecisionAction, decide
from src.message import MessagePlan, compose_message
from src.models import NormalizedContext

# The public compose contract has no time argument. Use the repository's fixed
# seed-replay instant so expiry decisions never depend on the host clock.
_COMPOSE_NOW = datetime(2026, 4, 26, 10, 0, tzinfo=timezone.utc)


def _context_parts(scope: str, value: Any) -> tuple[str, int, dict[str, Any]]:
    """Extract a context ID and payload from a dataset mapping or normalized model."""
    if isinstance(value, NormalizedContext):
        if value.context_scope != scope:
            raise ValueError(f"expected {scope} context")
        return value.context_id, value.version, deepcopy(value.raw_payload)
    if not isinstance(value, Mapping):
        raise ValueError(f"{scope} context must be a mapping or normalized context")

    payload = deepcopy(dict(value))
    id_fields = {
        "category": ("slug",),
        "merchant": ("merchant_id",),
        "trigger": ("trigger_id", "id"),
        "customer": ("customer_id",),
    }
    context_id = next((payload.get(field) for field in id_fields[scope] if payload.get(field)), None)
    if context_id is None:
        raise ValueError(f"{scope} context ID is missing")
    return str(context_id), 1, payload


def _result(plan: MessagePlan) -> dict[str, Any]:
    """Expose the documented composition fields without adding engine metadata."""
    return {
        "body": plan.body,
        "cta": plan.cta.value,
        "send_as": plan.send_as.value if plan.send_as is not None else None,
        "suppression_key": plan.suppression_key,
        "rationale": plan.rationale,
    }


def compose(
    category: Mapping[str, Any] | NormalizedContext,
    merchant: Mapping[str, Any] | NormalizedContext,
    trigger: Mapping[str, Any] | NormalizedContext,
    customer: Mapping[str, Any] | NormalizedContext | None = None,
) -> dict[str, Any]:
    """Compose a deterministic, grounded message from challenge context inputs.

    Inputs may be raw dictionaries loaded from the challenge dataset or the
    corresponding normalized models. The frozen decision and message engines
    remain authoritative; a non-SEND engine plan is returned without a body.
    """
    store = ContextStore()
    try:
        for scope, value in (("category", category), ("merchant", merchant), ("trigger", trigger)):
            context_id, version, payload = _context_parts(scope, value)
            store.put(scope, context_id, version, payload)
        if customer is not None:
            context_id, version, payload = _context_parts("customer", customer)
            store.put("customer", context_id, version, payload)

        trigger_id, _, _ = _context_parts("trigger", trigger)
        decision = decide(store, trigger_id, _COMPOSE_NOW)
        return _result(compose_message(store, decision))
    except (TypeError, ValueError):
        # Invalid or incomplete inputs cannot be turned into a marketing claim.
        return {
            "body": None,
            "cta": "none",
            "send_as": None,
            "suppression_key": None,
            "rationale": "Context is invalid or incomplete; no message was composed.",
        }
