"""Normalize flexible challenge JSON while retaining an untouched raw copy."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Mapping

from .models import CategoryContext, CustomerContext, MerchantContext, NormalizedContext, TriggerContext

_MISSING = object()


def _id(value: Any, field_name: str) -> str | None:
    if value is None or value is _MISSING:
        return None
    value = str(value).strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


def _payload(payload: Mapping[str, Any], context_id: Any, version: int) -> tuple[str, int, dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping")
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise ValueError("version must be a non-negative integer")
    normalized_id = _id(context_id, "context_id")
    if normalized_id is None:
        raise ValueError("context_id is required")
    raw = deepcopy(dict(payload))
    return normalized_id, version, raw


def _value(data: Mapping[str, Any], key: str) -> Any:
    value = data.get(key)
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    return None


def _items(data: Mapping[str, Any], key: str) -> list[Any] | None:
    value = data.get(key)
    if isinstance(value, list):
        return deepcopy(value)
    return None


def _text(data: Mapping[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _timestamp(value: Any) -> tuple[str | None, datetime | None]:
    """Keep the supplied timestamp string and parse a convenience datetime if valid."""
    if value is None:
        return None, None
    if isinstance(value, datetime):
        return value.isoformat(), value
    if not isinstance(value, str):
        return str(value), None
    parsed = None
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError:
        pass
    return value, parsed


def normalize_category(
    payload: Mapping[str, Any], context_id: Any = None, version: int = 1
) -> CategoryContext:
    """Normalize one category payload; omitted or malformed nested values stay absent."""
    cid = context_id if context_id is not None else payload.get("slug") if isinstance(payload, Mapping) else None
    cid, version, raw = _payload(payload, cid, version)
    return CategoryContext(
        context_id=cid, version=version, raw_payload=raw,
        slug=_text(raw, "slug"), display_name=_text(raw, "display_name"),
        voice=_value(raw, "voice"), offer_catalog=_items(raw, "offer_catalog"),
        peer_stats=_value(raw, "peer_stats"), digest=_items(raw, "digest"),
        seasonal_beats=_items(raw, "seasonal_beats"), trend_signals=_items(raw, "trend_signals"),
        patient_content_library=_items(raw, "patient_content_library"),
        regulatory_authorities=_items(raw, "regulatory_authorities"),
        professional_journals=_items(raw, "professional_journals"),
    )


def normalize_merchant(
    payload: Mapping[str, Any], context_id: Any = None, version: int = 1
) -> MerchantContext:
    """Normalize one merchant payload without deriving values from other contexts."""
    cid = context_id if context_id is not None else payload.get("merchant_id") if isinstance(payload, Mapping) else None
    cid, version, raw = _payload(payload, cid, version)
    return MerchantContext(
        context_id=cid, version=version, raw_payload=raw,
        merchant_id=_text(raw, "merchant_id"), category_slug=_text(raw, "category_slug"),
        identity=_value(raw, "identity"), subscription=_value(raw, "subscription"),
        performance=_value(raw, "performance"), offers=_items(raw, "offers"),
        conversation_history=_items(raw, "conversation_history"),
        customer_aggregate=_value(raw, "customer_aggregate"), signals=_items(raw, "signals"),
        review_themes=_items(raw, "review_themes"),
    )


def normalize_customer(
    payload: Mapping[str, Any], context_id: Any = None, version: int = 1
) -> CustomerContext:
    """Normalize one customer payload; missing consent is represented by None."""
    cid = context_id if context_id is not None else payload.get("customer_id") if isinstance(payload, Mapping) else None
    cid, version, raw = _payload(payload, cid, version)
    return CustomerContext(
        context_id=cid, version=version, raw_payload=raw,
        customer_id=_text(raw, "customer_id"), merchant_id=_text(raw, "merchant_id"),
        identity=_value(raw, "identity"), relationship=_value(raw, "relationship"),
        state=_text(raw, "state"), preferences=_value(raw, "preferences"), consent=_value(raw, "consent"),
    )


def normalize_trigger(
    payload: Mapping[str, Any], context_id: Any = None, version: int = 1
) -> TriggerContext:
    """Normalize a trigger; audience scope is retained separately from store scope."""
    id_alias = None
    if isinstance(payload, Mapping):
        if (context_id is None and payload.get("id") is not None
                and payload.get("trigger_id") is not None
                and str(payload["id"]).strip() != str(payload["trigger_id"]).strip()):
            raise ValueError("trigger id aliases 'id' and 'trigger_id' conflict")
        id_alias = payload.get("trigger_id", payload.get("id"))
    cid = context_id if context_id is not None else id_alias
    cid, version, raw = _payload(payload, cid, version)
    expires_at, expires_at_datetime = _timestamp(raw.get("expires_at"))
    urgency = raw.get("urgency")
    if isinstance(urgency, bool) or not isinstance(urgency, (int, float)):
        urgency = None
    return TriggerContext(
        context_id=cid, version=version, raw_payload=raw,
        trigger_id=_text(raw, "trigger_id") or _text(raw, "id"),
        scope=_text(raw, "scope"), kind=_text(raw, "kind"), source=_text(raw, "source"),
        merchant_id=_text(raw, "merchant_id"), customer_id=_text(raw, "customer_id"),
        payload=_value(raw, "payload"), urgency=urgency,
        suppression_key=_text(raw, "suppression_key"), expires_at=expires_at,
        expires_at_datetime=expires_at_datetime,
    )


def normalize_context(
    scope: str, context_id: Any, version: int, payload: Mapping[str, Any]
) -> NormalizedContext:
    """Dispatch normalization using the store's scope, not trigger audience scope."""
    normalizers = {
        "category": normalize_category,
        "merchant": normalize_merchant,
        "customer": normalize_customer,
        "trigger": normalize_trigger,
    }
    if scope not in normalizers:
        raise ValueError(f"unsupported context scope: {scope!r}")
    return normalizers[scope](payload, context_id=context_id, version=version)
