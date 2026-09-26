"""Flexible normalized context models for the Vera challenge data layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar


@dataclass
class NormalizedContext:
    """Common metadata retained for every stored context."""

    context_id: str
    version: int
    raw_payload: dict[str, Any]
    CONTEXT_SCOPE: ClassVar[str] = ""

    @property
    def context_scope(self) -> str:
        """Return the store scope for this model, distinct from trigger audience."""
        return self.CONTEXT_SCOPE


@dataclass
class CategoryContext(NormalizedContext):
    """Normalized category knowledge; nested values remain flexible JSON data."""

    slug: str | None = None
    display_name: str | None = None
    voice: dict[str, Any] | None = None
    offer_catalog: list[Any] | None = None
    peer_stats: dict[str, Any] | None = None
    digest: list[Any] | None = None
    seasonal_beats: list[Any] | None = None
    trend_signals: list[Any] | None = None
    patient_content_library: list[Any] | None = None
    regulatory_authorities: list[Any] | None = None
    professional_journals: list[Any] | None = None
    scope: str = "category"
    CONTEXT_SCOPE: ClassVar[str] = "category"


@dataclass
class MerchantContext(NormalizedContext):
    """Normalized merchant snapshot with optional nested data."""

    merchant_id: str | None = None
    category_slug: str | None = None
    identity: dict[str, Any] | None = None
    subscription: dict[str, Any] | None = None
    performance: dict[str, Any] | None = None
    offers: list[Any] | None = None
    conversation_history: list[Any] | None = None
    customer_aggregate: dict[str, Any] | None = None
    signals: list[Any] | None = None
    review_themes: list[Any] | None = None
    scope: str = "merchant"
    CONTEXT_SCOPE: ClassVar[str] = "merchant"


@dataclass
class CustomerContext(NormalizedContext):
    """Normalized customer profile; absent consent remains absent."""

    customer_id: str | None = None
    merchant_id: str | None = None
    identity: dict[str, Any] | None = None
    relationship: dict[str, Any] | None = None
    state: str | None = None
    preferences: dict[str, Any] | None = None
    consent: dict[str, Any] | None = None
    scope: str = "customer"
    CONTEXT_SCOPE: ClassVar[str] = "customer"


@dataclass
class TriggerContext(NormalizedContext):
    """Normalized event; ``scope`` is the event audience, not store scope."""

    trigger_id: str | None = None
    scope: str | None = None
    kind: str | None = None
    source: str | None = None
    merchant_id: str | None = None
    customer_id: str | None = None
    payload: dict[str, Any] | None = None
    urgency: int | float | None = None
    suppression_key: str | None = None
    expires_at: str | None = None
    expires_at_datetime: datetime | None = None
    CONTEXT_SCOPE: ClassVar[str] = "trigger"

