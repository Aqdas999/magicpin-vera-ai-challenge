"""Process-local, versioned storage for normalized challenge contexts."""

from __future__ import annotations

from copy import deepcopy
from threading import RLock
from typing import Any, Mapping

from .models import CategoryContext, CustomerContext, MerchantContext, NormalizedContext, TriggerContext
from .normalizer import normalize_context


class ContextStore:
    """Store the latest context per ``(store scope, context ID)``.

    ``put`` returns True only when a new or higher-version context is stored.
    Equal and lower versions are no-ops and return False; HTTP status policy is
    deliberately outside this class.
    """

    def __init__(self) -> None:
        self._contexts: dict[tuple[str, str], NormalizedContext] = {}
        self._lock = RLock()

    @staticmethod
    def _key(scope: str, context_id: Any) -> tuple[str, str]:
        if scope not in {"category", "merchant", "customer", "trigger"}:
            raise ValueError(f"unsupported context scope: {scope!r}")
        if context_id is None:
            raise ValueError("context_id is required")
        normalized_id = str(context_id).strip()
        if not normalized_id:
            raise ValueError("context_id must not be empty")
        return scope, normalized_id

    def put(self, scope: str, context_id: Any, version: int, payload: Mapping[str, Any]) -> bool:
        """Normalize and store a context unless its version is not newer."""
        key = self._key(scope, context_id)
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise ValueError("version must be a non-negative integer")
        with self._lock:
            current = self._contexts.get(key)
            if current is not None and version <= current.version:
                return False
            candidate = normalize_context(scope, context_id, version, payload)
            self._contexts[key] = candidate
            return True

    def get(self, scope: str, context_id: Any) -> NormalizedContext | None:
        """Return the stored model, or None when the scoped ID is absent."""
        key = self._key(scope, context_id)
        with self._lock:
            context = self._contexts.get(key)
            return deepcopy(context) if context is not None else None

    def get_version(self, scope: str, context_id: Any) -> int | None:
        """Return the current version for a scoped ID, or None if absent."""
        context = self.get(scope, context_id)
        return context.version if context is not None else None

    def has(self, scope: str, context_id: Any) -> bool:
        """Return whether a scoped context exists."""
        return self.get(scope, context_id) is not None

    def count(self, scope: str) -> int:
        """Count stored contexts in one scope."""
        if scope not in {"category", "merchant", "customer", "trigger"}:
            raise ValueError(f"unsupported context scope: {scope!r}")
        with self._lock:
            return sum(1 for stored_scope, _ in self._contexts if stored_scope == scope)

    def clear(self) -> None:
        """Remove all contexts."""
        with self._lock:
            self._contexts.clear()

    def snapshot_counts(self) -> dict[str, int]:
        """Return an independent count snapshot for all four store scopes."""
        scopes = ("category", "merchant", "customer", "trigger")
        with self._lock:
            return {scope: sum(1 for stored_scope, _ in self._contexts if stored_scope == scope)
                    for scope in scopes}

    def list_contexts(self, scope: str) -> list[NormalizedContext]:
        """Return detached contexts for one scope in stable context-ID order."""
        if scope not in {"category", "merchant", "customer", "trigger"}:
            raise ValueError(f"unsupported context scope: {scope!r}")
        with self._lock:
            return [
                deepcopy(self._contexts[(scope, context_id)])
                for context_id in sorted(
                    stored_id for stored_scope, stored_id in self._contexts if stored_scope == scope
                )
            ]

    @staticmethod
    def _merchant_key(merchant: MerchantContext) -> str | None:
        if merchant.merchant_id and merchant.context_id != merchant.merchant_id:
            return None
        return merchant.merchant_id or merchant.context_id

    @staticmethod
    def _customer_key(customer: CustomerContext) -> str | None:
        if customer.customer_id and customer.context_id != customer.customer_id:
            return None
        return customer.customer_id or customer.context_id

    @staticmethod
    def _trigger(value: TriggerContext | str, store: ContextStore) -> TriggerContext | None:
        if isinstance(value, TriggerContext):
            return value
        context = store.get("trigger", value)
        return context if isinstance(context, TriggerContext) else None

    def get_merchant_for_trigger(self, trigger: TriggerContext | str) -> MerchantContext | None:
        """Resolve a trigger's explicitly referenced merchant; never guess."""
        event = self._trigger(trigger, self)
        if event is None or not event.merchant_id:
            return None
        merchant = self.get("merchant", event.merchant_id)
        if not isinstance(merchant, MerchantContext):
            return None
        return merchant if self.validate_trigger_merchant_relationship(event, merchant) else None

    def get_customer_for_trigger(self, trigger: TriggerContext | str) -> CustomerContext | None:
        """Resolve an explicitly referenced customer if trigger and customer agree."""
        event = self._trigger(trigger, self)
        if event is None or event.scope != "customer" or not event.customer_id:
            return None
        customer = self.get("customer", event.customer_id)
        if not isinstance(customer, CustomerContext):
            return None
        if self._customer_key(customer) != event.customer_id:
            return None
        if not self.validate_customer_merchant_relationship(customer, event.merchant_id):
            return None
        return customer

    def get_category_for_merchant(self, merchant: MerchantContext | str) -> CategoryContext | None:
        """Resolve the category named by a merchant's category_slug."""
        business = merchant if isinstance(merchant, MerchantContext) else self.get("merchant", merchant)
        if not isinstance(business, MerchantContext) or not business.category_slug:
            return None
        category = self.get("category", business.category_slug)
        if not isinstance(category, CategoryContext):
            return None
        if category.slug and category.slug != business.category_slug:
            return None
        return category

    def validate_trigger_merchant_relationship(
        self, trigger: TriggerContext | str, merchant: MerchantContext | str
    ) -> bool:
        """Return True only when the trigger explicitly names this merchant."""
        event = self._trigger(trigger, self)
        business = merchant if isinstance(merchant, MerchantContext) else self.get("merchant", merchant)
        if event is None or not isinstance(business, MerchantContext) or not event.merchant_id:
            return False
        return event.merchant_id == self._merchant_key(business) == business.context_id

    def validate_customer_merchant_relationship(
        self, customer: CustomerContext | str, merchant: MerchantContext | str
    ) -> bool:
        """Return True only when the customer explicitly belongs to this merchant."""
        person = customer if isinstance(customer, CustomerContext) else self.get("customer", customer)
        business = merchant if isinstance(merchant, MerchantContext) else self.get("merchant", merchant)
        if not isinstance(person, CustomerContext) or not isinstance(business, MerchantContext):
            return False
        if self._customer_key(person) is None:
            return False
        merchant_key = self._merchant_key(business)
        if not merchant_key or not person.merchant_id:
            return False
        return person.merchant_id == merchant_key == business.context_id
