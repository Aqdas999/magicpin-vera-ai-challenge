"""Deterministic trigger eligibility and action planning for Stage 2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from .context_store import ContextStore
from .models import CategoryContext, CustomerContext, MerchantContext, TriggerContext


class DecisionAction(str, Enum):
    SEND = "SEND"
    WAIT = "WAIT"
    END = "END"
    SUPPRESS = "SUPPRESS"


class CTASemantic(str, Enum):
    NONE = "NONE"
    YES_NO = "YES_NO"
    OPEN_ENDED = "OPEN_ENDED"
    MULTI_CHOICE = "MULTI_CHOICE"
    CONFIRM_CANCEL = "CONFIRM_CANCEL"


@dataclass(frozen=True)
class GroundedFact:
    """A source path and canonical JSON value preserved for later stages."""

    source: str
    value_json: str


@dataclass(frozen=True)
class DecisionPlan:
    """Immutable decision result; it contains intent and facts, never message text."""

    action: DecisionAction
    trigger_id: str | None
    merchant_id: str | None
    customer_id: str | None
    trigger_kind: str | None
    reason_code: str
    intent: str | None = None
    cta: CTASemantic = CTASemantic.NONE
    grounded_facts: tuple[GroundedFact, ...] = ()


_CUSTOMER_SCOPED = frozenset({
    "recall_due", "wedding_package_followup", "customer_lapsed_hard",
    "customer_lapsed_soft", "trial_followup", "chronic_refill_due",
    "appointment_tomorrow", "unplanned_slot_open",
})
_MERCHANT_SCOPED = frozenset({
    "research_digest", "research_digest_release", "category_research_digest_release",
    "regulation_change", "perf_dip", "renewal_due", "festival_upcoming", "festival",
    "curious_ask_due", "scheduled_recurring", "winback_eligible", "ipl_match_today",
    "review_theme_emerged", "milestone_reached", "active_planning_intent",
    "seasonal_perf_dip", "supply_alert", "category_seasonal", "gbp_unverified",
    "cde_opportunity", "competitor_opened", "perf_spike", "dormant_with_vera",
    "category_trend_movement", "weather_heatwave", "local_news_event",
})
_SUPPORTED_KINDS = _CUSTOMER_SCOPED | _MERCHANT_SCOPED
_CUSTOMER_CONSENT_SCOPE = {
    "recall_due": "recall_reminders",
    "customer_lapsed_soft": "recall_reminders",
    "wedding_package_followup": "bridal_package_followup",
    "customer_lapsed_hard": "winback_offers",
    "trial_followup": "kids_program_updates",
    "chronic_refill_due": "refill_reminders",
    "appointment_tomorrow": "appointment_reminders",
    "unplanned_slot_open": "promotional_offers",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _facts(values: list[tuple[str, Any]]) -> tuple[GroundedFact, ...]:
    return tuple(GroundedFact(source, _canonical_json(value)) for source, value in values if value is not None)


def _payload_facts(payload: dict[str, Any], *keys: str) -> list[tuple[str, Any]]:
    return [(f"trigger.payload.{key}", payload[key]) for key in keys if key in payload and payload[key] is not None]


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _all_present(data: dict[str, Any], *keys: str) -> bool:
    return all(key in data and _present(data[key]) for key in keys)


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError:
        return None


def _comparable(left: datetime, right: datetime) -> bool:
    return (left.tzinfo is None) == (right.tzinfo is None)


def _make_plan(
    action: DecisionAction,
    trigger: TriggerContext | None,
    reason_code: str,
    *,
    intent: str | None = None,
    cta: CTASemantic = CTASemantic.NONE,
    facts: tuple[GroundedFact, ...] = (),
    merchant_id: str | None = None,
    customer_id: str | None = None,
    trigger_kind: str | None = None,
    trigger_id: str | None = None,
) -> DecisionPlan:
    return DecisionPlan(
        action=action,
        trigger_id=trigger_id if trigger_id is not None else ((trigger.trigger_id or trigger.context_id) if trigger else None),
        merchant_id=merchant_id if merchant_id is not None else (trigger.merchant_id if trigger else None),
        customer_id=customer_id if customer_id is not None else (trigger.customer_id if trigger else None),
        trigger_kind=trigger_kind if trigger_kind is not None else (trigger.kind if trigger else None),
        reason_code=reason_code,
        intent=intent,
        cta=cta,
        grounded_facts=facts,
    )


def _suppress(trigger: TriggerContext | None, reason_code: str, **kwargs: Any) -> DecisionPlan:
    return _make_plan(DecisionAction.SUPPRESS, trigger, reason_code, **kwargs)


def _send(
    trigger: TriggerContext,
    merchant: MerchantContext,
    intent: str,
    cta: CTASemantic,
    facts: list[tuple[str, Any]],
    customer: CustomerContext | None = None,
) -> DecisionPlan:
    return _make_plan(
        DecisionAction.SEND, trigger, "eligible_grounded_action",
        intent=intent, cta=cta, facts=_facts(facts), merchant_id=merchant.merchant_id,
        customer_id=customer.customer_id if customer else None,
    )


def _active_offers(merchant: MerchantContext) -> list[dict[str, Any]]:
    if not isinstance(merchant.offers, list):
        return []
    return [offer for offer in merchant.offers
            if isinstance(offer, dict) and offer.get("status") == "active"]


def _digest_item(category: CategoryContext, item_id: Any) -> dict[str, Any] | None:
    if not item_id or not isinstance(category.digest, list):
        return None
    for item in category.digest:
        if isinstance(item, dict) and item.get("id") == item_id:
            return item
    return None


def _find_digest_by_terms(category: CategoryContext, terms: tuple[str, ...]) -> dict[str, Any] | None:
    if not isinstance(category.digest, list):
        return None
    for item in category.digest:
        if not isinstance(item, dict):
            continue
        text = " ".join(str(item.get(key, "")) for key in ("title", "summary", "actionable")).casefold()
        if all(term.casefold() in text for term in terms):
            return item
    return None


def _digest_facts(item: dict[str, Any], prefix: str = "category.digest") -> list[tuple[str, Any]]:
    return [(f"{prefix}.{key}", item[key]) for key in ("id", "kind", "title", "source", "date", "summary", "actionable")
            if item.get(key) is not None]


def _consent_allows(customer: CustomerContext, kind: str) -> bool:
    consent = customer.consent
    required_scope = _CUSTOMER_CONSENT_SCOPE.get(kind)
    if not isinstance(consent, dict) or not _present(consent.get("opted_in_at")) or not required_scope:
        return False
    scopes = consent.get("scope")
    return isinstance(scopes, list) and required_scope in scopes


def _is_placeholder(payload: dict[str, Any] | None) -> bool:
    return not isinstance(payload, dict) or payload.get("placeholder") is True


def decide(store: ContextStore, trigger_id: str | None, now: datetime) -> DecisionPlan:
    """Evaluate one currently stored trigger using only supplied contexts and time."""
    if not isinstance(trigger_id, str) or not trigger_id.strip():
        return _suppress(None, "missing_trigger_id")
    stored = store.get("trigger", trigger_id)
    if not isinstance(stored, TriggerContext):
        return _suppress(None, "trigger_not_found", trigger_id=trigger_id)
    trigger = stored
    kind = (trigger.kind or "").strip().casefold()
    if kind not in _SUPPORTED_KINDS:
        return _suppress(trigger, "unknown_trigger_kind")
    if not isinstance(now, datetime):
        return _suppress(trigger, "invalid_now")

    if trigger.expires_at is not None:
        expiry = trigger.expires_at_datetime
        if expiry is None:
            return _make_plan(
                DecisionAction.WAIT, trigger, "expiry_unparseable",
                facts=_facts([("trigger.expires_at", trigger.expires_at)]),
            )
        if not _comparable(expiry, now):
            return _make_plan(
                DecisionAction.WAIT, trigger, "expiry_timezone_incomparable",
                facts=_facts([("trigger.expires_at", trigger.expires_at)]),
            )
        if expiry <= now:
            return _make_plan(
                DecisionAction.END, trigger, "trigger_expired",
                facts=_facts([("trigger.expires_at", trigger.expires_at)]),
            )

    expected_scope = "customer" if kind in _CUSTOMER_SCOPED else "merchant"
    if trigger.scope != expected_scope:
        return _suppress(trigger, "trigger_scope_mismatch")

    merchant = store.get_merchant_for_trigger(trigger)
    if not isinstance(merchant, MerchantContext) or not trigger.merchant_id:
        return _suppress(trigger, "merchant_relationship_missing")
    if merchant.merchant_id != trigger.merchant_id:
        return _suppress(trigger, "merchant_relationship_mismatch")
    category = store.get_category_for_merchant(merchant)
    if not isinstance(category, CategoryContext):
        return _suppress(trigger, "category_relationship_missing", merchant_id=merchant.merchant_id)
    if not category.slug or category.slug != merchant.category_slug:
        return _suppress(trigger, "category_relationship_mismatch", merchant_id=merchant.merchant_id)

    customer = None
    if expected_scope == "customer":
        customer = store.get_customer_for_trigger(trigger)
        if not isinstance(customer, CustomerContext) or not trigger.customer_id:
            return _suppress(trigger, "customer_relationship_missing", merchant_id=merchant.merchant_id)
        if customer.customer_id != trigger.customer_id:
            return _suppress(trigger, "customer_relationship_mismatch", merchant_id=merchant.merchant_id)
        if customer.merchant_id != merchant.merchant_id:
            return _suppress(trigger, "customer_merchant_mismatch", merchant_id=merchant.merchant_id)
        if not _consent_allows(customer, kind):
            return _suppress(
                trigger, "customer_consent_missing_or_out_of_scope",
                merchant_id=merchant.merchant_id, customer_id=customer.customer_id,
            )

    payload = trigger.payload
    if _is_placeholder(payload):
        return _suppress(trigger, "placeholder_or_missing_payload", merchant_id=merchant.merchant_id,
                         customer_id=customer.customer_id if customer else None)

    # Customer-specific families: only explicitly supplied facts are carried forward.
    if kind in {"recall_due", "customer_lapsed_soft"}:
        needed = ("service_due", "due_date") if kind == "recall_due" else ("last_visit", "due_date")
        if not _all_present(payload, *needed):
            return _suppress(trigger, "insufficient_trigger_payload", merchant_id=merchant.merchant_id,
                             customer_id=customer.customer_id)
        facts = _payload_facts(payload, *needed)
        if kind == "recall_due":
            facts.extend(_payload_facts(payload, "last_service_date", "available_slots"))
        return _send(trigger, merchant, "remind_customer_of_explicit_due_event_and_invite_booking",
                     CTASemantic.MULTI_CHOICE if payload.get("available_slots") else CTASemantic.OPEN_ENDED,
                     facts, customer)

    if kind == "wedding_package_followup":
        if category.slug != "salons":
            return _suppress(trigger, "category_not_supported_for_family", merchant_id=merchant.merchant_id,
                             customer_id=customer.customer_id)
        if not _all_present(payload, "wedding_date", "next_step_window_open"):
            return _suppress(trigger, "insufficient_trigger_payload", merchant_id=merchant.merchant_id,
                             customer_id=customer.customer_id)
        facts = _payload_facts(payload, "wedding_date", "trial_completed", "days_to_wedding", "next_step_window_open")
        return _send(trigger, merchant, "offer_a_consented_next_step_for_the_documented_wedding_timeline",
                     CTASemantic.YES_NO, facts, customer)

    if kind == "customer_lapsed_hard":
        if customer.state != "lapsed_hard" or not _all_present(payload, "days_since_last_visit"):
            return _suppress(trigger, "customer_state_or_lapse_fact_mismatch", merchant_id=merchant.merchant_id,
                             customer_id=customer.customer_id)
        facts = _payload_facts(payload, "days_since_last_visit", "previous_focus", "previous_membership_months")
        return _send(trigger, merchant, "propose_a_no_pressure_return_step_using_only_recorded_customer_context",
                     CTASemantic.YES_NO, facts, customer)

    if kind == "trial_followup":
        if not _all_present(payload, "trial_date"):
            return _suppress(trigger, "insufficient_trigger_payload", merchant_id=merchant.merchant_id,
                             customer_id=customer.customer_id)
        facts = _payload_facts(payload, "trial_date", "next_session_options")
        cta = CTASemantic.MULTI_CHOICE if payload.get("next_session_options") else CTASemantic.OPEN_ENDED
        return _send(trigger, merchant, "follow_up_on_the_recorded_trial_without_inventing_session_availability",
                     cta, facts, customer)

    if kind == "chronic_refill_due":
        if category.slug != "pharmacies":
            return _suppress(trigger, "category_not_supported_for_family", merchant_id=merchant.merchant_id,
                             customer_id=customer.customer_id)
        if not _all_present(payload, "molecule_list", "stock_runs_out_iso"):
            return _suppress(trigger, "insufficient_trigger_payload", merchant_id=merchant.merchant_id,
                             customer_id=customer.customer_id)
        facts = _payload_facts(payload, "molecule_list", "stock_runs_out_iso", "delivery_address_saved")
        return _send(trigger, merchant, "prompt_a_consent_scoped_refill_confirmation_without_claiming_stock_or_dosage",
                     CTASemantic.CONFIRM_CANCEL, facts, customer)

    # Merchant-facing families.
    if kind in {"research_digest", "research_digest_release", "category_research_digest_release"}:
        item = _digest_item(category, payload.get("top_item_id"))
        item_source = "category.digest"
        if item is None and isinstance(payload.get("top_item"), dict):
            item = payload["top_item"]
            item_source = "trigger.payload.top_item"
        if item is None or not _all_present(item, "title", "source"):
            return _suppress(trigger, "digest_item_not_grounded", merchant_id=merchant.merchant_id)
        facts = _payload_facts(payload, "category") + _digest_facts(item, prefix=item_source)
        return _send(trigger, merchant, "share_the_matched_source_cited_category_research_item",
                     CTASemantic.OPEN_ENDED, facts)

    if kind == "regulation_change":
        item = _digest_item(category, payload.get("top_item_id"))
        if item is None or not _all_present(item, "title", "source"):
            return _suppress(trigger, "regulatory_item_not_grounded", merchant_id=merchant.merchant_id)
        facts = _payload_facts(payload, "deadline_iso") + _digest_facts(item)
        return _send(trigger, merchant, "surface_the_matched_regulatory_source_and_its_supplied_deadline",
                     CTASemantic.YES_NO, facts)

    if kind == "cde_opportunity":
        item = _digest_item(category, payload.get("digest_item_id"))
        if item is None or not _all_present(item, "title", "source"):
            return _suppress(trigger, "cde_item_not_grounded", merchant_id=merchant.merchant_id)
        item_date = _as_datetime(item.get("date"))
        if item_date is None or not _comparable(item_date, now):
            return _suppress(trigger, "cde_date_unavailable_or_incomparable", merchant_id=merchant.merchant_id,
                             facts=_facts(_payload_facts(payload, "digest_item_id", "credits", "fee")))
        if item_date <= now:
            return _make_plan(DecisionAction.END, trigger, "cde_opportunity_has_started_or_passed",
                              merchant_id=merchant.merchant_id, facts=_facts(_digest_facts(item)))
        facts = _payload_facts(payload, "credits", "fee") + _digest_facts(item)
        return _send(trigger, merchant, "invite_the_merchant_to_the_matched_upcoming_professional_event",
                     CTASemantic.YES_NO, facts)

    if kind in {"perf_dip", "perf_spike", "seasonal_perf_dip"}:
        metric, delta = payload.get("metric"), payload.get("delta_pct")
        if not isinstance(metric, str) or not _number(delta) or not _all_present(payload, "window"):
            return _suppress(trigger, "insufficient_trigger_payload", merchant_id=merchant.merchant_id)
        if kind == "perf_dip" and delta >= 0 or kind == "perf_spike" and delta <= 0:
            return _suppress(trigger, "performance_direction_mismatch", merchant_id=merchant.merchant_id)
        if kind == "seasonal_perf_dip":
            if category.slug != "gyms" or payload.get("is_expected_seasonal") is not True or not _all_present(payload, "season_note"):
                return _suppress(trigger, "seasonal_context_not_supported", merchant_id=merchant.merchant_id)
        if not isinstance(merchant.performance, dict) or metric not in merchant.performance:
            return _suppress(trigger, "merchant_performance_missing", merchant_id=merchant.merchant_id)
        if kind == "seasonal_perf_dip":
            intent = "reframe_the_supplied_gym_dip_as_expected_and_focus_on_retention"
        elif kind == "perf_dip":
            intent = "surface_the_documented_metric_decline_and_offer_a_review"
        else:
            intent = "acknowledge_the_documented_metric_increase_without_claiming_its_cause"
        facts = _payload_facts(payload, "metric", "delta_pct", "window", "vs_baseline", "is_expected_seasonal", "season_note")
        facts.append((f"merchant.performance.{metric}", merchant.performance[metric]))
        return _send(trigger, merchant, intent, CTASemantic.OPEN_ENDED, facts)

    if kind == "renewal_due":
        subscription = merchant.subscription or {}
        if subscription.get("status") not in {"active", "trial"} or not _all_present(payload, "days_remaining", "plan", "renewal_amount"):
            return _suppress(trigger, "renewal_context_incomplete", merchant_id=merchant.merchant_id)
        if subscription.get("plan") is not None and subscription["plan"] != payload["plan"]:
            return _suppress(trigger, "renewal_plan_mismatch", merchant_id=merchant.merchant_id)
        if subscription.get("days_remaining") is not None and subscription["days_remaining"] != payload["days_remaining"]:
            return _suppress(trigger, "renewal_timing_mismatch", merchant_id=merchant.merchant_id)
        facts = _payload_facts(payload, "days_remaining", "plan", "renewal_amount")
        return _send(trigger, merchant, "present_the_supplied_plan_renewal_details", CTASemantic.CONFIRM_CANCEL, facts)

    if kind in {"festival_upcoming", "festival"}:
        relevance = payload.get("category_relevance")
        if not _all_present(payload, "festival", "date", "days_until") or not isinstance(relevance, list):
            return _suppress(trigger, "insufficient_trigger_payload", merchant_id=merchant.merchant_id)
        if category.slug not in relevance or not _number(payload["days_until"]) or payload["days_until"] <= 0:
            return _suppress(trigger, "festival_not_relevant_or_not_upcoming", merchant_id=merchant.merchant_id)
        return _send(trigger, merchant, "suggest_one_category_relevant_festival_preparation_step",
                     CTASemantic.OPEN_ENDED, _payload_facts(payload, "festival", "date", "days_until", "category_relevance"))

    if kind in {"curious_ask_due", "scheduled_recurring"}:
        if payload.get("ask_template") != "what_service_in_demand_this_week":
            return _suppress(trigger, "unsupported_recurring_ask", merchant_id=merchant.merchant_id)
        facts = _payload_facts(payload, "ask_template", "last_ask_at")
        last_ask = _as_datetime(payload.get("last_ask_at"))
        if payload.get("last_ask_at") is not None:
            if last_ask is None or not _comparable(last_ask, now):
                return _make_plan(DecisionAction.WAIT, trigger, "last_ask_time_unparseable",
                                  merchant_id=merchant.merchant_id, facts=_facts(facts))
            if now - last_ask < timedelta(days=7):
                return _make_plan(DecisionAction.WAIT, trigger, "weekly_ask_cadence_not_elapsed",
                                  merchant_id=merchant.merchant_id, facts=_facts(facts))
        return _send(trigger, merchant, "ask_one_documented_category_relevant_business_question",
                     CTASemantic.OPEN_ENDED, facts)

    if kind == "winback_eligible":
        if not merchant.subscription or merchant.subscription.get("status") != "expired":
            return _suppress(trigger, "merchant_not_expired", merchant_id=merchant.merchant_id)
        if not _all_present(payload, "days_since_expiry"):
            return _suppress(trigger, "insufficient_trigger_payload", merchant_id=merchant.merchant_id)
        return _send(trigger, merchant, "offer_a_grounded_subscription_reactivation_review",
                     CTASemantic.YES_NO, _payload_facts(payload, "days_since_expiry", "perf_dip_pct", "lapsed_customers_added_since_expiry"))

    if kind == "ipl_match_today":
        if category.slug != "restaurants":
            return _suppress(trigger, "category_not_supported_for_family", merchant_id=merchant.merchant_id)
        if not _all_present(payload, "match", "venue", "city", "match_time_iso") or not isinstance(payload.get("is_weeknight"), bool):
            return _suppress(trigger, "insufficient_trigger_payload", merchant_id=merchant.merchant_id)
        offers = _active_offers(merchant)
        if not offers or not _present(offers[0].get("title")):
            return _suppress(trigger, "active_merchant_offer_missing", merchant_id=merchant.merchant_id)
        facts = _payload_facts(payload, "match", "venue", "city", "match_time_iso", "is_weeknight")
        offer = offers[0]
        facts.extend([(f"merchant.offers[0].{key}", offer[key]) for key in ("id", "title", "status") if offer.get(key) is not None])
        if not payload["is_weeknight"]:
            item = _find_digest_by_terms(category, ("ipl", "saturdays", "underperformed"))
            if item is None:
                return _suppress(trigger, "weekend_match_evidence_missing", merchant_id=merchant.merchant_id)
            facts.extend(_digest_facts(item))
            intent = "avoid_a_generic_weekend_match_promo_and_review_the_existing_offer"
        else:
            intent = "consider_the_existing_merchant_offer_for_the_supplied_match"
        return _send(trigger, merchant, intent, CTASemantic.OPEN_ENDED, facts)

    if kind == "review_theme_emerged":
        theme, occurrences = payload.get("theme"), payload.get("occurrences_30d")
        if not isinstance(theme, str) or not _number(occurrences) or occurrences <= 0:
            return _suppress(trigger, "insufficient_trigger_payload", merchant_id=merchant.merchant_id)
        matching = next((item for item in (merchant.review_themes or [])
                         if isinstance(item, dict) and item.get("theme") == theme), None)
        if matching is None:
            return _suppress(trigger, "review_theme_not_in_merchant_context", merchant_id=merchant.merchant_id)
        facts = _payload_facts(payload, "theme", "occurrences_30d", "trend", "common_quote")
        facts.extend((f"merchant.review_themes.{theme}.{key}", matching[key]) for key in ("sentiment", "occurrences_30d") if matching.get(key) is not None)
        return _send(trigger, merchant, "review_the_matched_recurring_customer_feedback_theme",
                     CTASemantic.OPEN_ENDED, facts)

    if kind == "milestone_reached":
        value, target = payload.get("value_now"), payload.get("milestone_value")
        if not _all_present(payload, "metric", "value_now", "milestone_value") or not _number(value) or not _number(target):
            return _suppress(trigger, "insufficient_trigger_payload", merchant_id=merchant.merchant_id)
        imminent = payload.get("is_imminent") is True
        if value >= target:
            intent = "acknowledge_the_explicitly_reached_milestone"
        elif imminent:
            intent = "prepare_a_milestone_acknowledgement_without_claiming_it_was_reached"
        else:
            return _suppress(trigger, "milestone_not_reached_or_marked_imminent", merchant_id=merchant.merchant_id)
        return _send(trigger, merchant, intent, CTASemantic.YES_NO,
                     _payload_facts(payload, "metric", "value_now", "milestone_value", "is_imminent"))

    if kind == "active_planning_intent":
        if not _all_present(payload, "intent_topic", "merchant_last_message"):
            return _suppress(trigger, "explicit_planning_intent_missing", merchant_id=merchant.merchant_id)
        return _send(trigger, merchant, "produce_a_starter_plan_for_the_explicitly_requested_topic",
                     CTASemantic.OPEN_ENDED, _payload_facts(payload, "intent_topic", "merchant_last_message"))

    if kind == "supply_alert":
        if category.slug != "pharmacies":
            return _suppress(trigger, "category_not_supported_for_family", merchant_id=merchant.merchant_id)
        if not _all_present(payload, "molecule", "affected_batches", "manufacturer"):
            return _suppress(trigger, "insufficient_trigger_payload", merchant_id=merchant.merchant_id)
        if not isinstance(payload["affected_batches"], list) or not payload["affected_batches"]:
            return _suppress(trigger, "affected_batches_missing", merchant_id=merchant.merchant_id)
        facts = _payload_facts(payload, "alert_id", "molecule", "affected_batches", "manufacturer")
        return _send(trigger, merchant, "surface_the_supplied_batch_alert_and_propose_a_stock_check_without_patient_claims",
                     CTASemantic.YES_NO, facts)

    if kind == "category_seasonal":
        trends = payload.get("trends")
        if category.slug != "pharmacies" or not _all_present(payload, "season", "trends") or not isinstance(trends, list):
            return _suppress(trigger, "seasonal_category_payload_incomplete", merchant_id=merchant.merchant_id)
        return _send(trigger, merchant, "review_the_supplied_pharmacy_seasonal_demand_signals_for_shelf_planning",
                     CTASemantic.OPEN_ENDED, _payload_facts(payload, "season", "trends", "shelf_action_recommended"))

    if kind == "gbp_unverified":
        verified = merchant.identity.get("verified") if isinstance(merchant.identity, dict) else None
        if verified is not False or payload.get("verified") is not False or not _all_present(payload, "verification_path"):
            return _suppress(trigger, "gbp_verification_state_not_grounded", merchant_id=merchant.merchant_id)
        facts = _payload_facts(payload, "verification_path")
        facts.append(("merchant.identity.verified", verified))
        return _send(trigger, merchant, "offer_help_with_the_supplied_business_profile_verification_path",
                     CTASemantic.YES_NO, facts)

    if kind == "competitor_opened":
        if not _all_present(payload, "competitor_name", "distance_km", "their_offer", "opened_date") or not _number(payload["distance_km"]):
            return _suppress(trigger, "competitor_event_incomplete", merchant_id=merchant.merchant_id)
        return _send(trigger, merchant, "review_positioning_against_the_explicitly_named_nearby_competitor",
                     CTASemantic.OPEN_ENDED, _payload_facts(payload, "competitor_name", "distance_km", "their_offer", "opened_date"))

    if kind == "dormant_with_vera":
        days = payload.get("days_since_last_merchant_message")
        if not _number(days) or days <= 0:
            return _suppress(trigger, "dormancy_interval_missing", merchant_id=merchant.merchant_id)
        return _send(trigger, merchant, "resume_the_last_documented_topic_with_one_low_friction_step",
                     CTASemantic.OPEN_ENDED, _payload_facts(payload, "days_since_last_merchant_message", "last_topic"))

    if kind == "category_trend_movement":
        query, delta = payload.get("query"), payload.get("delta_yoy")
        if not isinstance(query, str) or not _number(delta) or not isinstance(category.trend_signals, list):
            return _suppress(trigger, "trend_payload_incomplete", merchant_id=merchant.merchant_id)
        signal = next((item for item in category.trend_signals
                       if isinstance(item, dict) and item.get("query") == query and item.get("delta_yoy") == delta), None)
        if signal is None:
            return _suppress(trigger, "trend_not_in_category_context", merchant_id=merchant.merchant_id)
        facts = _payload_facts(payload, "query", "delta_yoy")
        facts.extend((f"category.trend_signals.{query}.{key}", signal[key])
                     for key in ("segment_age", "skew") if signal.get(key) is not None)
        return _send(trigger, merchant, "share_the_exact_matched_category_search_trend",
                     CTASemantic.OPEN_ENDED, facts)

    # These kinds are named in the brief/design, but the supplied materials do not
    # define stable payload fields or a sufficiently specific decision rule.
    if kind in {"weather_heatwave", "local_news_event", "appointment_tomorrow", "unplanned_slot_open"}:
        return _suppress(trigger, "trigger_semantics_or_payload_schema_underspecified", merchant_id=merchant.merchant_id,
                         customer_id=customer.customer_id if customer else None)

    return _suppress(trigger, "no_safe_family_rule", merchant_id=merchant.merchant_id,
                     customer_id=customer.customer_id if customer else None)
