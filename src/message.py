"""Deterministic, grounded outbound message composition for Stage 3A."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .context_store import ContextStore
from .decision import CTASemantic, DecisionAction, DecisionPlan, GroundedFact
from .models import CategoryContext, CustomerContext, MerchantContext, TriggerContext


class MessageCTA(str, Enum):
    NONE = "none"
    YES_NO = "binary_yes_no"
    OPEN_ENDED = "open_ended"
    MULTI_CHOICE = "multi_choice_slot"
    CONFIRM_CANCEL = "binary_confirm_cancel"


class SendAs(str, Enum):
    VERA = "vera"
    MERCHANT_ON_BEHALF = "merchant_on_behalf"


@dataclass(frozen=True)
class MessagePlan:
    """Immutable message result. Non-send plans have no body or send identity."""

    action: DecisionAction
    trigger_id: str | None
    merchant_id: str | None
    customer_id: str | None
    trigger_kind: str | None
    body: str | None
    cta: MessageCTA
    send_as: SendAs | None
    suppression_key: str | None
    rationale: str
    reason_code: str | None = None


class _UnsafeComposition(ValueError):
    pass


class _Evidence:
    """Reads required values only from the DecisionPlan or its linked contexts."""

    def __init__(
        self,
        plan: DecisionPlan,
        trigger: TriggerContext,
        category: CategoryContext,
        merchant: MerchantContext,
        customer: CustomerContext | None,
    ):
        self.plan = plan
        self.trigger = trigger
        self.category = category
        self.merchant = merchant
        self.customer = customer
        self.used: list[str] = []
        self._facts = {fact.source: fact for fact in plan.grounded_facts}

    @staticmethod
    def _canonical(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _source_values(self, source: str) -> list[Any]:
        parts = source.split(".")
        if parts[0] == "trigger" and len(parts) >= 2:
            if parts[1] == "payload":
                value: Any = self.trigger.payload
                for part in parts[2:]:
                    if not isinstance(value, dict) or part not in value:
                        return []
                    value = value[part]
                return [value]
            if len(parts) == 2 and hasattr(self.trigger, parts[1]):
                return [getattr(self.trigger, parts[1])]
        if parts[0] == "category" and len(parts) >= 3:
            collection_name = parts[1]
            if collection_name == "digest" and len(parts) == 3:
                items = self.category.digest
                return [item[parts[2]] for item in items or []
                        if isinstance(item, dict) and parts[2] in item]
            if collection_name == "trend_signals" and len(parts) == 4:
                items = self.category.trend_signals
                return [item[parts[3]] for item in items or []
                        if isinstance(item, dict) and item.get("query") == parts[2] and parts[3] in item]
        if parts[0] == "merchant" and len(parts) >= 3:
            if parts[1] == "performance" and len(parts) == 3 and isinstance(self.merchant.performance, dict):
                return [self.merchant.performance[parts[2]]] if parts[2] in self.merchant.performance else []
            if parts[1] == "identity" and len(parts) == 3 and isinstance(self.merchant.identity, dict):
                return [self.merchant.identity[parts[2]]] if parts[2] in self.merchant.identity else []
            if parts[1].startswith("offers[") and parts[1].endswith("]") and len(parts) == 3:
                try:
                    index = int(parts[1][7:-1])
                    offer = self.merchant.offers[index]
                    return [offer[parts[2]]] if isinstance(offer, dict) and parts[2] in offer else []
                except (IndexError, TypeError, ValueError):
                    return []
            if parts[1] == "review_themes" and len(parts) == 4:
                for item in self.merchant.review_themes or []:
                    if isinstance(item, dict) and item.get("theme") == parts[2] and parts[3] in item:
                        return [item[parts[3]]]
        return []

    @staticmethod
    def display(value: Any) -> str:
        if isinstance(value, str):
            return value.strip().replace("_", " ")
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return format(value, "g")
        if isinstance(value, list):
            return ", ".join(_Evidence.display(item) for item in value)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        raise _UnsafeComposition("unsupported or empty value")

    def fact(self, *sources: str, required: bool = True, track: bool = True) -> Any:
        for source in sources:
            fact = self._facts.get(source)
            if fact is None:
                continue
            try:
                value = json.loads(fact.value_json)
            except (TypeError, json.JSONDecodeError) as exc:
                raise _UnsafeComposition("grounded fact is not valid JSON") from exc
            if value is None:
                continue
            candidates = self._source_values(source)
            if not any(self._canonical(candidate) == self._canonical(value) for candidate in candidates):
                raise _UnsafeComposition(f"grounded value does not match source: {source}")
            rendered = self.display(value)
            if not rendered:
                continue
            if track:
                self.used.append(rendered)
            return value
        if required:
            raise _UnsafeComposition(f"required grounded fact missing: {sources[0]}")
        return None

    def context_text(self, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise _UnsafeComposition("linked context value missing")
        rendered = value.strip()
        self.used.append(rendered)
        return rendered

    def text(self, *sources: str, required: bool = True) -> str | None:
        value = self.fact(*sources, required=required)
        return self.display(value) if value is not None else None


@dataclass(frozen=True)
class _Draft:
    lead: str
    rationale: str
    question: str | None
    evidence: tuple[str, ...]
    options: tuple[str, ...] = ()


_CTA_FROM_SEMANTIC = {
    CTASemantic.NONE: MessageCTA.NONE,
    CTASemantic.YES_NO: MessageCTA.YES_NO,
    CTASemantic.OPEN_ENDED: MessageCTA.OPEN_ENDED,
    CTASemantic.MULTI_CHOICE: MessageCTA.MULTI_CHOICE,
    CTASemantic.CONFIRM_CANCEL: MessageCTA.CONFIRM_CANCEL,
}


def _render_list(evidence: _Evidence, source: str, *, required: bool = True) -> tuple[str | None, tuple[str, ...]]:
    used_count = len(evidence.used)
    values = evidence.fact(source, required=required)
    if values is None:
        return None, ()
    if not isinstance(values, list) or not values:
        raise _UnsafeComposition("choice or list data is not a non-empty list")
    # The source fact is the structured list; render an explicit `label` for
    # slot/session option objects when supplied, otherwise render each value.
    del evidence.used[used_count:]
    rendered_values = []
    for value in values:
        if isinstance(value, dict) and isinstance(value.get("label"), str) and value["label"].strip():
            rendered_value = value["label"].strip()
        else:
            rendered_value = evidence.display(value)
        evidence.used.append(rendered_value)
        rendered_values.append(rendered_value)
    rendered = tuple(rendered_values)
    if any(not item for item in rendered):
        raise _UnsafeComposition("choice list contains an empty option")
    return ", ".join(rendered), rendered


def _build_draft(plan: DecisionPlan, ev: _Evidence) -> _Draft:
    intent = plan.intent
    question: str | None = None
    options: tuple[str, ...] = ()

    if intent == "remind_customer_of_explicit_due_event_and_invite_booking":
        due = ev.text("trigger.payload.due_date")
        if plan.trigger_kind == "customer_lapsed_soft":
            last_visit = ev.text("trigger.payload.last_visit")
            lead = f"Your recorded last visit was {last_visit}, and your recall window opens on {due}."
            question = "Would you like to book a time?"
            rationale = f"Customer recall follow-up based on the recorded last visit {last_visit} and recall window {due}."
        else:
            service = ev.text("trigger.payload.service_due")
            lead = f"Your recorded {service} is due on {due}."
            if plan.cta == CTASemantic.MULTI_CHOICE:
                _, options = _render_list(ev, "trigger.payload.available_slots")
            question = "Would you like to book a time?"
            rationale = f"Customer reminder for the recorded {service} due date {due}."

    elif intent == "offer_a_consented_next_step_for_the_documented_wedding_timeline":
        wedding = ev.text("trigger.payload.wedding_date")
        window = ev.text("trigger.payload.next_step_window_open")
        lead = f"Your recorded wedding date is {wedding}; the supplied next-step window is {window}."
        question = "Would you like to discuss a next step?"
        rationale = f"Following up on the recorded wedding date {wedding} during the supplied next-step window."

    elif intent == "propose_a_no_pressure_return_step_using_only_recorded_customer_context":
        days = ev.text("trigger.payload.days_since_last_visit")
        focus = ev.text("trigger.payload.previous_focus", required=False)
        lead = f"It has been {days} days since your last visit."
        if focus:
            lead += f" Your recorded previous focus was {focus}."
        question = "Would you like to discuss a low-pressure return option?"
        rationale = f"Customer return follow-up based on the recorded {days}-day lapse."

    elif intent == "follow_up_on_the_recorded_trial_without_inventing_session_availability":
        trial_date = ev.text("trigger.payload.trial_date")
        lead = f"Following up on your recorded trial date, {trial_date}."
        if plan.cta == CTASemantic.MULTI_CHOICE:
            _, options = _render_list(ev, "trigger.payload.next_session_options")
            question = "Which supplied session option would you like to discuss?"
        else:
            question = "What would you like to know about a follow-up session?"
        rationale = f"Following up on the trial recorded for {trial_date}; no availability is assumed."

    elif intent == "prompt_a_consent_scoped_refill_confirmation_without_claiming_stock_or_dosage":
        molecules, _ = _render_list(ev, "trigger.payload.molecule_list")
        date = ev.text("trigger.payload.stock_runs_out_iso")
        lead = f"The refill record lists {molecules}, with a recorded run-out date of {date}."
        question = "Would you like to confirm a refill, or cancel this request?"
        rationale = f"Refill follow-up using the recorded molecule list and run-out date {date}."

    elif intent == "share_the_matched_source_cited_category_research_item":
        title = ev.text("category.digest.title", "trigger.payload.top_item.title")
        source = ev.text("category.digest.source", "trigger.payload.top_item.source")
        lead = f"A relevant research item is titled “{title}” and lists {source} as its source."
        question = "Would you like to review it?"
        rationale = f"Sharing the grounded research item “{title}” from {source}."

    elif intent == "surface_the_matched_regulatory_source_and_its_supplied_deadline":
        title = ev.text("category.digest.title")
        source = ev.text("category.digest.source")
        deadline = ev.text("trigger.payload.deadline_iso", required=False)
        lead = f"The regulatory item “{title}” is from {source}."
        if deadline:
            lead += f" Its supplied deadline is {deadline}."
        question = "Would you like to review the update?"
        rationale = f"Surfacing the matched regulatory item “{title}” from {source}."

    elif intent in {
        "surface_the_documented_metric_decline_and_offer_a_review",
        "acknowledge_the_documented_metric_increase_without_claiming_its_cause",
        "reframe_the_supplied_gym_dip_as_expected_and_focus_on_retention",
    }:
        metric = ev.text("trigger.payload.metric")
        delta = ev.fact("trigger.payload.delta_pct", track=False)
        if not isinstance(delta, (int, float)) or isinstance(delta, bool) or delta == 0:
            raise _UnsafeComposition("performance delta has no usable direction")
        amount = ev.display(abs(delta))
        ev.used.append(amount)
        window = ev.text("trigger.payload.window")
        movement = "increased" if delta > 0 else "declined"
        lead = f"Your recorded {metric} {movement} by {amount}% over {window}."
        note = ev.text("trigger.payload.season_note", required=False)
        if note:
            lead += f" The supplied seasonal note says: {note}."
        question = "Would you like to review possible next steps?"
        rationale = f"Responding to the documented {amount}% {metric} {movement} over {window}."

    elif intent == "present_the_supplied_plan_renewal_details":
        days = ev.text("trigger.payload.days_remaining")
        plan_name = ev.text("trigger.payload.plan")
        amount = ev.text("trigger.payload.renewal_amount")
        lead = f"Your {plan_name} plan has {days} days remaining; the supplied renewal amount is {amount}."
        question = "Would you like to renew this plan?"
        rationale = f"Presenting the recorded {plan_name} renewal details and amount {amount}."

    elif intent == "suggest_one_category_relevant_festival_preparation_step":
        festival = ev.text("trigger.payload.festival")
        date = ev.text("trigger.payload.date")
        days = ev.text("trigger.payload.days_until")
        lead = f"{festival} is listed for {date}, {days} days away."
        question = "What preparation step would be most useful for your business?"
        rationale = f"Planning around the supplied {festival} date {date}."

    elif intent == "ask_one_documented_category_relevant_business_question":
        template = ev.fact("trigger.payload.ask_template", track=False)
        if template != "what_service_in_demand_this_week":
            raise _UnsafeComposition("unsupported curiosity question template")
        lead = "Quick question for this week."
        question = "What service are customers asking about most?"
        rationale = "Using the documented weekly service-demand question template."

    elif intent == "offer_a_grounded_subscription_reactivation_review":
        days = ev.text("trigger.payload.days_since_expiry")
        lead = f"The supplied subscription context records expiry {days} days ago."
        question = "Would you like to review reactivation?"
        rationale = f"Offering a subscription reactivation review based on the recorded {days}-day interval."

    elif intent in {
        "consider_the_existing_merchant_offer_for_the_supplied_match",
        "avoid_a_generic_weekend_match_promo_and_review_the_existing_offer",
    }:
        match = ev.text("trigger.payload.match")
        venue = ev.text("trigger.payload.venue")
        city = ev.text("trigger.payload.city")
        match_time = ev.text("trigger.payload.match_time_iso")
        offer = ev.text("merchant.offers[0].title")
        weeknight = ev.fact("trigger.payload.is_weeknight", track=False)
        lead = f"For {match} at {venue}, {city} ({match_time}), your recorded offer is {offer}."
        if weeknight is False or intent == "avoid_a_generic_weekend_match_promo_and_review_the_existing_offer":
            digest = ev.text("category.digest.title", required=False)
            if digest is None:
                raise _UnsafeComposition("weekend match evidence is missing")
            lead += f" The category digest notes: {digest}."
        question = "Would you like to review this offer for the match?"
        rationale = f"Reviewing the recorded merchant offer {offer} for the supplied match context."

    elif intent == "invite_the_merchant_to_the_matched_upcoming_professional_event":
        title = ev.text("category.digest.title")
        source = ev.text("category.digest.source")
        date = ev.text("category.digest.date")
        credits = ev.text("trigger.payload.credits")
        fee = ev.text("trigger.payload.fee")
        lead = f"The listed professional event is “{title}” from {source} on {date}; the trigger lists {credits} credits and fee {fee}."
        question = "Would you like to review the event details?"
        rationale = f"Sharing the matched professional event “{title}” dated {date}."

    elif intent == "review_the_matched_recurring_customer_feedback_theme":
        theme = ev.text("trigger.payload.theme")
        count = ev.text("trigger.payload.occurrences_30d")
        lead = f"The supplied review theme is “{theme}”, appearing {count} times in the last 30 days."
        question = "What would you like to address first?"
        rationale = f"Reviewing the matched feedback theme “{theme}” and its supplied 30-day count."

    elif intent in {
        "acknowledge_the_explicitly_reached_milestone",
        "prepare_a_milestone_acknowledgement_without_claiming_it_was_reached",
    }:
        metric = ev.text("trigger.payload.metric")
        value = ev.text("trigger.payload.value_now")
        target = ev.text("trigger.payload.milestone_value")
        if intent == "acknowledge_the_explicitly_reached_milestone":
            lead = f"Your recorded {metric} value has reached {value} against the {target} milestone."
            rationale = f"Acknowledging the recorded {metric} milestone at {value}."
        else:
            lead = f"Your recorded {metric} value is {value}, with a milestone at {target}."
            rationale = f"Preparing for the supplied {metric} milestone at {target} without claiming it is reached."
        question = "Would you like to plan a next step?"

    elif intent == "produce_a_starter_plan_for_the_explicitly_requested_topic":
        topic = ev.text("trigger.payload.intent_topic")
        lead = f"The documented planning topic is {topic}."
        question = "What should the starter plan focus on first?"
        rationale = f"Continuing the explicitly requested topic {topic}."

    elif intent == "surface_the_supplied_batch_alert_and_propose_a_stock_check_without_patient_claims":
        molecule = ev.text("trigger.payload.molecule")
        batches, _ = _render_list(ev, "trigger.payload.affected_batches")
        manufacturer = ev.text("trigger.payload.manufacturer")
        lead = f"The supplied alert lists {molecule}, batches {batches}, and manufacturer {manufacturer}."
        question = "Would you like to check these listed batches?"
        rationale = f"Surfacing the supplied batch alert for {molecule} from {manufacturer}."

    elif intent == "review_the_supplied_pharmacy_seasonal_demand_signals_for_shelf_planning":
        season = ev.text("trigger.payload.season")
        trends, _ = _render_list(ev, "trigger.payload.trends")
        lead = f"The supplied {season} demand signals are: {trends}."
        question = "Which signal would you like to review for shelf planning?"
        rationale = f"Reviewing supplied {season} demand signals for shelf planning."

    elif intent == "offer_help_with_the_supplied_business_profile_verification_path":
        path = ev.text("trigger.payload.verification_path")
        lead = f"The business profile is marked unverified; the supplied verification path is {path}."
        question = "Would you like help with this path?"
        rationale = "Offering help with the recorded business-profile verification path."

    elif intent == "review_positioning_against_the_explicitly_named_nearby_competitor":
        name = ev.text("trigger.payload.competitor_name")
        distance = ev.text("trigger.payload.distance_km")
        offer = ev.text("trigger.payload.their_offer")
        opened = ev.text("trigger.payload.opened_date")
        lead = f"The supplied update names {name}, {distance} km away, with {offer}; its opening date is {opened}."
        question = "What aspect of your positioning would you like to review?"
        rationale = f"Reviewing the supplied nearby competitor update for {name}."

    elif intent == "resume_the_last_documented_topic_with_one_low_friction_step":
        days = ev.text("trigger.payload.days_since_last_merchant_message")
        topic = ev.text("trigger.payload.last_topic", required=False)
        lead = f"It has been {days} days since the last recorded merchant message."
        if topic:
            lead += f" The last recorded topic was {topic}."
        question = "Would you like to pick this topic back up?"
        rationale = f"Resuming the recorded topic after {days} days without a merchant message."

    elif intent == "share_the_exact_matched_category_search_trend":
        query = ev.text("trigger.payload.query")
        delta = ev.text("trigger.payload.delta_yoy")
        lead = f"The category trend records a year-over-year change of {delta} for “{query}”."
        question = "How would you like to respond to this trend?"
        rationale = f"Sharing the matched category search trend for {query}."

    else:
        raise _UnsafeComposition("unsupported decision intent")

    if ev.customer is not None and isinstance(ev.customer.identity, dict):
        name = ev.customer.identity.get("name")
    elif isinstance(ev.merchant.identity, dict):
        name = ev.merchant.identity.get("owner_first_name")
    else:
        name = None
    if isinstance(name, str) and name.strip():
        salutation = ev.context_text(name)
        lead = f"Hi {salutation}, {lead}"

    if plan.cta == CTASemantic.MULTI_CHOICE and not options:
        raise _UnsafeComposition("multi-choice CTA has no grounded options")
    return _Draft(lead, rationale, question, tuple(ev.used), options)


def _message_cta(semantic: CTASemantic) -> MessageCTA:
    try:
        return _CTA_FROM_SEMANTIC[semantic]
    except (KeyError, TypeError) as exc:
        raise _UnsafeComposition("unsupported CTA semantic") from exc


def _append_cta(draft: _Draft, semantic: CTASemantic) -> str:
    if semantic == CTASemantic.NONE:
        return draft.lead
    if semantic == CTASemantic.MULTI_CHOICE:
        if not draft.options:
            raise _UnsafeComposition("multi-choice CTA has no options")
        choices = "; ".join(f"{index + 1}) {option}" for index, option in enumerate(draft.options))
        question = f"Which option works for you: {choices}?"
    elif semantic == CTASemantic.CONFIRM_CANCEL:
        question = "Would you like to proceed? Reply CONFIRM or CANCEL."
    else:
        question = draft.question
        if not question or "?" not in question:
            raise _UnsafeComposition("CTA question is missing")
    return f"{draft.lead} {question}"


def _non_send(decision: DecisionPlan, action: DecisionAction, reason: str) -> MessagePlan:
    return MessagePlan(
        action=action,
        trigger_id=decision.trigger_id,
        merchant_id=decision.merchant_id,
        customer_id=decision.customer_id,
        trigger_kind=decision.trigger_kind,
        body=None,
        cta=MessageCTA.NONE,
        send_as=None,
        suppression_key=None,
        rationale=reason,
        reason_code=reason,
    )


def _fallback_suppression(decision: DecisionPlan, reason: str) -> MessagePlan:
    return _non_send(decision, DecisionAction.SUPPRESS, reason)


def compose_message(store: ContextStore, decision: DecisionPlan) -> MessagePlan:
    """Compose one deterministic message from an authoritative SEND plan.

    This function does not re-evaluate eligibility. It validates linked context
    and the plan's grounded facts, and safely suppresses if composition is unsafe.
    """
    if decision.action != DecisionAction.SEND:
        return _non_send(decision, decision.action, f"stage2_{decision.action.value.lower()}:{decision.reason_code}")

    if not decision.trigger_id or not decision.merchant_id or not decision.trigger_kind:
        return _fallback_suppression(decision, "message_context_missing")
    trigger = store.get("trigger", decision.trigger_id)
    if not isinstance(trigger, TriggerContext) or trigger.kind != decision.trigger_kind:
        return _fallback_suppression(decision, "message_trigger_mismatch")
    if trigger.merchant_id != decision.merchant_id:
        return _fallback_suppression(decision, "message_merchant_mismatch")
    merchant = store.get_merchant_for_trigger(trigger)
    if not isinstance(merchant, MerchantContext) or merchant.merchant_id != decision.merchant_id:
        return _fallback_suppression(decision, "message_merchant_missing")
    category = store.get_category_for_merchant(merchant)
    if not isinstance(category, CategoryContext):
        return _fallback_suppression(decision, "message_category_missing")

    customer: CustomerContext | None = None
    if decision.customer_id is not None:
        if trigger.scope != "customer" or trigger.customer_id != decision.customer_id:
            return _fallback_suppression(decision, "message_customer_mismatch")
        customer = store.get_customer_for_trigger(trigger)
        if not isinstance(customer, CustomerContext) or customer.customer_id != decision.customer_id:
            return _fallback_suppression(decision, "message_customer_missing")
    elif trigger.scope == "customer":
        return _fallback_suppression(decision, "message_customer_missing")

    try:
        cta = _message_cta(decision.cta)
        evidence = _Evidence(decision, trigger, category, merchant, customer)
        draft = _build_draft(decision, evidence)
        body = _append_cta(draft, decision.cta)
        if body.count("?") != (0 if decision.cta == CTASemantic.NONE else 1):
            raise _UnsafeComposition("message must contain exactly one CTA question")
        if any(value not in body for value in draft.evidence):
            raise _UnsafeComposition("message omitted or altered a source value")
        taboos = ((category.voice or {}).get("vocab_taboo", []) if isinstance(category.voice, dict) else [])
        if any(isinstance(word, str) and word and word.casefold() in body.casefold() for word in taboos):
            raise _UnsafeComposition("message conflicts with category voice taboo")
        send_as = SendAs.MERCHANT_ON_BEHALF if customer is not None else SendAs.VERA
        suppression_key = trigger.suppression_key or ":".join((
            "vera", decision.merchant_id, decision.customer_id or "merchant",
            decision.trigger_kind, decision.intent or "action",
        ))
        return MessagePlan(
            action=DecisionAction.SEND,
            trigger_id=decision.trigger_id,
            merchant_id=decision.merchant_id,
            customer_id=decision.customer_id,
            trigger_kind=decision.trigger_kind,
            body=body,
            cta=cta,
            send_as=send_as,
            suppression_key=suppression_key,
            rationale=draft.rationale,
            reason_code=None,
        )
    except _UnsafeComposition as exc:
        return _fallback_suppression(decision, f"message_grounding_failed:{exc}")
