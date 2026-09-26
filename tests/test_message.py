"""Stage 3A deterministic message composition tests."""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import unittest

from src.context_store import ContextStore
from src.decision import CTASemantic, DecisionAction, DecisionPlan, decide
from src.message import MessageCTA, MessagePlan, SendAs, compose_message


NOW = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[1]


def seed_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class MessageTests(unittest.TestCase):
    def make_decision(
        self,
        *,
        kind="perf_dip",
        payload=None,
        merchant_id="m1",
        category_slug="restaurants",
        merchant_fields=None,
        customer=None,
        scope="merchant",
        trigger_id="t1",
        suppression_key=None,
    ):
        store = ContextStore()
        store.put("category", category_slug, 1, {
            "slug": category_slug,
            "voice": {"vocab_taboo": []},
        })
        merchant_context = {
            "merchant_id": merchant_id,
            "category_slug": category_slug,
            "identity": {"name": "Sample Business", "owner_first_name": "Asha", "verified": True},
            **(merchant_fields or {}),
        }
        store.put("merchant", merchant_id, 1, merchant_context)
        customer_id = None
        if customer is not None:
            customer_id = customer.get("customer_id", "c1")
            store.put("customer", customer_id, 1, {
                "customer_id": customer_id,
                "merchant_id": merchant_id,
                "identity": {"name": "Riya", "language_pref": "hi-en mix", "age_band": "25-35"},
                "consent": {"opted_in_at": "2025-01-01", "scope": ["recall_reminders", "bridal_package_followup", "refill_reminders"]},
                **customer,
            })
        trigger = {
            "trigger_id": trigger_id,
            "scope": scope,
            "kind": kind,
            "merchant_id": merchant_id,
            "customer_id": customer_id if scope == "customer" else None,
            "payload": payload or {},
        }
        if suppression_key is not None:
            trigger["suppression_key"] = suppression_key
        store.put("trigger", trigger_id, 1, trigger)
        return store, decide(store, trigger_id, NOW)

    def recall_plan(self, *, customer_id="c1", merchant_id="m1", slots=None):
        return self.make_decision(
            kind="recall_due",
            scope="customer",
            merchant_id=merchant_id,
            category_slug="dentists",
            customer={"customer_id": customer_id},
            payload={
                "service_due": "cleaning",
                "due_date": "2025-06-10",
                **({"available_slots": slots} if slots is not None else {}),
            },
        )

    def test_non_send_decisions_never_create_outbound_messages(self):
        store, send_decision = self.make_decision(
            payload={"metric": "orders", "delta_pct": -8, "window": "last_7d"},
            merchant_fields={"performance": {"orders": 24}},
        )
        for action in (DecisionAction.WAIT, DecisionAction.END, DecisionAction.SUPPRESS):
            with self.subTest(action=action):
                plan = compose_message(store, replace(send_decision, action=action))
                self.assertEqual(plan.action, action)
                self.assertIsNone(plan.body)
                self.assertEqual(plan.cta, MessageCTA.NONE)
                self.assertIsNone(plan.send_as)

    def test_performance_message_is_grounded_specific_and_deterministic(self):
        store, decision = self.make_decision(
            payload={"metric": "orders", "delta_pct": -8, "window": "last_7d"},
            merchant_fields={"performance": {"orders": 24}},
        )
        first = compose_message(store, decision)
        second = compose_message(store, decision)
        self.assertEqual(first, second)
        self.assertEqual(first.action, DecisionAction.SEND)
        self.assertEqual(first.cta, MessageCTA.OPEN_ENDED)
        self.assertEqual(first.send_as, SendAs.VERA)
        self.assertIn("orders", first.body)
        self.assertIn("8%", first.body)
        self.assertIn("last 7d", first.body)
        self.assertEqual(first.body.count("?"), 1)
        self.assertNotIn("because", first.body.casefold())
        self.assertNotIn("competitor", first.body.casefold())
        self.assertTrue(first.suppression_key)
        self.assertTrue(first.rationale)
        json.dumps(asdict(first), ensure_ascii=False)

    def test_customer_message_uses_explicit_identity_and_due_facts(self):
        store, decision = self.recall_plan()
        plan = compose_message(store, decision)
        self.assertEqual(plan.action, DecisionAction.SEND)
        self.assertEqual(plan.send_as, SendAs.MERCHANT_ON_BEHALF)
        self.assertIn("Riya", plan.body)
        self.assertIn("cleaning", plan.body)
        self.assertIn("2025-06-10", plan.body)
        self.assertNotIn("25-35", plan.body)
        self.assertNotIn("hi-en", plan.body.casefold())

    def test_customer_lapsed_soft_composes_from_its_decision_facts(self):
        store, decision = self.make_decision(
            kind="customer_lapsed_soft",
            scope="customer",
            category_slug="dentists",
            customer={"customer_id": "c1"},
            payload={"last_visit": "2025-05-01", "due_date": "2025-11-01"},
            suppression_key="soft-recall:c1",
        )

        self.assertEqual(decision.action, DecisionAction.SEND)
        self.assertEqual(
            {fact.source: json.loads(fact.value_json) for fact in decision.grounded_facts},
            {
                "trigger.payload.last_visit": "2025-05-01",
                "trigger.payload.due_date": "2025-11-01",
            },
        )
        plan = compose_message(store, decision)

        self.assertEqual(plan.action, DecisionAction.SEND)
        self.assertTrue(plan.body)
        self.assertIn("Riya", plan.body)
        self.assertIn("2025-05-01", plan.body)
        self.assertIn("2025-11-01", plan.body)
        self.assertNotIn("service_due", plan.body)
        self.assertNotIn("cleaning", plan.body.casefold())
        self.assertEqual(decision.cta, CTASemantic.OPEN_ENDED)
        self.assertEqual(plan.cta, MessageCTA.OPEN_ENDED)
        self.assertEqual(plan.send_as, SendAs.MERCHANT_ON_BEHALF)
        self.assertEqual(plan.suppression_key, "soft-recall:c1")
        self.assertTrue(plan.rationale)
        self.assertEqual(plan, compose_message(store, decision))

    def test_customer_lapsed_soft_missing_grounded_fact_fails_closed(self):
        store, decision = self.make_decision(
            kind="customer_lapsed_soft",
            scope="customer",
            category_slug="dentists",
            customer={"customer_id": "c1"},
            payload={"last_visit": "2025-05-01", "due_date": "2025-11-01"},
        )
        sparse = replace(
            decision,
            grounded_facts=tuple(
                fact for fact in decision.grounded_facts
                if fact.source != "trigger.payload.last_visit"
            ),
        )

        plan = compose_message(store, sparse)

        self.assertEqual(plan.action, DecisionAction.SUPPRESS)
        self.assertIsNone(plan.body)
        self.assertIn("message_grounding_failed", plan.reason_code)

    def test_multichoice_uses_only_explicit_options(self):
        slots = [
            {"label": "Tue 4 PM", "iso": "2025-06-03T16:00:00Z"},
            {"label": "Wed 11 AM", "iso": "2025-06-04T11:00:00Z"},
        ]
        store, decision = self.recall_plan(slots=slots)
        plan = compose_message(store, decision)
        self.assertEqual(plan.cta, MessageCTA.MULTI_CHOICE)
        self.assertIn("Tue 4 PM", plan.body)
        self.assertIn("Wed 11 AM", plan.body)
        self.assertNotIn("Thu", plan.body)
        self.assertEqual(plan.body.count("?"), 1)

    def test_multichoice_without_grounded_options_fails_closed(self):
        store, decision = self.recall_plan()
        contradictory = replace(decision, cta=CTASemantic.MULTI_CHOICE)
        plan = compose_message(store, contradictory)
        self.assertEqual(plan.action, DecisionAction.SUPPRESS)
        self.assertIsNone(plan.body)
        self.assertIn("message_grounding_failed", plan.reason_code)

    def test_yes_no_cta_asks_one_clear_question(self):
        store, decision = self.make_decision(
            kind="wedding_package_followup",
            scope="customer",
            category_slug="salons",
            customer={"customer_id": "c1"},
            payload={"wedding_date": "2025-10-01", "next_step_window_open": "bridal_trial"},
        )
        plan = compose_message(store, decision)
        self.assertEqual(plan.cta, MessageCTA.YES_NO)
        self.assertIn("Would you like", plan.body)
        self.assertEqual(plan.body.count("?"), 1)

    def test_open_ended_cta_asks_one_question(self):
        store, decision = self.make_decision(
            kind="curious_ask_due",
            payload={"ask_template": "what_service_in_demand_this_week"},
        )
        plan = compose_message(store, decision)
        self.assertEqual(plan.cta, MessageCTA.OPEN_ENDED)
        self.assertIn("What service", plan.body)
        self.assertEqual(plan.body.count("?"), 1)

    def test_confirm_cancel_cta_is_explicit(self):
        store, decision = self.make_decision(
            kind="renewal_due",
            payload={"days_remaining": 5, "plan": "Pro", "renewal_amount": "₹4,999"},
            merchant_fields={"subscription": {"status": "active"}},
        )
        plan = compose_message(store, decision)
        self.assertEqual(plan.cta, MessageCTA.CONFIRM_CANCEL)
        self.assertIn("CONFIRM", plan.body)
        self.assertIn("CANCEL", plan.body)
        self.assertEqual(plan.body.count("?"), 1)

    def test_none_cta_does_not_manufacture_a_question(self):
        store, decision = self.make_decision(
            payload={"metric": "orders", "delta_pct": -8, "window": "last_7d"},
            merchant_fields={"performance": {"orders": 24}},
        )
        plan = compose_message(store, replace(decision, cta=CTASemantic.NONE))
        self.assertEqual(plan.cta, MessageCTA.NONE)
        self.assertEqual(plan.body.count("?"), 0)

    def test_pharmacy_refill_copy_avoids_medical_and_inventory_claims(self):
        store, decision = self.make_decision(
            kind="chronic_refill_due",
            scope="customer",
            category_slug="pharmacies",
            customer={"customer_id": "c1"},
            payload={"molecule_list": ["medicine_x"], "stock_runs_out_iso": "2025-06-20"},
        )
        plan = compose_message(store, decision)
        self.assertEqual(plan.action, DecisionAction.SEND)
        lowered = plan.body.casefold()
        for unsafe_claim in ("dosage", "diagnosis", "treatment recommendation", "in stock", "stock is available"):
            self.assertNotIn(unsafe_claim, lowered)

    def test_merchant_message_uses_only_the_recorded_offer(self):
        store, decision = self.make_decision(
            kind="ipl_match_today",
            payload={
                "match": "Team A vs Team B", "venue": "Ground", "city": "Delhi",
                "match_time_iso": "2025-06-02T19:00:00Z", "is_weeknight": True,
            },
            merchant_fields={"offers": [{"title": "Recorded Meal Offer", "status": "active"}]},
        )
        plan = compose_message(store, decision)
        self.assertEqual(plan.action, DecisionAction.SEND)
        self.assertIn("Recorded Meal Offer", plan.body)
        self.assertNotIn("10% off", plan.body)

    def test_unsupported_intent_and_missing_fact_fail_closed(self):
        store, decision = self.make_decision(
            payload={"metric": "orders", "delta_pct": -8, "window": "last_7d"},
            merchant_fields={"performance": {"orders": 24}},
        )
        unsupported = compose_message(store, replace(decision, intent="invented_intent"))
        self.assertEqual(unsupported.action, DecisionAction.SUPPRESS)
        self.assertIsNone(unsupported.body)
        incomplete = replace(
            decision,
            grounded_facts=tuple(f for f in decision.grounded_facts if f.source != "trigger.payload.delta_pct"),
        )
        missing = compose_message(store, incomplete)
        self.assertEqual(missing.action, DecisionAction.SUPPRESS)
        self.assertIsNone(missing.body)
        fabricated = replace(
            decision,
            grounded_facts=tuple(
                replace(f, value_json="99") if f.source == "trigger.payload.delta_pct" else f
                for f in decision.grounded_facts
            ),
        )
        rejected = compose_message(store, fabricated)
        self.assertEqual(rejected.action, DecisionAction.SUPPRESS)
        self.assertIsNone(rejected.body)

    def test_sparse_grounding_never_fills_missing_claims_with_marketing_copy(self):
        cases = []

        perf_store, perf = self.make_decision(
            payload={"metric": "orders", "delta_pct": -8, "window": "last_7d"},
            merchant_fields={"performance": {"orders": 24}},
        )
        cases.append((perf_store, replace(
            perf,
            grounded_facts=tuple(f for f in perf.grounded_facts if f.source == "trigger.payload.delta_pct"),
        )))

        due_store, due = self.recall_plan()
        cases.append((due_store, replace(
            due,
            grounded_facts=tuple(f for f in due.grounded_facts if f.source == "trigger.payload.due_date"),
        )))

        offer_store, offer = self.make_decision(
            kind="ipl_match_today",
            payload={
                "match": "Team A vs Team B", "venue": "Ground", "city": "Delhi",
                "match_time_iso": "2025-06-02T19:00:00Z", "is_weeknight": True,
            },
            merchant_fields={"offers": [{"title": "Recorded Meal Offer", "status": "active"}]},
        )
        cases.append((offer_store, replace(
            offer,
            grounded_facts=tuple(f for f in offer.grounded_facts if f.source == "merchant.offers[0].title"),
        )))

        for store, sparse_plan in cases:
            with self.subTest(intent=sparse_plan.intent):
                message = compose_message(store, sparse_plan)
                self.assertEqual(message.action, DecisionAction.SUPPRESS)
                self.assertIsNone(message.body)
                self.assertIn("message_grounding_failed", message.reason_code)

    def test_suppression_key_is_stable_and_separates_logical_actions(self):
        first_store, first_decision = self.make_decision(
            payload={"metric": "orders", "delta_pct": -8, "window": "last_7d"},
            merchant_fields={"performance": {"orders": 24}},
        )
        repeat = compose_message(first_store, first_decision)
        second_store, second_decision = self.make_decision(
            merchant_id="m2",
            trigger_id="t2",
            payload={"metric": "orders", "delta_pct": -8, "window": "last_7d"},
            merchant_fields={"performance": {"orders": 24}},
        )
        other_merchant = compose_message(second_store, second_decision)
        self.assertEqual(compose_message(first_store, first_decision).suppression_key, repeat.suppression_key)
        self.assertNotEqual(repeat.suppression_key, other_merchant.suppression_key)
        self.assertNotIn(first_decision.trigger_id, repeat.suppression_key)

    def test_seed_decisions_compose_deterministically(self):
        categories = {
            item["slug"]: item
            for path in (ROOT / "dataset" / "categories").glob("*.json")
            for item in [seed_json(path)]
        }
        merchants = {
            item["merchant_id"]: item
            for item in seed_json(ROOT / "dataset" / "merchants_seed.json")["merchants"]
        }
        customers = {
            item["customer_id"]: item
            for item in seed_json(ROOT / "dataset" / "customers_seed.json")["customers"]
        }
        triggers = seed_json(ROOT / "dataset" / "triggers_seed.json")["triggers"]
        self.assertEqual(len(triggers), 25)

        for raw_trigger in triggers:
            with self.subTest(kind=raw_trigger["kind"]):
                store = ContextStore()
                merchant = merchants[raw_trigger["merchant_id"]]
                category = categories[merchant["category_slug"]]
                store.put("category", category["slug"], 1, category)
                store.put("merchant", merchant["merchant_id"], 1, merchant)
                if raw_trigger.get("customer_id"):
                    customer = customers[raw_trigger["customer_id"]]
                    store.put("customer", customer["customer_id"], 1, customer)
                trigger_id = raw_trigger.get("id", raw_trigger.get("trigger_id"))
                store.put("trigger", trigger_id, 1, raw_trigger)
                decision = decide(store, trigger_id, datetime(2026, 4, 26, 10, 0, tzinfo=timezone.utc))
                first = compose_message(store, decision)
                second = compose_message(store, decision)
                self.assertIsInstance(first, MessagePlan)
                self.assertEqual(first, second)
                self.assertEqual(first.action, DecisionAction.SEND)
                self.assertTrue(first.body)
                self.assertTrue(first.suppression_key)
                self.assertTrue(first.rationale)
                self.assertLessEqual(first.body.count("?"), 1)
                if first.cta != MessageCTA.NONE:
                    self.assertEqual(first.body.count("?"), 1)
                self.assertEqual(
                    first.send_as,
                    SendAs.MERCHANT_ON_BEHALF if decision.customer_id else SendAs.VERA,
                )
                self.assertNotIn("trigger.payload.", first.body)
                self.assertNotIn("category.digest.", first.body)
                json.dumps(asdict(first), ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
