"""Replay supplied seed triggers against only their explicitly linked contexts."""

from datetime import datetime, timezone
import json
from pathlib import Path
import unittest

from src.context_store import ContextStore
from src.decision import DecisionAction, DecisionPlan, decide


ROOT = Path(__file__).resolve().parents[1]
REPLAY_NOW = datetime(2026, 4, 26, 10, 0, tzinfo=timezone.utc)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _subtree_json_values(value):
    """Return canonical JSON for every subtree in a JSON-compatible object."""
    found = {json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))}
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, list):
        children = value
    else:
        return found
    for child in children:
        found.update(_subtree_json_values(child))
    return found


class SeedReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.categories = {
            item["slug"]: item
            for path in (ROOT / "dataset" / "categories").glob("*.json")
            for item in [_read_json(path)]
        }
        cls.merchants = {
            item["merchant_id"]: item
            for item in _read_json(ROOT / "dataset" / "merchants_seed.json")["merchants"]
        }
        cls.customers = {
            item["customer_id"]: item
            for item in _read_json(ROOT / "dataset" / "customers_seed.json")["customers"]
        }
        cls.triggers = _read_json(ROOT / "dataset" / "triggers_seed.json")["triggers"]

    def make_store(self, trigger):
        store = ContextStore()
        merchant = self.merchants.get(trigger.get("merchant_id"))
        if merchant is not None:
            category = self.categories.get(merchant.get("category_slug"))
            if category is not None:
                store.put("category", category["slug"], 1, category)
            store.put("merchant", merchant["merchant_id"], 1, merchant)
        customer = self.customers.get(trigger.get("customer_id"))
        if customer is not None:
            store.put("customer", customer["customer_id"], 1, customer)
        trigger_id = trigger.get("trigger_id", trigger.get("id"))
        store.put("trigger", trigger_id, 1, trigger)
        return store, trigger_id

    def test_every_seed_trigger_replays_deterministically_with_grounded_facts(self):
        self.assertTrue(self.triggers, "the supplied trigger seed should not be empty")
        for seed_trigger in self.triggers:
            with self.subTest(trigger=seed_trigger.get("id", seed_trigger.get("trigger_id"))):
                store, trigger_id = self.make_store(seed_trigger)
                plan = decide(store, trigger_id, REPLAY_NOW)
                repeated = decide(store, trigger_id, REPLAY_NOW)
                self.assertIsInstance(plan, DecisionPlan)
                self.assertIn(plan.action, set(DecisionAction))
                self.assertEqual(plan, repeated)
                self.assertEqual(plan.trigger_id, trigger_id)
                self.assertIn(plan.action, {
                    DecisionAction.SEND, DecisionAction.WAIT,
                    DecisionAction.END, DecisionAction.SUPPRESS,
                })

                raw_by_scope = {
                    "trigger": seed_trigger,
                    "merchant": self.merchants.get(seed_trigger.get("merchant_id")),
                }
                merchant = raw_by_scope["merchant"]
                if merchant is not None:
                    raw_by_scope["category"] = self.categories.get(merchant.get("category_slug"))
                customer = self.customers.get(seed_trigger.get("customer_id"))
                if customer is not None:
                    raw_by_scope["customer"] = customer
                for fact in plan.grounded_facts:
                    scope = fact.source.split(".", 1)[0]
                    self.assertIn(scope, raw_by_scope, fact.source)
                    self.assertIsNotNone(raw_by_scope[scope], fact.source)
                    actual_values = _subtree_json_values(raw_by_scope[scope])
                    self.assertIn(fact.value_json, actual_values, f"ungrounded fact {fact.source}")

                if plan.action == DecisionAction.SEND and seed_trigger.get("scope") == "customer":
                    self.assertIsNotNone(customer)
                    consent = customer.get("consent") or {}
                    self.assertTrue(consent.get("opted_in_at"))
                    required_scope = {
                        "recall_due": "recall_reminders",
                        "customer_lapsed_soft": "recall_reminders",
                        "wedding_package_followup": "bridal_package_followup",
                        "customer_lapsed_hard": "winback_offers",
                        "trial_followup": "kids_program_updates",
                        "chronic_refill_due": "refill_reminders",
                        "appointment_tomorrow": "appointment_reminders",
                        "unplanned_slot_open": "promotional_offers",
                    }.get(seed_trigger.get("kind"))
                    self.assertIsNotNone(required_scope)
                    self.assertIn(required_scope, consent.get("scope", []))

    def test_placeholder_payload_cannot_become_actionable_for_seed_kinds(self):
        for seed_trigger in self.triggers:
            with self.subTest(trigger=seed_trigger.get("id", seed_trigger.get("trigger_id"))):
                store, trigger_id = self.make_store(seed_trigger)
                placeholder_trigger = dict(seed_trigger)
                placeholder_trigger["payload"] = {"placeholder": True}
                store.put("trigger", trigger_id, 2, placeholder_trigger)
                plan = decide(store, trigger_id, datetime(2000, 1, 1, tzinfo=timezone.utc))
                self.assertEqual(plan.action, DecisionAction.SUPPRESS)
                self.assertEqual(plan.reason_code, "placeholder_or_missing_payload")


if __name__ == "__main__":
    unittest.main()
