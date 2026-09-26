"""Focused tests for the public bot.compose adapter."""

from copy import deepcopy
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import unittest

from bot import compose
from src.context_store import ContextStore
from src.decision import decide
from src.message import compose_message
from src.normalizer import normalize_category, normalize_customer, normalize_merchant, normalize_trigger


ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 4, 26, 10, 0, tzinfo=timezone.utc)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


class BotComposeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.categories = {
            item["slug"]: item
            for path in (ROOT / "dataset" / "categories").glob("*.json")
            for item in [_load(path)]
        }
        cls.merchants = {
            item["merchant_id"]: item
            for item in _load(ROOT / "dataset" / "merchants_seed.json")["merchants"]
        }
        cls.customers = {
            item["customer_id"]: item
            for item in _load(ROOT / "dataset" / "customers_seed.json")["customers"]
        }
        cls.triggers = {
            item["id"]: item
            for item in _load(ROOT / "dataset" / "triggers_seed.json")["triggers"]
        }

    def _inputs(self, trigger_id):
        trigger = self.triggers[trigger_id]
        merchant = self.merchants[trigger["merchant_id"]]
        category = self.categories[merchant["category_slug"]]
        customer = self.customers.get(trigger.get("customer_id"))
        return category, merchant, trigger, customer

    def test_import_and_documented_signature(self):
        self.assertTrue(callable(compose))
        self.assertEqual(
            list(inspect.signature(compose).parameters),
            ["category", "merchant", "trigger", "customer"],
        )
        self.assertIsNone(inspect.signature(compose).parameters["customer"].default)

    def test_send_result_matches_frozen_engine(self):
        category, merchant, trigger, _customer = self._inputs(
            "trg_001_research_digest_dentists"
        )
        store = ContextStore()
        store.put("category", category["slug"], 1, category)
        store.put("merchant", merchant["merchant_id"], 1, merchant)
        store.put("trigger", trigger["id"], 1, trigger)
        decision = decide(store, trigger["id"], NOW)
        expected = compose_message(store, decision)

        actual = compose(category, merchant, trigger)

        self.assertEqual(expected.action.value, "SEND")
        self.assertEqual(
            actual,
            {
                "body": expected.body,
                "cta": expected.cta.value,
                "send_as": expected.send_as.value,
                "suppression_key": expected.suppression_key,
                "rationale": expected.rationale,
            },
        )

    def test_customer_scoped_send_uses_supplied_customer(self):
        category, merchant, trigger, customer = self._inputs("trg_003_recall_due_priya")
        result = compose(category, merchant, trigger, customer)
        self.assertTrue(result["body"])
        self.assertEqual(result["send_as"], "merchant_on_behalf")
        self.assertIn(customer["identity"]["name"], result["body"])

    def test_normalized_context_models_are_accepted(self):
        category, merchant, trigger, customer = self._inputs("trg_003_recall_due_priya")
        result = compose(
            normalize_category(category),
            normalize_merchant(merchant),
            normalize_trigger(trigger),
            normalize_customer(customer),
        )
        self.assertTrue(result["body"])
        self.assertEqual(result["send_as"], "merchant_on_behalf")
    def test_known_suppress_case_stays_non_send(self):
        category, merchant, trigger, _customer = self._inputs(
            "trg_001_research_digest_dentists"
        )
        suppressed = deepcopy(trigger)
        suppressed["id"] = "trg_placeholder_non_send"
        suppressed["payload"] = {"placeholder": True}

        result = compose(category, merchant, suppressed)

        self.assertIsNone(result["body"])
        self.assertEqual(result["cta"], "none")
        self.assertIsNone(result["send_as"])
        self.assertIn("suppress", result["rationale"])

    def test_identical_inputs_are_deterministic(self):
        category, merchant, trigger, _customer = self._inputs(
            "trg_001_research_digest_dentists"
        )
        self.assertEqual(
            compose(category, merchant, trigger),
            compose(category, merchant, trigger),
        )

    def test_adapter_does_not_mutate_inputs(self):
        category, merchant, trigger, customer = self._inputs("trg_003_recall_due_priya")
        before = deepcopy((category, merchant, trigger, customer))
        compose(category, merchant, trigger, customer=customer)
        self.assertEqual((category, merchant, trigger, customer), before)

    def test_invalid_inputs_fail_closed_without_a_message(self):
        result = compose({}, {}, {})
        self.assertIsNone(result["body"])
        self.assertIsNone(result["send_as"])
        self.assertEqual(result["cta"], "none")


if __name__ == "__main__":
    unittest.main()


