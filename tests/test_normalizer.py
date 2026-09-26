"""Tests for flexible normalization and preservation of seed context payloads."""

import json
import unittest
from pathlib import Path

from src.normalizer import normalize_category, normalize_customer, normalize_merchant, normalize_trigger

DATASET = Path(__file__).resolve().parents[1] / "dataset"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


class NormalizerTests(unittest.TestCase):
    def test_missing_optional_merchant_fields_are_absent(self):
        merchant = normalize_merchant({"merchant_id": "m1"})
        self.assertIsNone(merchant.identity)
        self.assertIsNone(merchant.performance)
        self.assertIsNone(merchant.offers)
        self.assertEqual(merchant.scope, "merchant")

    def test_missing_customer_consent_is_not_invented(self):
        customer = normalize_customer({"customer_id": "c1", "merchant_id": "m1"})
        self.assertIsNone(customer.consent)

    def test_missing_offers_remain_missing_and_empty_offers_remain_empty(self):
        self.assertIsNone(normalize_merchant({"merchant_id": "m1"}).offers)
        self.assertEqual(normalize_merchant({"merchant_id": "m2", "offers": []}).offers, [])

    def test_category_catalog_is_not_copied_into_merchant_offers(self):
        category = normalize_category({"slug": "dentists", "offer_catalog": [{"title": "Clean"}]})
        merchant = normalize_merchant({"merchant_id": "m1", "category_slug": category.slug})
        self.assertEqual(len(category.offer_catalog), 1)
        self.assertIsNone(merchant.offers)

    def test_placeholder_trigger_payload_is_preserved_without_inference(self):
        source = {"id": "t1", "payload": {"placeholder": True, "metric_or_topic": "perf_dip"}}
        trigger = normalize_trigger(source)
        self.assertEqual(trigger.payload, source["payload"])
        self.assertTrue(trigger.payload["placeholder"])
        self.assertIsNone(trigger.merchant_id)

    def test_trigger_without_customer_id_stays_without_customer(self):
        trigger = normalize_trigger({"id": "t1", "scope": "merchant", "merchant_id": "m1"})
        self.assertEqual(trigger.scope, "merchant")
        self.assertIsNone(trigger.customer_id)
        self.assertEqual(trigger.context_scope, "trigger")

    def test_expiry_is_preserved_and_parsed_without_host_time(self):
        value = "2026-05-03T00:00:00Z"
        trigger = normalize_trigger({"id": "t1", "expires_at": value})
        self.assertEqual(trigger.expires_at, value)
        self.assertEqual(trigger.expires_at_datetime.isoformat(), "2026-05-03T00:00:00+00:00")

    def test_unparseable_expiry_is_preserved(self):
        value = "not-a-date"
        trigger = normalize_trigger({"id": "t1", "expires_at": value})
        self.assertEqual(trigger.expires_at, value)
        self.assertIsNone(trigger.expires_at_datetime)

    def test_raw_payload_is_deeply_preserved_and_detached(self):
        payload = {"merchant_id": "m1", "unknown": {"nested": [1, 2]}}
        merchant = normalize_merchant(payload)
        self.assertEqual(merchant.raw_payload, payload)
        self.assertIsNot(merchant.raw_payload, payload)
        payload["unknown"]["nested"].append(3)
        self.assertEqual(merchant.raw_payload["unknown"]["nested"], [1, 2])

    def test_unknown_fields_do_not_break_normalization(self):
        merchant = normalize_merchant({"merchant_id": "m1", "future_field": {"x": 1}})
        self.assertEqual(merchant.raw_payload["future_field"], {"x": 1})

    def test_seed_category_merchant_customer_and_trigger_normalize(self):
        categories = [load_json(path) for path in (DATASET / "categories").glob("*.json")]
        merchant_seed = load_json(DATASET / "merchants_seed.json")["merchants"][0]
        customer_seed = load_json(DATASET / "customers_seed.json")["customers"][0]
        trigger_seed = load_json(DATASET / "triggers_seed.json")["triggers"][0]

        self.assertTrue(all(normalize_category(item).slug for item in categories))
        self.assertEqual(normalize_merchant(merchant_seed).merchant_id, merchant_seed["merchant_id"])
        self.assertEqual(normalize_customer(customer_seed).customer_id, customer_seed["customer_id"])
        self.assertEqual(normalize_trigger(trigger_seed).trigger_id, trigger_seed["id"])

    def test_all_seed_records_normalize(self):
        merchants = load_json(DATASET / "merchants_seed.json")["merchants"]
        customers = load_json(DATASET / "customers_seed.json")["customers"]
        triggers = load_json(DATASET / "triggers_seed.json")["triggers"]
        self.assertEqual(len([normalize_merchant(item) for item in merchants]), len(merchants))
        self.assertEqual(len([normalize_customer(item) for item in customers]), len(customers))
        self.assertEqual(len([normalize_trigger(item) for item in triggers]), len(triggers))

    def test_unambiguous_trigger_id_alias(self):
        trigger = normalize_trigger({"trigger_id": 123, "expires_at": None})
        self.assertEqual(trigger.context_id, "123")
        self.assertEqual(trigger.trigger_id, "123")


if __name__ == "__main__":
    unittest.main()
