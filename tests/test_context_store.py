"""Tests for the in-memory versioned context store."""

import unittest

from src.context_store import ContextStore
from src.models import CategoryContext, CustomerContext, MerchantContext, TriggerContext


class ContextStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = ContextStore()

    def test_insert_and_retrieve_each_scope(self):
        records = [
            ("category", "food", {"slug": "food"}, CategoryContext),
            ("merchant", "m1", {"merchant_id": "m1"}, MerchantContext),
            ("customer", "c1", {"customer_id": "c1"}, CustomerContext),
            ("trigger", "t1", {"id": "t1"}, TriggerContext),
        ]
        for scope, context_id, payload, model_type in records:
            with self.subTest(scope=scope):
                self.assertTrue(self.store.put(scope, context_id, 1, payload))
                self.assertIsInstance(self.store.get(scope, context_id), model_type)
                self.assertEqual(self.store.get_version(scope, context_id), 1)
                self.assertTrue(self.store.has(scope, context_id))

    def test_same_id_is_isolated_across_scopes(self):
        self.store.put("merchant", "shared", 1, {"merchant_id": "shared"})
        self.store.put("customer", "shared", 1, {"customer_id": "shared"})
        self.assertIsInstance(self.store.get("merchant", "shared"), MerchantContext)
        self.assertIsInstance(self.store.get("customer", "shared"), CustomerContext)
        self.assertEqual(self.store.snapshot_counts(), {
            "category": 0, "merchant": 1, "customer": 1, "trigger": 0
        })

    def test_get_returns_isolated_context_copy(self):
        self.store.put(
            "merchant",
            "m1",
            1,
            {
                "merchant_id": "m1",
                "offers": [{"name": "Original"}],
                "identity": {"name": "Merchant"},
            },
        )

        first = self.store.get("merchant", "m1")
        first.offers.append({"name": "Injected"})
        first.identity["name"] = "Mutated"

        second = self.store.get("merchant", "m1")

        self.assertEqual(second.offers, [{"name": "Original"}])
        self.assertEqual(second.identity, {"name": "Merchant"})
        self.assertIsNone(self.store.get("merchant", "missing"))

    def test_higher_version_replaces(self):
        self.assertTrue(self.store.put("merchant", "m1", 1, {"merchant_id": "m1", "name": "old"}))
        self.assertTrue(self.store.put("merchant", "m1", 2, {"merchant_id": "m1", "name": "new"}))
        record = self.store.get("merchant", "m1")
        self.assertEqual(record.version, 2)
        self.assertEqual(record.raw_payload["name"], "new")

    def test_lower_version_does_not_replace(self):
        self.store.put("merchant", "m1", 2, {"merchant_id": "m1", "name": "new"})
        self.assertFalse(self.store.put("merchant", "m1", 1, {"merchant_id": "m1", "name": "old"}))
        record = self.store.get("merchant", "m1")
        self.assertEqual((record.version, record.raw_payload["name"]), (2, "new"))

    def test_same_version_is_noop_and_does_not_duplicate(self):
        self.store.put("merchant", "m1", 1, {"merchant_id": "m1", "name": "first"})
        self.assertFalse(self.store.put("merchant", "m1", 1, {"merchant_id": "m1", "name": "replay"}))
        record = self.store.get("merchant", "m1")
        self.assertEqual(record.raw_payload["name"], "first")
        self.assertEqual(self.store.count("merchant"), 1)

    def test_normalized_ids_are_strings(self):
        self.store.put("merchant", 123, 1, {"merchant_id": 123})
        self.assertTrue(self.store.has("merchant", "123"))

    def test_snapshot_counts_and_clear(self):
        self.store.put("category", "cat", 1, {"slug": "cat"})
        self.store.put("merchant", "m", 1, {"merchant_id": "m"})
        counts = self.store.snapshot_counts()
        counts["merchant"] = 99
        self.assertEqual(self.store.count("merchant"), 1)
        self.store.clear()
        self.assertEqual(self.store.snapshot_counts(), {
            "category": 0, "merchant": 0, "customer": 0, "trigger": 0
        })

    def test_list_contexts_is_sorted_and_returns_isolated_copies(self):
        self.store.put("trigger", "t2", 1, {"id": "t2", "payload": {"nested": [2]}})
        self.store.put("trigger", "t1", 1, {"id": "t1", "payload": {"nested": [1]}})
        listed = self.store.list_contexts("trigger")
        self.assertEqual([item.context_id for item in listed], ["t1", "t2"])
        listed[0].payload["nested"].append(99)
        self.assertEqual(self.store.get("trigger", "t1").payload, {"nested": [1]})
        with self.assertRaises(ValueError):
            self.store.list_contexts("invalid")

    def test_invalid_outer_scope_id_and_version_raise_useful_errors(self):
        for args, message in [
            (("bogus", "x", 1, {}), "unsupported context scope"),
            (("merchant", "", 1, {}), "context_id"),
            (("merchant", "x", -1, {}), "version"),
            (("merchant", "x", 1, None), "payload"),
        ]:
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, message):
                self.store.put(*args)

    def test_get_merchant_for_trigger_requires_exact_relationship(self):
        self.store.put("merchant", "m1", 1, {"merchant_id": "m1"})
        self.store.put("trigger", "t1", 1, {"id": "t1", "merchant_id": "m1"})
        self.store.put("trigger", "t2", 1, {"id": "t2", "merchant_id": "missing"})
        self.assertEqual(self.store.get_merchant_for_trigger("t1").context_id, "m1")
        self.assertIsNone(self.store.get_merchant_for_trigger("t2"))

    def test_category_helper_uses_explicit_slug(self):
        self.store.put("category", "salons", 1, {"slug": "salons"})
        self.store.put("merchant", "m1", 1, {"merchant_id": "m1", "category_slug": "salons"})
        self.assertEqual(self.store.get_category_for_merchant("m1").context_id, "salons")
        self.assertIsNone(self.store.get_category_for_merchant("unknown"))

    def test_customer_helpers_require_matching_merchant(self):
        self.store.put("merchant", "m1", 1, {"merchant_id": "m1"})
        self.store.put("merchant", "m2", 1, {"merchant_id": "m2"})
        self.store.put("customer", "c1", 1, {"customer_id": "c1", "merchant_id": "m1"})
        self.store.put("trigger", "t1", 1, {
            "id": "t1", "scope": "customer", "merchant_id": "m1", "customer_id": "c1"
        })
        self.store.put("trigger", "t2", 1, {
            "id": "t2", "scope": "customer", "merchant_id": "m2", "customer_id": "c1"
        })
        self.assertTrue(self.store.validate_customer_merchant_relationship("c1", "m1"))
        self.assertFalse(self.store.validate_customer_merchant_relationship("c1", "m2"))
        self.assertEqual(self.store.get_customer_for_trigger("t1").context_id, "c1")
        self.assertIsNone(self.store.get_customer_for_trigger("t2"))

    def test_trigger_without_customer_id_does_not_invent_customer(self):
        self.store.put("merchant", "m1", 1, {"merchant_id": "m1"})
        self.store.put("trigger", "t1", 1, {"id": "t1", "scope": "merchant", "merchant_id": "m1"})
        self.assertIsNone(self.store.get_customer_for_trigger("t1"))

    def test_trigger_customer_scope_requires_merchant_context_for_customer_join(self):
        self.store.put("customer", "c1", 1, {"customer_id": "c1", "merchant_id": "m1"})
        self.store.put("trigger", "t1", 1, {
            "id": "t1", "scope": "customer", "merchant_id": "m1", "customer_id": "c1"
        })
        self.assertIsNone(self.store.get_customer_for_trigger("t1"))


if __name__ == "__main__":
    unittest.main()
