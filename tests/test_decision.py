"""Stage 2 trigger eligibility and deterministic planning tests."""

from datetime import datetime, timezone
import unittest

from src.context_store import ContextStore
from src.decision import CTASemantic, DecisionAction, decide


NOW = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.store = ContextStore()

    def add_merchant(self, category="restaurants", **extra):
        self.store.put("category", category, 1, {"slug": category, "digest": extra.pop("digest", [])})
        payload = {"merchant_id": "m1", "category_slug": category, **extra}
        self.store.put("merchant", "m1", 1, payload)

    def add_customer(self, **extra):
        payload = {
            "customer_id": "c1", "merchant_id": "m1",
            "consent": {"opted_in_at": "2025-01-01", "scope": ["recall_reminders"]},
            **extra,
        }
        self.store.put("customer", "c1", 1, payload)

    def add_trigger(self, kind="research_digest", scope="merchant", payload=None, **extra):
        fields = {
            "trigger_id": "t1", "scope": scope, "kind": kind,
            "merchant_id": "m1", "payload": payload or {}, **extra,
        }
        if scope == "customer":
            fields.setdefault("customer_id", "c1")
        self.store.put("trigger", "t1", 1, fields)

    def test_missing_trigger_returns_suppress_and_missing_id_is_preserved(self):
        plan = decide(self.store, "not-there", NOW)
        self.assertEqual(plan.action, DecisionAction.SUPPRESS)
        self.assertEqual(plan.reason_code, "trigger_not_found")
        self.assertEqual(plan.trigger_id, "not-there")
        self.assertIsNone(decide(self.store, "", NOW).trigger_id)

    def test_unknown_kind_suppresses(self):
        self.add_trigger(kind="unknown")
        plan = decide(self.store, "t1", NOW)
        self.assertEqual((plan.action, plan.reason_code), (DecisionAction.SUPPRESS, "unknown_trigger_kind"))

    def test_expiry_boundary_and_future(self):
        self.add_merchant(digest=[{"id": "d1", "title": "Demand", "source": "Report"}])
        self.add_trigger(payload={"title": "x"}, expires_at="2025-06-01T12:00:00Z")
        self.assertEqual(decide(self.store, "t1", NOW).action, DecisionAction.END)
        self.store.put("trigger", "t1", 2, {
            "trigger_id": "t1", "scope": "merchant", "kind": "research_digest",
            "merchant_id": "m1", "payload": {"top_item_id": "d1"}, "expires_at": "2025-06-01T12:00:01Z",
        })
        self.assertEqual(decide(self.store, "t1", NOW).action, DecisionAction.SEND)
        self.store.put("trigger", "t1", 3, {
            "trigger_id": "t1", "scope": "merchant", "kind": "research_digest",
            "merchant_id": "m1", "payload": {"top_item_id": "d1"}, "expires_at": "2025-06-01T11:59:59Z",
        })
        self.assertEqual(decide(self.store, "t1", NOW).action, DecisionAction.END)

    def test_unparseable_or_timezone_incomparable_expiry_waits(self):
        self.add_trigger(payload={"title": "x"}, expires_at="not-a-date")
        self.assertEqual(decide(self.store, "t1", NOW).action, DecisionAction.WAIT)
        self.store.put("trigger", "t1", 2, {
            "trigger_id": "t1", "scope": "merchant", "kind": "research_digest",
            "merchant_id": "m1", "payload": {}, "expires_at": "2025-06-01T12:00:00",
        })
        self.assertEqual(decide(self.store, "t1", NOW).reason_code, "expiry_timezone_incomparable")

    def test_invalid_now_is_handled_without_using_host_clock(self):
        self.add_trigger(expires_at="2025-06-01T12:00:00Z")
        self.assertEqual(decide(self.store, "t1", None).reason_code, "invalid_now")

    def test_scope_mismatch_suppresses(self):
        self.add_trigger(scope="customer")
        plan = decide(self.store, "t1", NOW)
        self.assertEqual(plan.reason_code, "trigger_scope_mismatch")

    def test_missing_merchant_or_category_suppresses(self):
        self.add_trigger(payload={"metric": "orders", "delta_pct": -10, "window": "week"}, kind="perf_dip")
        self.assertEqual(decide(self.store, "t1", NOW).reason_code, "merchant_relationship_missing")
        self.add_merchant()
        self.store.clear()
        self.add_trigger(kind="perf_dip", payload={"metric": "orders", "delta_pct": -10, "window": "week"})
        self.store.put("merchant", "m1", 1, {"merchant_id": "m1", "category_slug": "missing"})
        self.assertEqual(decide(self.store, "t1", NOW).reason_code, "category_relationship_missing")

    def test_research_digest_requires_exact_category_item(self):
        self.add_merchant(digest=[{"id": "d1", "title": "Demand", "source": "Report"}])
        self.add_trigger(payload={"top_item_id": "d1"})
        plan = decide(self.store, "t1", NOW)
        self.assertEqual(plan.action, DecisionAction.SEND)
        self.assertTrue(all(f.value_json for f in plan.grounded_facts))
        digest_facts = [fact for fact in plan.grounded_facts if fact.source.startswith("category.digest.")]
        self.assertTrue(digest_facts)
        self.assertTrue(all(fact.source.startswith("category.digest.") for fact in digest_facts))
        self.store.put("trigger", "t1", 2, {
            "trigger_id": "t1", "scope": "merchant", "kind": "research_digest",
            "merchant_id": "m1", "payload": {"top_item_id": "invented"},
        })
        self.assertEqual(decide(self.store, "t1", NOW).reason_code, "digest_item_not_grounded")

    def test_research_digest_accepts_documented_embedded_top_item(self):
        self.add_merchant()
        embedded_item = {"title": "Inline research item", "source": "Example Journal"}
        self.add_trigger(payload={"top_item": embedded_item})

        plan = decide(self.store, "t1", NOW)

        # challenge-brief.md's Dr. Meera example supplies top_item inline and
        # does not reference category.digest by ID, so the trigger payload is
        # itself an explicitly documented grounding source for this form.
        self.assertEqual(plan.action, DecisionAction.SEND)
        title_fact = next(fact for fact in plan.grounded_facts if fact.value_json == '"Inline research item"')
        source_fact = next(fact for fact in plan.grounded_facts if fact.value_json == '"Example Journal"')
        self.assertEqual(title_fact.source, "trigger.payload.top_item.title")
        self.assertEqual(source_fact.source, "trigger.payload.top_item.source")
        self.assertNotEqual(title_fact.source, "category.digest.title")
        self.assertNotEqual(source_fact.source, "category.digest.source")

        repeated = decide(self.store, "t1", NOW)
        self.assertEqual(plan, repeated)

    def test_performance_dip_direction_and_evidence(self):
        self.add_merchant(performance={"orders": 12})
        self.add_trigger(kind="perf_dip", payload={"metric": "orders", "delta_pct": -8, "window": "last_7d"})
        plan = decide(self.store, "t1", NOW)
        self.assertEqual(plan.action, DecisionAction.SEND)
        self.assertEqual(plan.cta, CTASemantic.OPEN_ENDED)
        self.store.put("trigger", "t1", 2, {
            "trigger_id": "t1", "scope": "merchant", "kind": "perf_dip", "merchant_id": "m1",
            "payload": {"metric": "orders", "delta_pct": 8, "window": "last_7d"},
        })
        self.assertEqual(decide(self.store, "t1", NOW).reason_code, "performance_direction_mismatch")

    def test_customer_send_requires_relationship_and_matching_consent(self):
        self.add_merchant()
        self.add_customer()
        self.add_trigger(kind="recall_due", scope="customer", payload={"service_due": "cleaning", "due_date": "2025-06-05"})
        plan = decide(self.store, "t1", NOW)
        self.assertEqual(plan.action, DecisionAction.SEND)
        self.assertEqual(plan.customer_id, "c1")
        self.store.put("customer", "c1", 2, {
            "customer_id": "c1", "merchant_id": "m1",
            "consent": {"opted_in_at": "2025-01-01", "scope": ["promotional_offers"]},
        })
        self.assertEqual(decide(self.store, "t1", NOW).reason_code, "customer_consent_missing_or_out_of_scope")

    def test_customer_trigger_referencing_absent_customer_suppresses(self):
        self.add_merchant()
        self.store.put("customer", "c2", 1, {
            "customer_id": "c2", "merchant_id": "m1",
            "consent": {"opted_in_at": "2025-01-01", "scope": ["recall_reminders"]},
        })
        self.add_trigger(kind="recall_due", scope="customer", customer_id="c1",
                         payload={"service_due": "cleaning", "due_date": "2025-06-05"})

        plan = decide(self.store, "t1", NOW)

        self.assertNotEqual(plan.action, DecisionAction.SEND)
        self.assertEqual(plan.reason_code, "customer_relationship_missing")

    def test_customer_from_different_merchant_suppresses(self):
        self.add_merchant()
        self.add_customer(merchant_id="m2")
        self.add_trigger(kind="recall_due", scope="customer", customer_id="c1",
                         payload={"service_due": "cleaning", "due_date": "2025-06-05"})

        plan = decide(self.store, "t1", NOW)

        self.assertNotEqual(plan.action, DecisionAction.SEND)
        self.assertEqual(plan.reason_code, "customer_relationship_missing")

    def test_matching_customer_merchant_and_category_still_send(self):
        self.add_merchant(category="dentists")
        self.add_customer(merchant_id="m1")
        self.add_trigger(kind="recall_due", scope="customer", customer_id="c1",
                         payload={"service_due": "cleaning", "due_date": "2025-06-05"})

        plan = decide(self.store, "t1", NOW)

        self.assertEqual(plan.action, DecisionAction.SEND)
        self.assertEqual(plan.merchant_id, "m1")
        self.assertEqual(plan.customer_id, "c1")

    def test_placeholder_and_sparse_payload_suppress(self):
        self.add_merchant(performance={"orders": 4})
        self.add_trigger(kind="perf_dip", payload={"placeholder": True, "metric": "orders"})
        self.assertEqual(decide(self.store, "t1", NOW).reason_code, "placeholder_or_missing_payload")

    def test_curiosity_cadence_waits_for_seven_days(self):
        self.add_merchant()
        self.add_trigger(kind="curious_ask_due", payload={
            "ask_template": "what_service_in_demand_this_week", "last_ask_at": "2025-05-28T12:00:00Z",
        })
        plan = decide(self.store, "t1", NOW)
        self.assertEqual((plan.action, plan.reason_code), (DecisionAction.WAIT, "weekly_ask_cadence_not_elapsed"))

    def test_deterministic_plan_and_no_context_mutation(self):
        self.add_merchant(performance={"orders": 4})
        self.add_trigger(kind="perf_dip", payload={"metric": "orders", "delta_pct": -2, "window": "week"})
        before = decide(self.store, "t1", NOW)
        retrieved = self.store.get("merchant", "m1")
        retrieved.performance["orders"] = 999
        after = decide(self.store, "t1", NOW)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
