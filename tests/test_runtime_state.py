"""Stage 3B process-local runtime state tests."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import unittest

from src.context_store import ContextStore
from src.decision import DecisionAction, decide
from src.message import compose_message
from src.runtime_state import (
    ConversationStatus,
    OptOutScope,
    RuntimeStateStore,
    TurnRole,
    body_fingerprint,
)


T0 = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)


class RuntimeStateTests(unittest.TestCase):
    def setUp(self):
        self.store = RuntimeStateStore()

    def create(self, conversation_id="conv1", merchant_id="m1", customer_id=None, at=T0):
        return self.store.create_conversation(conversation_id, merchant_id, at, customer_id)

    def sample_message_plan(self):
        contexts = ContextStore()
        contexts.put("category", "restaurants", 1, {"slug": "restaurants", "voice": {"vocab_taboo": []}})
        contexts.put("merchant", "m1", 1, {
            "merchant_id": "m1", "category_slug": "restaurants",
            "identity": {"name": "Sample Business", "owner_first_name": "Asha", "verified": True},
            "performance": {"orders": 24},
        })
        contexts.put("trigger", "t1", 1, {
            "trigger_id": "t1", "scope": "merchant", "kind": "perf_dip", "merchant_id": "m1",
            "payload": {"metric": "orders", "delta_pct": -8, "window": "last_7d"},
        })
        decision = decide(contexts, "t1", T0)
        self.assertEqual(decision.action, DecisionAction.SEND)
        return compose_message(contexts, decision)

    def test_merchant_conversation_starts_open_and_empty(self):
        state = self.create()
        self.assertEqual(state.conversation_id, "conv1")
        self.assertEqual(state.merchant_id, "m1")
        self.assertIsNone(state.customer_id)
        self.assertEqual(state.status, ConversationStatus.OPEN)
        self.assertEqual(state.turns, [])
        self.assertEqual(state.outbound_history, [])

    def test_customer_conversation_preserves_both_relationship_ids(self):
        state = self.create(customer_id="c1")
        self.assertEqual((state.merchant_id, state.customer_id), ("m1", "c1"))

    def test_same_conversation_id_is_idempotent_and_conflicts_reject(self):
        first = self.create()
        second = self.create(at=T0 + timedelta(days=1))
        self.assertEqual(first, second)
        self.assertEqual(self.store.counts()["conversations"], 1)
        with self.assertRaises(ValueError):
            self.create(merchant_id="m2")
        self.assertEqual(self.store.get_conversation("conv1").merchant_id, "m1")

    def test_turn_order_roles_bodies_and_timestamps_are_preserved(self):
        self.create(customer_id="c1")
        t1 = self.store.add_turn("conv1", "turn1", TurnRole.CUSTOMER, "Hello", T0)
        t2 = self.store.add_turn("conv1", "turn2", TurnRole.VERA, "Hi", T0 + timedelta(minutes=1))
        state = self.store.get_conversation("conv1")
        self.assertEqual(state.turns, [t1, t2])
        self.assertEqual([turn.role for turn in state.turns], [TurnRole.CUSTOMER, TurnRole.VERA])
        self.assertEqual([turn.body for turn in state.turns], ["Hello", "Hi"])
        self.assertEqual(state.turns[1].timestamp, T0 + timedelta(minutes=1))

    def test_turn_id_is_idempotent_and_conflicts_reject(self):
        self.create()
        turn = self.store.add_turn("conv1", "same", "merchant", "Okay", T0)
        self.assertEqual(self.store.add_turn("conv1", "same", "merchant", "Okay", T0), turn)
        self.assertEqual(len(self.store.get_conversation("conv1").turns), 1)
        with self.assertRaises(ValueError):
            self.store.add_turn("conv1", "same", "merchant", "Different", T0)

    def test_inbound_rejects_vera_role(self):
        self.create()
        with self.assertRaises(ValueError):
            self.store.record_inbound("conv1", "t1", "vera", "outbound", T0)

    def test_outbound_first_detection_and_message_fields(self):
        self.create()
        plan = self.sample_message_plan()
        self.assertFalse(self.store.has_outbound("conv1"))
        self.assertTrue(self.store.is_first_outbound("conv1"))
        record = self.store.record_outbound("conv1", plan, T0, event_id="out1")
        self.assertTrue(self.store.has_outbound("conv1"))
        self.assertFalse(self.store.is_first_outbound("conv1"))
        self.assertEqual(record.body, plan.body)
        self.assertEqual(record.send_as, plan.send_as.value)
        self.assertEqual(record.suppression_key, plan.suppression_key)
        self.assertEqual(record.timestamp, T0)
        self.assertEqual(record.body_fingerprint, body_fingerprint(plan.body))
        self.assertEqual(record.action, "send")
        self.assertEqual(record.trigger_id, "t1")
        state = self.store.get_conversation("conv1")
        self.assertEqual(state.turns[0].role, TurnRole.VERA)
        self.assertEqual(state.turns[0].turn_id, "out1")

    def test_outbound_query_rejects_unknown_conversation(self):
        with self.assertRaises(KeyError):
            self.store.has_outbound("missing")
        with self.assertRaises(KeyError):
            self.store.is_first_outbound("missing")

    def test_outbound_event_idempotency_and_conflict(self):
        self.create()
        plan = self.sample_message_plan()
        first = self.store.record_outbound("conv1", plan, T0, event_id="event1")
        self.assertEqual(self.store.record_outbound("conv1", plan, T0, event_id="event1"), first)
        self.assertEqual(self.store.counts()["outbounds"], 1)
        with self.assertRaises(ValueError):
            self.store.record_outbound("conv1", replace(plan, body=plan.body + " Changed"), T0, event_id="event1")

    def test_body_fingerprint_uses_only_body(self):
        digest = hashlib.sha256("same body".encode("utf-8")).hexdigest()
        self.assertEqual(body_fingerprint("same body"), digest)
        self.assertEqual(body_fingerprint("same body"), body_fingerprint("same body"))
        self.assertNotEqual(body_fingerprint("same body"), body_fingerprint("different body"))

    def test_outbound_body_lookup_is_exact_and_conversation_scoped(self):
        self.create("conv1")
        self.create("conv2")
        plan = self.sample_message_plan()
        self.store.record_outbound("conv1", plan, T0)

        self.assertTrue(self.store.has_outbound_body("conv1", plan.body))
        self.assertFalse(self.store.has_outbound_body("conv1", plan.body + " Changed"))
        self.assertFalse(self.store.has_outbound_body("conv2", plan.body))

    def test_suppression_is_idempotent_and_keys_are_independent(self):
        first = self.store.record_suppression("opaque:a", T0)
        self.assertEqual(self.store.record_suppression("opaque:a", T0 + timedelta(days=1)), first)
        self.store.record_suppression("opaque:b", T0)
        self.assertTrue(self.store.has_suppression("opaque:a"))
        self.assertTrue(self.store.has_suppression("opaque:b"))
        self.assertFalse(self.store.has_suppression("opaque:c"))
        self.assertEqual(self.store.counts()["suppressions"], 2)

    def test_recording_outbound_persists_its_suppression_key(self):
        self.create()
        plan = self.sample_message_plan()
        self.store.record_outbound("conv1", plan, T0)
        self.assertTrue(self.store.has_suppression(plan.suppression_key))

    def test_opt_out_persists_without_text_parsing_and_scopes_are_distinct(self):
        self.create(customer_id="c1")
        self.store.record_opt_out("m1", OptOutScope.CONVERSATION, T0, customer_id="c1", conversation_id="conv1")
        self.assertTrue(self.store.is_opted_out("m1", "conversation", customer_id="c1", conversation_id="conv1"))
        self.assertFalse(self.store.is_opted_out("m1", "customer_merchant", customer_id="c1"))
        self.assertFalse(self.store.is_opted_out("m1", "merchant"))
        self.assertTrue(self.store.get_conversation("conv1").contact_blocked)
        self.assertTrue(self.store.clear_opt_out("m1", "conversation", customer_id="c1", conversation_id="conv1"))
        self.assertFalse(self.store.get_conversation("conv1").contact_blocked)

    def test_customer_merchant_opt_out_isolated_by_both_ids_and_persistent(self):
        self.store.record_opt_out("m1", "customer_merchant", T0, customer_id="c1", expires_at=T0 + timedelta(days=2))
        self.assertTrue(self.store.is_opted_out("m1", "customer_merchant", customer_id="c1"))
        self.assertTrue(self.store.is_opted_out("m1", "customer_merchant", customer_id="c1", at=T0 + timedelta(days=1)))
        self.assertFalse(self.store.is_opted_out("m1", "customer_merchant", customer_id="c1", at=T0 + timedelta(days=2)))
        self.assertFalse(self.store.is_opted_out("m2", "customer_merchant", customer_id="c1"))
        self.assertFalse(self.store.is_opted_out("m1", "customer_merchant", customer_id="c2"))

    def test_get_and_snapshot_are_detached_including_nested_history(self):
        self.create()
        self.store.add_turn("conv1", "turn1", "merchant", "Original", T0)
        state = self.store.get_conversation("conv1")
        state.turns.append(state.turns[0])
        state.outbound_history.append(None)
        state.contact_blocked = True
        again = self.store.get_conversation("conv1")
        self.assertEqual(len(again.turns), 1)
        self.assertEqual(again.outbound_history, [])
        self.assertFalse(again.contact_blocked)
        snapshot = self.store.snapshot()
        snapshot["conversations"]["conv1"]["turns"].clear()
        self.assertEqual(len(self.store.get_conversation("conv1").turns), 1)

    def test_invalid_naive_timestamps_are_rejected_without_host_clock(self):
        with self.assertRaises(ValueError):
            self.create(at=datetime(2025, 6, 1))
        self.create()
        with self.assertRaises(ValueError):
            self.store.add_turn("conv1", "turn1", "merchant", "x", datetime(2025, 6, 1))

    def test_conversation_can_be_ended_only_explicitly(self):
        self.create()
        self.assertEqual(self.store.get_conversation("conv1").status, ConversationStatus.OPEN)
        ended = self.store.end_conversation("conv1", T0 + timedelta(days=400))
        self.assertEqual(ended.status, ConversationStatus.ENDED)
        with self.assertRaises(ValueError):
            self.store.record_outbound("conv1", self.sample_message_plan(), T0, event_id="late")

    def test_deterministic_operation_sequence_and_snapshot(self):
        outputs = []
        for _ in range(2):
            store = RuntimeStateStore()
            store.create_conversation("cv", "m", T0, "c")
            store.add_turn("cv", "t", "customer", "Hi", T0)
            store.record_suppression("k", T0)
            store.record_opt_out("m", "customer_merchant", T0, customer_id="c")
            outputs.append(store.snapshot())
        self.assertEqual(outputs[0], outputs[1])

    def test_clear_resets_every_state_collection(self):
        self.create(customer_id="c1")
        self.store.record_suppression("k", T0)
        self.store.record_opt_out("m1", "customer_merchant", T0, customer_id="c1")
        self.store.clear()
        self.assertIsNone(self.store.get_conversation("conv1"))
        self.assertFalse(self.store.has_suppression("k"))
        self.assertFalse(self.store.is_opted_out("m1", "customer_merchant", customer_id="c1"))
        self.assertEqual(self.store.counts(), {"conversations": 0, "turns": 0, "outbounds": 0, "suppressions": 0, "opt_outs": 0})

    def test_concurrent_duplicate_turn_insert_is_atomic(self):
        self.create()
        def insert():
            return self.store.add_turn("conv1", "same", "merchant", "one", T0)
        with ThreadPoolExecutor(max_workers=8) as pool:
            turns = list(pool.map(lambda _: insert(), range(32)))
        self.assertTrue(all(turn == turns[0] for turn in turns))
        self.assertEqual(len(self.store.get_conversation("conv1").turns), 1)

    def test_real_stage3a_message_plan_is_consumed_without_mutation(self):
        self.create()
        plan = self.sample_message_plan()
        before = plan
        stored = self.store.record_outbound("conv1", plan, T0, event_id="send1")
        self.assertEqual(plan, before)
        self.assertEqual(stored.body, plan.body)
        self.assertEqual(self.store.get_conversation("conv1").outbound_history, [stored])


if __name__ == "__main__":
    unittest.main()
