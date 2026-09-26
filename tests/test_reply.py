"""Stage 3E reply endpoint tests using the real Flask and runtime stores."""

from datetime import datetime, timezone
import json
from pathlib import Path
import unittest

from src.api import create_app
from src.runtime_state import ConversationStatus, OptOutScope, TurnRole


ROOT = Path(__file__).resolve().parents[1]
META = {
    "team_name": "Test", "team_members": [], "model": "", "approach": "test",
    "contact_email": "", "version": "test", "submitted_at": "",
}
NOW = "2026-04-26T10:35:00Z"


class ReplyTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(metadata=META)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        state = self.app.extensions["vera_state"]
        self.store = state["context_store"]
        self.runtime = state["runtime_store"]

    def reply_body(self, **overrides):
        body = {
            "conversation_id": "conv-reply",
            "merchant_id": "m1",
            "customer_id": None,
            "from_role": "merchant",
            "message": "Thanks, I will take a look.",
            "received_at": "2026-04-26T10:42:00Z",
            "turn_number": 2,
        }
        body.update(overrides)
        return body

    def post_reply(self, **overrides):
        return self.client.post("/v1/reply", json=self.reply_body(**overrides))

    def push(self, scope, context_id, payload):
        response = self.client.post("/v1/context", json={
            "scope": scope,
            "context_id": context_id,
            "version": 1,
            "payload": payload,
            "delivered_at": NOW,
        })
        self.assertEqual(response.status_code, 200, response.get_json())

    def create_tick_conversation(self):
        self.push("category", "restaurants", {"slug": "restaurants"})
        self.push("merchant", "m1", {"merchant_id": "m1", "category_slug": "restaurants"})
        self.push("trigger", "t1", {
            "id": "t1", "scope": "merchant", "kind": "curious_ask_due", "merchant_id": "m1",
            "payload": {"ask_template": "what_service_in_demand_this_week"},
            "suppression_key": "tick-key-1",
        })
        response = self.client.post("/v1/tick", json={"now": NOW, "available_triggers": ["t1"]})
        self.assertEqual(response.status_code, 200, response.get_json())
        action = response.get_json()["actions"][0]
        return action["conversation_id"], action

    def seed_jida_research_context(self, *, create_outbound=True):
        category = json.loads((ROOT / "dataset" / "categories" / "dentists.json").read_text(encoding="utf-8"))
        merchant = next(
            item for item in json.loads((ROOT / "dataset" / "merchants_seed.json").read_text(encoding="utf-8"))["merchants"]
            if item["merchant_id"] == "m_001_drmeera_dentist_delhi"
        )
        trigger = next(
            item for item in json.loads((ROOT / "dataset" / "triggers_seed.json").read_text(encoding="utf-8"))["triggers"]
            if item["id"] == "trg_001_research_digest_dentists"
        )
        self.push("category", category["slug"], category)
        self.push("merchant", merchant["merchant_id"], merchant)
        self.push("trigger", trigger["id"], trigger)
        if not create_outbound:
            return trigger["merchant_id"]
        tick = self.client.post("/v1/tick", json={
            "now": NOW, "available_triggers": [trigger["id"]],
        })
        self.assertEqual(tick.status_code, 200, tick.get_json())
        action = tick.get_json()["actions"][0]
        return action["conversation_id"], action

    def test_documented_gst_curveball_redirects_only_with_established_jida_topic(self):
        conversation_id, action = self.seed_jida_research_context()
        response = self.post_reply(
            conversation_id=conversation_id,
            merchant_id=action["merchant_id"],
            message="Btw can you also help me with my GST filing this month?",
            turn_number=2,
        )
        result = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(result["action"], "send")
        self.assertTrue(result["body"].strip())
        self.assertIn("JIDA", result["body"])
        self.assertIn("research", result["body"].casefold())
        for unsupported in ("your CA", "PDF", "schedule"):
            self.assertNotIn(unsupported.casefold(), result["body"].casefold())
        self.assertEqual(result["cta"], "open_ended")
        self.assertEqual(set(result), {"action", "body", "cta", "rationale"})
        state = self.runtime.get_conversation(conversation_id)
        self.assertEqual(state.outbound_history[-1].send_as, "vera")
        self.assertEqual(state.outbound_history[-1].body, result["body"])

    def test_gst_without_prior_jida_outbound_waits_even_when_context_exists(self):
        merchant_id = self.seed_jida_research_context(create_outbound=False)
        response = self.post_reply(
            conversation_id="gst-without-topic",
            merchant_id=merchant_id,
            message="Btw can you also help me with my GST filing this month?",
        )
        result = response.get_json()
        self.assertEqual(result["action"], "wait")
        self.assertEqual(result["wait_seconds"], 1800)
        self.assertNotIn("JIDA", result.get("body", ""))
        self.assertEqual(self.runtime.counts()["outbounds"], 0)

    def test_gst_after_stop_does_not_reopen_jida_conversation(self):
        conversation_id, action = self.seed_jida_research_context()
        stop = self.post_reply(
            conversation_id=conversation_id,
            merchant_id=action["merchant_id"],
            message="Stop messaging me.",
            turn_number=2,
        )
        self.assertEqual(stop.get_json()["action"], "end")
        state_before_gst = self.runtime.get_conversation(conversation_id)
        turns_before_gst = len(state_before_gst.turns)
        outbounds_before_gst = len(state_before_gst.outbound_history)

        after_end = self.post_reply(
            conversation_id=conversation_id,
            merchant_id=action["merchant_id"],
            message="Btw can you also help me with my GST filing this month?",
            turn_number=3,
        )
        self.assertEqual(after_end.get_json()["action"], "end")
        ended = self.runtime.get_conversation(conversation_id)
        self.assertEqual(ended.status, ConversationStatus.ENDED)
        self.assertEqual(len(ended.turns), turns_before_gst)
        self.assertEqual(len(ended.outbound_history), outbounds_before_gst)

    def test_unknown_conversation_initializes_from_supplied_merchant(self):
        response = self.post_reply(conversation_id="sim-conv", merchant_id="merchant-x")
        self.assertEqual(response.status_code, 200)
        state = self.runtime.get_conversation("sim-conv")
        self.assertEqual(state.merchant_id, "merchant-x")
        self.assertIsNone(state.customer_id)
        self.assertEqual(len(state.turns), 1)

    def test_unknown_customer_reply_initializes_with_supplied_customer_relationship(self):
        response = self.client.post("/v1/reply", json=self.reply_body(
            conversation_id="customer-conv",
            merchant_id="merchant-x",
            customer_id="customer-x",
            from_role="customer",
            message="I have a question.",
        ))
        self.assertEqual(response.status_code, 200)
        state = self.runtime.get_conversation("customer-conv")
        self.assertEqual((state.merchant_id, state.customer_id), ("merchant-x", "customer-x"))
        self.assertEqual(state.turns[0].role, TurnRole.CUSTOMER)

    def test_unknown_conversation_without_merchant_is_rejected(self):
        response = self.client.post("/v1/reply", json=self.reply_body(conversation_id="unknown", merchant_id=None))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "invalid_merchant_id")
        self.assertIsNone(self.runtime.get_conversation("unknown"))

    def test_unknown_conversation_missing_merchant_field_is_rejected(self):
        body = self.reply_body(conversation_id="unknown")
        body.pop("merchant_id")
        response = self.client.post("/v1/reply", json=body)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "missing_merchant_id")

    def test_existing_conversation_records_inbound_and_preserves_participants(self):
        self.runtime.create_conversation("conv-reply", "m1", datetime(2026, 4, 26, 10, tzinfo=timezone.utc), "c1")
        response = self.client.post("/v1/reply", json=self.reply_body(customer_id="c1"))
        self.assertEqual(response.status_code, 200)
        state = self.runtime.get_conversation("conv-reply")
        self.assertEqual((state.merchant_id, state.customer_id), ("m1", "c1"))
        self.assertEqual(state.turns[0].role, TurnRole.MERCHANT)
        self.assertEqual(state.turns[0].body, "Thanks, I will take a look.")

    def test_participant_mismatch_is_rejected_without_recording_turn(self):
        self.runtime.create_conversation("conv-reply", "m1", datetime(2026, 4, 26, 10, tzinfo=timezone.utc))
        response = self.post_reply(merchant_id="m2")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.get_json()["error"], "participant_mismatch")
        self.assertEqual(self.runtime.counts()["turns"], 0)

    def test_exact_duplicate_request_records_only_one_inbound_turn(self):
        body = self.reply_body()
        first = self.client.post("/v1/reply", json=body)
        second = self.client.post("/v1/reply", json=body)
        self.assertEqual((first.status_code, second.status_code), (200, 200))
        self.assertEqual(len(self.runtime.get_conversation("conv-reply").turns), 1)

    def test_duplicate_send_intent_does_not_record_or_return_a_second_send(self):
        body = self.reply_body(message="Ok lets do it. Whats next?")
        first = self.client.post("/v1/reply", json=body)
        second = self.client.post("/v1/reply", json=body)
        self.assertEqual(first.get_json()["action"], "send")
        self.assertEqual(second.get_json()["action"], "wait")
        self.assertEqual(self.runtime.counts()["outbounds"], 1)
        self.assertEqual(self.runtime.counts()["turns"], 2)

    def test_distinct_turn_cannot_repeat_outbound_body_in_same_conversation(self):
        body = self.reply_body(message="Ok lets do it. Whats next?")
        first = self.client.post("/v1/reply", json=body)
        first_body = first.get_json()["body"]
        self.assertEqual(first.get_json()["action"], "send")
        self.assertEqual(len(self.runtime.get_conversation("conv-reply").outbound_history), 1)

        body["turn_number"] = 3
        body["received_at"] = "2026-04-26T10:43:00Z"
        second = self.client.post("/v1/reply", json=body)

        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.get_json()["action"], "wait")
        state = self.runtime.get_conversation("conv-reply")
        self.assertEqual([item.body for item in state.outbound_history], [first_body])
        self.assertEqual(self.runtime.counts()["outbounds"], 1)

    def test_same_outbound_body_is_allowed_in_a_different_conversation(self):
        message = "Ok lets do it. Whats next?"
        first = self.client.post("/v1/reply", json=self.reply_body(message=message))
        second = self.client.post("/v1/reply", json=self.reply_body(
            conversation_id="conv-other", message=message,
        ))

        self.assertEqual(first.get_json()["action"], "send")
        self.assertEqual(second.get_json()["action"], "send")
        self.assertEqual(first.get_json()["body"], second.get_json()["body"])
        self.assertEqual(self.runtime.counts()["outbounds"], 2)

    def test_distinct_turn_numbers_preserve_identical_messages_as_separate_turns(self):
        body = self.reply_body(message="Same text")
        self.client.post("/v1/reply", json=body)
        body["turn_number"] = 3
        body["received_at"] = "2026-04-26T10:43:00Z"
        self.client.post("/v1/reply", json=body)
        state = self.runtime.get_conversation("conv-reply")
        self.assertEqual(len(state.turns), 2)
        self.assertEqual([turn.body for turn in state.turns], ["Same text", "Same text"])
        self.assertNotEqual(state.turns[0].turn_id, state.turns[1].turn_id)

    def test_supplied_timestamp_is_preserved_as_aware_datetime(self):
        response = self.post_reply(received_at="2026-04-26T16:12:00+05:30")
        self.assertEqual(response.status_code, 200)
        turn = self.runtime.get_conversation("conv-reply").turns[0]
        self.assertEqual(turn.timestamp.isoformat(), "2026-04-26T10:42:00+00:00")

    def test_naive_timestamp_is_rejected(self):
        response = self.post_reply(received_at="2026-04-26T10:42:00")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "invalid_received_at")
        self.assertIsNone(self.runtime.get_conversation("conv-reply"))

    def test_stop_ends_conversation_and_persists_conversation_scope_opt_out(self):
        response = self.post_reply(message="STOP")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["action"], "end")
        state = self.runtime.get_conversation("conv-reply")
        self.assertEqual(state.status, ConversationStatus.ENDED)
        self.assertTrue(state.contact_blocked)
        self.assertTrue(self.runtime.is_opted_out(
            "m1", OptOutScope.CONVERSATION, conversation_id="conv-reply", at=datetime(2026, 4, 26, 11, tzinfo=timezone.utc),
        ))

    def test_negated_stop_phrase_is_not_misclassified_as_opt_out(self):
        response = self.post_reply(message="Please don't stop messaging me.")
        self.assertEqual(response.get_json()["action"], "wait")
        self.assertEqual(self.runtime.get_conversation("conv-reply").status, ConversationStatus.OPEN)

    def test_stop_on_tick_conversation_prevents_repeated_tick_outbound(self):
        conversation_id, _action = self.create_tick_conversation()
        stop = self.client.post("/v1/reply", json=self.reply_body(
            conversation_id=conversation_id, merchant_id="m1", message="Stop messaging me.", turn_number=2,
        ))
        self.assertEqual(stop.get_json()["action"], "end")
        replay = self.client.post("/v1/tick", json={"now": "2026-04-26T11:00:00Z", "available_triggers": ["t1"]})
        self.assertEqual(replay.get_json(), {"actions": []})
        self.assertEqual(self.runtime.counts()["outbounds"], 1)
        self.assertEqual(self.runtime.get_conversation(conversation_id).status, ConversationStatus.ENDED)

    def test_hostile_stop_phrase_ends_without_reply_copy(self):
        response = self.post_reply(message="Stop messaging me. This is useless spam.")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["action"], "end")
        self.assertNotIn("body", response.get_json())

    def test_canned_auto_reply_returns_deterministic_wait(self):
        response = self.post_reply(message="Thank you for contacting us! Our team will respond shortly.")
        self.assertEqual(response.get_json()["action"], "wait")
        self.assertEqual(response.get_json()["wait_seconds"], 14400)

    def test_documented_named_auto_reply_returns_wait(self):
        response = self.post_reply(message="Thank you for contacting Dr. Meera's Dental Clinic! Our team will respond shortly.")
        self.assertEqual(response.get_json()["action"], "wait")

    def test_third_consecutive_distinct_canned_auto_reply_ends_conversation(self):
        canned = "Thank you for contacting us! Our team will respond shortly."
        responses = []
        for turn_number, minute in ((2, "42"), (3, "43"), (4, "44")):
            responses.append(self.post_reply(
                message=canned,
                turn_number=turn_number,
                received_at=f"2026-04-26T10:{minute}:00Z",
            ).get_json())

        self.assertEqual([item["action"] for item in responses], ["wait", "wait", "end"])
        self.assertEqual(self.runtime.get_conversation("conv-reply").status, ConversationStatus.ENDED)

    def test_exact_duplicate_canned_request_does_not_advance_occurrence_count(self):
        canned = "Thank you for contacting us! Our team will respond shortly."
        request = self.reply_body(message=canned)
        first = self.client.post("/v1/reply", json=request).get_json()
        duplicate = self.client.post("/v1/reply", json=request).get_json()

        self.assertEqual((first["action"], duplicate["action"]), ("wait", "wait"))
        self.assertEqual(len(self.runtime.get_conversation("conv-reply").turns), 1)

        second_occurrence = self.post_reply(
            message=canned, turn_number=3, received_at="2026-04-26T10:43:00Z",
        ).get_json()
        self.assertEqual(second_occurrence["action"], "wait")
        self.assertEqual(self.runtime.get_conversation("conv-reply").status, ConversationStatus.OPEN)

        third_occurrence = self.post_reply(
            message=canned, turn_number=4, received_at="2026-04-26T10:44:00Z",
        ).get_json()
        self.assertEqual(third_occurrence["action"], "end")

    def test_different_inbound_message_resets_consecutive_auto_reply_count(self):
        canned = "Thank you for contacting us! Our team will respond shortly."
        first_two = [
            self.post_reply(
                message=canned,
                turn_number=turn_number,
                received_at=f"2026-04-26T10:{minute}:00Z",
            ).get_json()
            for turn_number, minute in ((2, "42"), (3, "43"))
        ]
        interruption = self.post_reply(
            message="Thanks, I will review this shortly.",
            turn_number=4,
            received_at="2026-04-26T10:44:00Z",
        ).get_json()
        after_interruption = [
            self.post_reply(
                message=canned,
                turn_number=turn_number,
                received_at=f"2026-04-26T10:{minute}:00Z",
            ).get_json()
            for turn_number, minute in ((5, "45"), (6, "46"))
        ]

        self.assertEqual([item["action"] for item in first_two], ["wait", "wait"])
        self.assertEqual(interruption["action"], "wait")
        self.assertEqual([item["action"] for item in after_interruption], ["wait", "wait"])
        self.assertEqual(self.runtime.get_conversation("conv-reply").status, ConversationStatus.OPEN)

    def test_different_canned_text_does_not_count_as_same_auto_reply(self):
        canned_a = "Thank you for contacting Clinic A! Our team will respond shortly."
        canned_b = "Thank you for contacting Clinic B! Our team will respond shortly."
        messages = [canned_a, canned_b, canned_a, canned_a, canned_a]
        responses = [
            self.post_reply(
                message=message,
                turn_number=turn_number,
                received_at=f"2026-04-26T10:{41 + turn_number}:00Z",
            ).get_json()
            for turn_number, message in enumerate(messages, start=2)
        ]

        self.assertEqual(
            [item["action"] for item in responses],
            ["wait", "wait", "wait", "wait", "end"],
        )

    def test_reply_after_third_auto_reply_does_not_reopen_ended_conversation(self):
        canned = "Thank you for contacting us! Our team will respond shortly."
        for turn_number, minute in ((2, "42"), (3, "43"), (4, "44")):
            response = self.post_reply(
                message=canned,
                turn_number=turn_number,
                received_at=f"2026-04-26T10:{minute}:00Z",
            ).get_json()
        self.assertEqual(response["action"], "end")
        ended_state = self.runtime.get_conversation("conv-reply")
        turn_count = len(ended_state.turns)
        outbound_count = self.runtime.counts()["outbounds"]

        after_end = self.post_reply(
            message=canned,
            turn_number=5,
            received_at="2026-04-26T10:45:00Z",
        ).get_json()

        self.assertEqual(after_end["action"], "end")
        self.assertEqual(self.runtime.get_conversation("conv-reply").status, ConversationStatus.ENDED)
        self.assertEqual(len(self.runtime.get_conversation("conv-reply").turns), turn_count)
        self.assertEqual(self.runtime.counts()["outbounds"], outbound_count)

    def test_simulator_auto_reply_conversation_ids_are_initialized(self):
        for conversation_id in ("conv_auto_1", "conv_auto_2", "conv_auto_3", "conv_auto_4"):
            with self.subTest(conversation_id=conversation_id):
                response = self.client.post("/v1/reply", json=self.reply_body(
                    conversation_id=conversation_id,
                    message="Thank you for contacting us! Our team will respond shortly.",
                ))
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json()["action"], "wait")
                self.assertIsNotNone(self.runtime.get_conversation(conversation_id))

    def test_simulator_intent_and_hostile_examples(self):
        intent = self.client.post("/v1/reply", json=self.reply_body(
            conversation_id="conv_intent_1", message="Ok lets do it. Whats next?",
        ))
        hostile = self.client.post("/v1/reply", json=self.reply_body(
            conversation_id="conv_hostile", message="Stop messaging me. This is useless spam.",
        ))
        self.assertEqual(intent.get_json()["action"], "send")
        self.assertTrue(intent.get_json()["body"])
        self.assertEqual(hostile.get_json()["action"], "end")

    def test_explicit_intent_returns_nonempty_action_body_and_outbound_record(self):
        response = self.post_reply(message="Ok lets do it. Whats next?")
        result = response.get_json()
        self.assertEqual(result["action"], "send")
        self.assertTrue(result["body"].strip())
        self.assertEqual(result["cta"], "none")
        self.assertNotIn("?", result["body"])
        self.assertEqual(self.runtime.counts()["outbounds"], 1)

    def test_ordinary_open_ended_reply_waits_without_business_action(self):
        response = self.post_reply(message="Can you tell me more about this?")
        self.assertEqual(response.get_json()["action"], "wait")
        self.assertNotIn("body", response.get_json())
        self.assertEqual(self.runtime.counts()["outbounds"], 0)

    def test_ended_conversation_does_not_reopen_or_add_turn(self):
        self.runtime.create_conversation("ended", "m1", datetime(2026, 4, 26, 10, tzinfo=timezone.utc))
        self.runtime.end_conversation("ended", datetime(2026, 4, 26, 10, 30, tzinfo=timezone.utc))
        response = self.client.post("/v1/reply", json=self.reply_body(conversation_id="ended"))
        self.assertEqual(response.get_json()["action"], "end")
        state = self.runtime.get_conversation("ended")
        self.assertEqual(state.status, ConversationStatus.ENDED)
        self.assertEqual(state.turns, [])

    def test_response_shapes_are_json_serializable(self):
        for message in ("stop", "Thank you for contacting us! Our team will respond shortly.", "Ok lets do it. Whats next?", "Unclassified"):
            with self.subTest(message=message):
                response = self.post_reply(conversation_id="conv-" + str(len(message)), message=message)
                self.assertEqual(response.status_code, 200)
                decoded = json.loads(response.get_data(as_text=True))
                self.assertIn(decoded["action"], {"send", "wait", "end"})
                if decoded["action"] == "send":
                    self.assertEqual(set(decoded), {"action", "body", "cta", "rationale"})
                    self.assertTrue(decoded["body"])
                elif decoded["action"] == "wait":
                    self.assertEqual(set(decoded), {"action", "wait_seconds", "rationale"})
                else:
                    self.assertEqual(set(decoded), {"action", "rationale"})

    def test_runtime_state_persists_across_another_request_and_fresh_app_is_isolated(self):
        self.post_reply(message="Hello")
        health = self.client.get("/v1/healthz")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(self.runtime.counts()["turns"], 1)
        fresh = create_app(metadata=META)
        fresh.config["TESTING"] = True
        self.assertIsNone(fresh.extensions["vera_state"]["runtime_store"].get_conversation("conv-reply"))

    def test_missing_and_invalid_envelope_fields_return_structured_json(self):
        for body, error in (
            ({}, "missing_required_fields"),
            (self.reply_body(from_role="other"), "invalid_from_role"),
            (self.reply_body(message=None), "invalid_message"),
            (self.reply_body(turn_number=True), "invalid_turn_number"),
        ):
            with self.subTest(error=error):
                response = self.client.post("/v1/reply", json=body)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.get_json()["error"], error)

    def test_real_context_tick_reply_flow(self):
        conversation_id, action = self.create_tick_conversation()
        self.assertEqual(self.runtime.counts()["conversations"], 1)
        response = self.client.post("/v1/reply", json=self.reply_body(
            conversation_id=conversation_id,
            merchant_id=action["merchant_id"],
            customer_id=action["customer_id"],
            message="Ok lets do it. Whats next?",
            turn_number=2,
        ))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["action"], "send")
        state = self.runtime.get_conversation(conversation_id)
        self.assertEqual([turn.role for turn in state.turns], [TurnRole.MERCHANT, TurnRole.VERA])
        self.assertEqual(len(state.outbound_history), 2)
        self.assertEqual(self.runtime.counts()["conversations"], 1)

    def test_no_host_clock_value_is_used_for_reply_business_state(self):
        response = self.post_reply(
            message="Ok lets do it. Whats next?",
            received_at="2020-01-02T03:04:05Z",
        )
        self.assertEqual(response.status_code, 200)
        state = self.runtime.get_conversation("conv-reply")
        self.assertEqual(state.turns[0].timestamp.isoformat(), "2020-01-02T03:04:05+00:00")
        self.assertEqual(state.outbound_history[0].timestamp.isoformat(), "2020-01-02T03:04:05+00:00")


if __name__ == "__main__":
    unittest.main()
