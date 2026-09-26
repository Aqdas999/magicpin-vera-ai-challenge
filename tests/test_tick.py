"""Stage 3D tick orchestration tests over the real Flask app and frozen stages."""

from datetime import datetime, timezone
import json
from pathlib import Path
import unittest

from src.api import MAX_ACTIONS_PER_TICK, create_app
from src.decision import DecisionAction, decide
from src.message import compose_message
from src.runtime_state import ConversationStatus


ROOT = Path(__file__).resolve().parents[1]
NOW_TEXT = "2026-04-26T10:35:00Z"
NOW = datetime(2026, 4, 26, 10, 35, tzinfo=timezone.utc)
META = {
    "team_name": "Test", "team_members": [], "model": "", "approach": "test",
    "contact_email": "", "version": "test", "submitted_at": "",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


class TickTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(metadata=META)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        self.store = self.app.extensions["vera_state"]["context_store"]
        self.runtime = self.app.extensions["vera_state"]["runtime_store"]

    def push(self, scope, context_id, payload, version=1):
        response = self.client.post("/v1/context", json={
            "scope": scope, "context_id": context_id, "version": version,
            "payload": payload, "delivered_at": NOW_TEXT,
        })
        self.assertIn(response.status_code, (200,), response.get_json())
        return response.get_json()

    def tick(self, trigger_ids, now=NOW_TEXT):
        return self.client.post("/v1/tick", json={"now": now, "available_triggers": trigger_ids})

    def seed_base(self):
        categories = {
            item["slug"]: item
            for path in (ROOT / "dataset" / "categories").glob("*.json")
            for item in [read_json(path)]
        }
        merchants = {item["merchant_id"]: item for item in read_json(ROOT / "dataset" / "merchants_seed.json")["merchants"]}
        customers = {item["customer_id"]: item for item in read_json(ROOT / "dataset" / "customers_seed.json")["customers"]}
        triggers = read_json(ROOT / "dataset" / "triggers_seed.json")["triggers"]
        for category in categories.values():
            self.push("category", category["slug"], category)
        for merchant in merchants.values():
            self.push("merchant", merchant["merchant_id"], merchant)
        for customer in customers.values():
            self.push("customer", customer["customer_id"], customer)
        trigger_ids = []
        for trigger in triggers:
            trigger_id = trigger.get("trigger_id", trigger.get("id"))
            trigger_ids.append(trigger_id)
            self.push("trigger", trigger_id, trigger)
        return categories, merchants, customers, triggers, trigger_ids

    def test_empty_tick_returns_exact_documented_response(self):
        response = self.tick([])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"actions": []})

    def test_single_valid_trigger_uses_frozen_decision_and_composer(self):
        categories, merchants, _customers, triggers, trigger_ids = self.seed_base()
        selected = next(item for item in triggers if item.get("kind") == "research_digest")
        trigger_id = selected.get("trigger_id", selected.get("id"))
        decision = decide(self.store, trigger_id, NOW)
        expected = compose_message(self.store, decision)
        self.assertEqual(decision.action, DecisionAction.SEND)
        response = self.tick([trigger_id])
        self.assertEqual(response.status_code, 200)
        actions = response.get_json()["actions"]
        action = next(item for item in actions if item["trigger_id"] == expected.trigger_id)
        self.assertEqual(action["body"], expected.body)
        self.assertEqual(action["send_as"], expected.send_as.value)
        self.assertEqual(action["cta"], expected.cta.value)
        self.assertEqual(action["suppression_key"], expected.suppression_key)
        self.assertEqual(action["rationale"], expected.rationale)

    def test_repeated_tick_is_suppressed_and_outbound_recorded_once(self):
        self.push("category", "restaurants", {"slug": "restaurants"})
        self.push("merchant", "m1", {"merchant_id": "m1", "category_slug": "restaurants"})
        self.push("trigger", "t1", {
            "id": "t1", "scope": "merchant", "kind": "curious_ask_due", "merchant_id": "m1",
            "payload": {"ask_template": "what_service_in_demand_this_week"},
        })
        first = self.tick(["t1"]).get_json()
        second = self.tick(["t1"]).get_json()
        self.assertEqual(len(first["actions"]), 1)
        self.assertEqual(second, {"actions": []})
        self.assertEqual(self.runtime.counts()["outbounds"], 1)

    def test_two_valid_triggers_with_distinct_keys_both_send(self):
        self.push("category", "restaurants", {"slug": "restaurants", "digest": [{"id": "d", "title": "Headline", "source": "Journal"}]})
        self.push("merchant", "m1", {"merchant_id": "m1", "category_slug": "restaurants", "identity": {"owner_first_name": "A"}})
        trigger_ids = []
        for tid, title, key in (("t1", "Headline", "key-1"), ("t2", "Headline", "key-2")):
            trigger_ids.append(tid)
            self.push("trigger", tid, {
                "id": tid, "scope": "merchant", "kind": "research_digest", "merchant_id": "m1",
                "payload": {"top_item": {"title": title, "source": "Journal"}}, "suppression_key": key,
            })
        response = self.tick(trigger_ids)
        self.assertEqual(response.status_code, 200)
        self.assertEqual({item["trigger_id"] for item in response.get_json()["actions"]}, set(trigger_ids))
        self.assertEqual(self.runtime.counts()["outbounds"], 2)

    def test_same_suppression_key_emits_only_one_action_in_tick(self):
        self.push("category", "restaurants", {"slug": "restaurants"})
        self.push("merchant", "m1", {"merchant_id": "m1", "category_slug": "restaurants"})
        for tid in ("t1", "t2"):
            self.push("trigger", tid, {
                "id": tid, "scope": "merchant", "kind": "curious_ask_due", "merchant_id": "m1",
                "payload": {"ask_template": "what_service_in_demand_this_week"}, "suppression_key": "shared",
            })
        result = self.tick(["t2", "t1"]).get_json()["actions"]
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["trigger_id"], "t1")
        self.assertEqual(self.runtime.counts()["outbounds"], 1)

    def test_wait_end_and_suppress_never_become_send(self):
        self.push("trigger", "wait", {
            "id": "wait", "scope": "merchant", "kind": "research_digest", "expires_at": "not-a-date",
        })
        self.push("trigger", "end", {
            "id": "end", "scope": "merchant", "kind": "research_digest", "expires_at": "2026-04-26T10:34:59Z",
        })
        self.push("trigger", "suppress", {"id": "suppress", "scope": "merchant", "kind": "unknown"})
        plans = {tid: decide(self.store, tid, NOW).action for tid in ("wait", "end", "suppress")}
        self.assertEqual(plans, {"wait": DecisionAction.WAIT, "end": DecisionAction.END, "suppress": DecisionAction.SUPPRESS})
        self.assertEqual(self.tick(["wait", "end", "suppress"]).get_json(), {"actions": []})

    def test_missing_merchant_relationship_and_unknown_trigger_are_skipped(self):
        self.push("trigger", "missing-merchant", {
            "id": "missing-merchant", "scope": "merchant", "kind": "curious_ask_due", "merchant_id": "absent",
            "payload": {"ask_template": "what_service_in_demand_this_week"},
        })
        self.assertEqual(self.tick(["missing-merchant", "not-stored"]).get_json(), {"actions": []})

    def test_customer_consent_version_update_is_respected(self):
        self.push("category", "dentists", {"slug": "dentists"})
        self.push("merchant", "m1", {"merchant_id": "m1", "category_slug": "dentists"})
        self.push("customer", "c1", {
            "customer_id": "c1", "merchant_id": "m1", "consent": {"opted_in_at": "2025-01-01", "scope": ["recall_reminders"]},
        })
        trigger = {
            "id": "t1", "scope": "customer", "kind": "recall_due", "merchant_id": "m1", "customer_id": "c1",
            "payload": {"service_due": "cleaning", "due_date": "2026-05-10"},
        }
        self.push("trigger", "t1", trigger)
        first = self.tick(["t1"])
        self.assertEqual(len(first.get_json()["actions"]), 1)
        self.push("customer", "c1", {
            "customer_id": "c1", "merchant_id": "m1", "consent": {"opted_in_at": "2025-01-01", "scope": []},
        }, version=2)
        self.assertEqual(decide(self.store, "t1", NOW).action, DecisionAction.SUPPRESS)
        self.assertEqual(self.tick(["t1"]).get_json(), {"actions": []})

    def test_wrong_or_missing_customer_never_sends(self):
        self.push("category", "dentists", {"slug": "dentists"})
        self.push("merchant", "m1", {"merchant_id": "m1", "category_slug": "dentists"})
        self.push("merchant", "m2", {"merchant_id": "m2", "category_slug": "dentists"})
        self.push("customer", "c1", {
            "customer_id": "c1", "merchant_id": "m2", "consent": {"opted_in_at": "2025-01-01", "scope": ["recall_reminders"]},
        })
        for tid, cid in (("wrong", "c1"), ("absent", "missing")):
            self.push("trigger", tid, {
                "id": tid, "scope": "customer", "kind": "recall_due", "merchant_id": "m1", "customer_id": cid,
                "payload": {"service_due": "cleaning", "due_date": "2026-05-10"},
            })
        self.assertEqual(self.tick(["wrong", "absent"]).get_json(), {"actions": []})

    def test_action_order_is_stable_and_ids_are_deterministic(self):
        self.push("category", "restaurants", {"slug": "restaurants"})
        self.push("merchant", "m1", {"merchant_id": "m1", "category_slug": "restaurants"})
        for tid in ("z-trigger", "a-trigger"):
            self.push("trigger", tid, {
                "id": tid, "scope": "merchant", "kind": "curious_ask_due", "merchant_id": "m1",
                "payload": {"ask_template": "what_service_in_demand_this_week"}, "suppression_key": f"key-{tid}",
            })
        # Different logical keys are required for independent sends.
        first_app = self.tick(["z-trigger", "a-trigger"]).get_json()["actions"]
        self.assertEqual([a["trigger_id"] for a in first_app], ["a-trigger", "z-trigger"])
        self.assertEqual(len({a["conversation_id"] for a in first_app}), 2)

        equivalent = TickTests()
        equivalent.setUp()
        equivalent.push("category", "restaurants", {"slug": "restaurants"})
        equivalent.push("merchant", "m1", {"merchant_id": "m1", "category_slug": "restaurants"})
        for tid in ("z-trigger", "a-trigger"):
            equivalent.push("trigger", tid, {
                "id": tid, "scope": "merchant", "kind": "curious_ask_due", "merchant_id": "m1",
                "payload": {"ask_template": "what_service_in_demand_this_week"}, "suppression_key": f"key-{tid}",
            })
        equivalent_actions = equivalent.tick(["a-trigger", "z-trigger"]).get_json()["actions"]
        self.assertEqual(
            [item["conversation_id"] for item in first_app],
            [item["conversation_id"] for item in equivalent_actions],
        )

    def test_explicit_tick_time_controls_expiry_without_host_clock(self):
        self.push("trigger", "t1", {
            "id": "t1", "scope": "merchant", "kind": "research_digest", "expires_at": "2026-04-27T00:00:00Z",
        })
        self.assertEqual(self.tick(["t1"], now="2026-04-27T00:00:00Z").get_json(), {"actions": []})
        self.assertEqual(decide(self.store, "t1", NOW).action, DecisionAction.SUPPRESS)
        after_expiry = datetime(2026, 4, 27, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(decide(self.store, "t1", after_expiry).action, DecisionAction.END)

    def test_new_trigger_version_with_new_key_can_send_again(self):
        self.push("category", "restaurants", {"slug": "restaurants"})
        self.push("merchant", "m1", {"merchant_id": "m1", "category_slug": "restaurants"})
        def trigger(title, key):
            return {
                "id": "t1", "scope": "merchant", "kind": "research_digest", "merchant_id": "m1",
                "payload": {"top_item": {"title": title, "source": "Journal"}}, "suppression_key": key,
            }
        self.push("trigger", "t1", trigger("First", "key-v1"), version=1)
        first = self.tick(["t1"]).get_json()["actions"]
        self.assertEqual(len(first), 1)
        first_body = first[0]["body"]
        self.push("trigger", "t1", trigger("Updated", "key-v2"), version=2)
        second = self.tick(["t1"]).get_json()["actions"]
        self.assertEqual(len(second), 1)
        self.assertNotEqual(first_body, second[0]["body"])
        self.assertNotEqual(first[0]["conversation_id"], second[0]["conversation_id"])
        self.assertIn("Updated", second[0]["body"])
        self.assertEqual(self.runtime.counts()["outbounds"], 2)

    def test_updated_trigger_with_same_suppression_key_stays_suppressed(self):
        self.push("category", "restaurants", {"slug": "restaurants"})
        self.push("merchant", "m1", {"merchant_id": "m1", "category_slug": "restaurants"})
        self.push("trigger", "t1", {
            "id": "t1", "scope": "merchant", "kind": "research_digest", "merchant_id": "m1",
            "payload": {"top_item": {"title": "First", "source": "Journal"}},
            "suppression_key": "stable-key",
        }, version=1)
        first = self.tick(["t1"]).get_json()["actions"]
        self.assertEqual(len(first), 1)

        self.push("trigger", "t1", {
            "id": "t1", "scope": "merchant", "kind": "research_digest", "merchant_id": "m1",
            "payload": {"top_item": {"title": "Updated", "source": "Journal"}},
            "suppression_key": "stable-key",
        }, version=2)
        self.assertEqual(self.tick(["t1"]).get_json(), {"actions": []})
        self.assertEqual(self.runtime.counts()["outbounds"], 1)

    def test_updated_trigger_does_not_reopen_after_stop(self):
        self.push("category", "restaurants", {"slug": "restaurants"})
        self.push("merchant", "m1", {"merchant_id": "m1", "category_slug": "restaurants"})
        self.push("trigger", "t1", {
            "id": "t1", "scope": "merchant", "kind": "research_digest", "merchant_id": "m1",
            "payload": {"top_item": {"title": "First", "source": "Journal"}},
            "suppression_key": "key-v1",
        }, version=1)
        first = self.tick(["t1"]).get_json()["actions"]
        self.assertEqual(len(first), 1)
        stop = self.client.post("/v1/reply", json={
            "conversation_id": first[0]["conversation_id"],
            "merchant_id": "m1",
            "from_role": "merchant",
            "message": "Stop messaging me.",
            "received_at": "2026-04-26T10:42:00Z",
            "turn_number": 2,
        })
        self.assertEqual(stop.get_json()["action"], "end")

        self.push("trigger", "t1", {
            "id": "t1", "scope": "merchant", "kind": "research_digest", "merchant_id": "m1",
            "payload": {"top_item": {"title": "Updated", "source": "Journal"}},
            "suppression_key": "key-v2",
        }, version=2)
        self.assertEqual(self.tick(["t1"]).get_json(), {"actions": []})
        self.assertEqual(self.runtime.counts()["outbounds"], 1)

    def test_category_update_without_new_trigger_version_does_not_resend_same_event(self):
        self.push("category", "dentists", {
            "slug": "dentists",
            "digest": [{"id": "item", "kind": "research", "title": "Original", "source": "Journal"}],
        })
        self.push("merchant", "m1", {"merchant_id": "m1", "category_slug": "dentists"})
        self.push("trigger", "t1", {
            "id": "t1", "scope": "merchant", "kind": "research_digest", "merchant_id": "m1",
            "payload": {"category": "dentists", "top_item_id": "item"}, "suppression_key": "key-v1",
        })
        first = self.tick(["t1"]).get_json()["actions"]
        self.assertEqual(len(first), 1)
        self.push("category", "dentists", {
            "slug": "dentists",
            "digest": [{"id": "item", "kind": "research", "title": "Updated", "source": "Journal"}],
        }, version=2)
        self.assertEqual(self.tick(["t1"]).get_json(), {"actions": []})
        self.assertEqual(self.runtime.counts()["outbounds"], 1)

    def test_same_trigger_version_never_reuses_conversation_after_related_context_changes(self):
        self.push("category", "dentists", {
            "slug": "dentists",
            "digest": [{"id": "item", "kind": "research", "title": "Original", "source": "Journal"}],
        })
        self.push("merchant", "m1", {"merchant_id": "m1", "category_slug": "dentists"})
        self.push("trigger", "t1", {
            "id": "t1", "scope": "merchant", "kind": "research_digest", "merchant_id": "m1",
            "payload": {"category": "dentists", "top_item_id": "item"}, "suppression_key": "key-v1",
        })
        first = self.tick(["t1"]).get_json()["actions"]
        self.assertEqual(len(first), 1)
        self.push("category", "dentists", {
            "slug": "dentists",
            "digest": [{"id": "item", "kind": "research", "title": "Updated", "source": "Journal"}],
        }, version=2)
        # The trigger version and key are unchanged, so this is not a new tick event.
        self.assertEqual(self.tick(["t1"]).get_json(), {"actions": []})
        self.assertEqual(self.runtime.counts()["outbounds"], 1)

    def test_tick_skips_same_body_for_new_key_without_blocking_other_candidate(self):
        self.push("category", "restaurants", {"slug": "restaurants"})
        self.push("merchant", "m1", {"merchant_id": "m1", "category_slug": "restaurants"})
        payload = {"ask_template": "what_service_in_demand_this_week"}
        self.push("trigger", "t1", {
            "id": "t1", "scope": "merchant", "kind": "curious_ask_due", "merchant_id": "m1",
            "payload": payload, "suppression_key": "key-v1",
        })
        first = self.tick(["t1"]).get_json()["actions"]
        self.assertEqual(len(first), 1)
        duplicate_body = first[0]["body"]

        self.push("trigger", "t1", {
            "id": "t1", "scope": "merchant", "kind": "curious_ask_due", "merchant_id": "m1",
            "payload": payload, "suppression_key": "key-v2",
        }, version=2)
        self.push("trigger", "t2", {
            "id": "t2", "scope": "merchant", "kind": "curious_ask_due", "merchant_id": "m1",
            "payload": payload, "suppression_key": "key-t2",
        })

        actions = self.tick(["t1", "t2"]).get_json()["actions"]

        self.assertEqual([item["trigger_id"] for item in actions], ["t2"])
        self.assertEqual(actions[0]["body"], duplicate_body)
        self.assertEqual(self.runtime.counts()["outbounds"], 2)
        first_conversation = self.runtime.get_conversation(first[0]["conversation_id"])
        second_conversation = self.runtime.get_conversation(actions[0]["conversation_id"])
        self.assertEqual([item.body for item in first_conversation.outbound_history], [duplicate_body])
        self.assertEqual([item.body for item in second_conversation.outbound_history], [duplicate_body])

    def test_tick_state_persists_across_requests_and_fresh_app_isolated(self):
        self.push("category", "restaurants", {"slug": "restaurants"})
        self.push("merchant", "m1", {"merchant_id": "m1", "category_slug": "restaurants"})
        self.push("trigger", "t1", {
            "id": "t1", "scope": "merchant", "kind": "curious_ask_due", "merchant_id": "m1",
            "payload": {"ask_template": "what_service_in_demand_this_week"},
        })
        self.assertEqual(len(self.tick(["t1"]).get_json()["actions"]), 1)
        self.assertEqual(self.tick(["t1"]).get_json(), {"actions": []})
        self.assertEqual(self.runtime.counts()["outbounds"], 1)
        fresh = create_app(metadata=META)
        fresh.config["TESTING"] = True
        response = fresh.test_client().post("/v1/tick", json={"now": NOW_TEXT, "available_triggers": ["t1"]})
        self.assertEqual(response.get_json(), {"actions": []})
        self.assertEqual(fresh.extensions["vera_state"]["runtime_store"].counts()["outbounds"], 0)

    def test_seed_trigger_replay_matches_frozen_plans_and_composer(self):
        _categories, _merchants, _customers, triggers, trigger_ids = self.seed_base()
        expected = {}
        for trigger in triggers:
            tid = trigger.get("trigger_id", trigger.get("id"))
            plan = decide(self.store, tid, NOW)
            if plan.action == DecisionAction.SEND:
                composed = compose_message(self.store, plan)
                if composed.action == DecisionAction.SEND:
                    expected[composed.trigger_id] = composed
        self.assertEqual(len(triggers), 25)
        result_actions = []
        # Challenge/simulator examples batch trigger IDs; this also exercises the 20-action cap.
        for start in range(0, len(trigger_ids), 5):
            response = self.tick(trigger_ids[start:start + 5])
            self.assertEqual(response.status_code, 200)
            result_actions.extend(response.get_json()["actions"])
        actual = {action["trigger_id"]: action for action in result_actions}
        self.assertEqual(set(actual), set(expected))
        for trigger_id, plan in expected.items():
            self.assertEqual(actual[trigger_id]["body"], plan.body)
            self.assertEqual(actual[trigger_id]["send_as"], plan.send_as.value)
            self.assertEqual(actual[trigger_id]["cta"], plan.cta.value)
            self.assertEqual(actual[trigger_id]["suppression_key"], plan.suppression_key)
        self.assertTrue(all(len(self.tick(trigger_ids).get_json()["actions"]) <= MAX_ACTIONS_PER_TICK for _ in range(1)))

        equivalent = TickTests()
        equivalent.setUp()
        _cats, _merchants, _people, _triggers, equivalent_ids = equivalent.seed_base()
        equivalent_actions = []
        for start in range(0, len(equivalent_ids), 5):
            equivalent_actions.extend(equivalent.tick(equivalent_ids[start:start + 5]).get_json()["actions"])
        self.assertEqual(result_actions, equivalent_actions)

    def test_action_cap_leaves_remaining_candidates_for_later_tick(self):
        self.push("category", "restaurants", {"slug": "restaurants"})
        self.push("merchant", "m1", {"merchant_id": "m1", "category_slug": "restaurants"})
        ids = []
        for index in range(MAX_ACTIONS_PER_TICK + 1):
            tid = f"t{index:02d}"
            ids.append(tid)
            self.push("trigger", tid, {
                "id": tid, "scope": "merchant", "kind": "curious_ask_due", "merchant_id": "m1",
                "payload": {"ask_template": "what_service_in_demand_this_week"}, "suppression_key": f"key-{tid}",
            })
        first = self.tick(ids).get_json()["actions"]
        second = self.tick(ids).get_json()["actions"]
        self.assertEqual(len(first), MAX_ACTIONS_PER_TICK)
        self.assertEqual(len(second), 1)
        self.assertEqual(first[0]["trigger_id"], "t00")
        self.assertEqual(second[0]["trigger_id"], "t20")

    def test_actions_are_json_contract_shapes(self):
        self.push("category", "restaurants", {"slug": "restaurants"})
        self.push("merchant", "m1", {"merchant_id": "m1", "category_slug": "restaurants"})
        self.push("trigger", "t1", {
            "id": "t1", "scope": "merchant", "kind": "curious_ask_due", "merchant_id": "m1",
            "payload": {"ask_template": "what_service_in_demand_this_week"},
        })
        response = self.tick(["t1"])
        decoded = json.loads(response.data.decode("utf-8"))
        self.assertEqual(set(decoded), {"actions"})
        action = decoded["actions"][0]
        self.assertEqual(set(action), {
            "conversation_id", "merchant_id", "customer_id", "send_as", "trigger_id",
            "template_name", "template_params", "body", "cta", "suppression_key", "rationale",
        })
        self.assertEqual(action["template_params"], [action["body"]])
        self.assertEqual(action["template_name"], "vera_generic_v1")

    def test_malformed_tick_requests_return_structured_errors(self):
        cases = [
            (b"{bad", 400, "invalid_json"),
            (json.dumps({"now": "not-a-date", "available_triggers": []}).encode(), 400, "invalid_now"),
            (json.dumps({"now": "2026-04-26T10:35:00", "available_triggers": []}).encode(), 400, "invalid_now"),
            (json.dumps({"now": NOW_TEXT, "available_triggers": "t1"}).encode(), 400, "invalid_available_triggers"),
            (b'{"now":"2026-04-26T10:35:00Z","available_triggers":[],"x":NaN}', 400, "invalid_json"),
        ]
        for raw, status, reason in cases:
            with self.subTest(reason=reason, raw=raw):
                response = self.client.post("/v1/tick", data=raw, content_type="application/json")
                self.assertEqual(response.status_code, status)
                self.assertEqual(response.get_json()["error"], reason)


if __name__ == "__main__":
    unittest.main()
