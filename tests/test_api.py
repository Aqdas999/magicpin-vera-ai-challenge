"""Stage 3C HTTP contract tests using Flask's in-process client."""

import json
from pathlib import Path
import unittest

from src.api import MAX_REQUEST_BYTES, create_app


ROOT = Path(__file__).resolve().parents[1]
DELIVERED_AT = "2026-04-26T10:00:00Z"
META = {
    "team_name": "Configured Team",
    "team_members": ["Member One"],
    "model": "Configured Model",
    "approach": "Configured approach",
    "contact_email": "configured@example.test",
    "version": "test-version",
    "submitted_at": "2026-04-26T08:00:00Z",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(metadata=META)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        self.store = self.app.extensions["vera_state"]["context_store"]

    def envelope(self, scope="category", context_id="dentists", version=1, payload=None, delivered_at=DELIVERED_AT):
        if payload is None:
            payload = {"slug": context_id}
        return {
            "scope": scope,
            "context_id": context_id,
            "version": version,
            "payload": payload,
            "delivered_at": delivered_at,
        }

    def post(self, data):
        return self.client.post("/v1/context", json=data)

    def test_health_before_context_is_zero_and_uptime_is_integer(self):
        response = self.client.get("/v1/healthz")
        body = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["status"], "ok")
        self.assertIsInstance(body["uptime_seconds"], int)
        self.assertGreaterEqual(body["uptime_seconds"], 0)
        self.assertEqual(body["contexts_loaded"], {"category": 0, "merchant": 0, "customer": 0, "trigger": 0})

    def test_context_acceptance_ack_and_delivered_at_are_preserved(self):
        response = self.post(self.envelope())
        body = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(body["accepted"])
        self.assertRegex(body["ack_id"], r"^ack_[0-9a-f]{24}$")
        self.assertEqual(body["stored_at"], DELIVERED_AT)
        self.assertEqual(self.client.get("/v1/healthz").get_json()["contexts_loaded"]["category"], 1)

    def test_all_four_store_scopes_are_counted(self):
        entries = [
            self.envelope("category", "dentists", payload={"slug": "dentists"}),
            self.envelope("merchant", "m1", payload={"merchant_id": "m1", "category_slug": "dentists"}),
            self.envelope("customer", "c1", payload={"customer_id": "c1", "merchant_id": "m1"}),
            self.envelope("trigger", "t1", payload={"id": "t1", "scope": "merchant", "merchant_id": "m1"}),
        ]
        for entry in entries:
            with self.subTest(scope=entry["scope"]):
                self.assertEqual(self.post(entry).status_code, 200)
        counts = self.client.get("/v1/healthz").get_json()["contexts_loaded"]
        self.assertEqual(counts, {"category": 1, "merchant": 1, "customer": 1, "trigger": 1})

    def test_higher_version_replaces_without_increasing_count(self):
        self.assertEqual(self.post(self.envelope(payload={"slug": "dentists", "label": "v1"})).status_code, 200)
        self.assertEqual(self.post(self.envelope(version=2, payload={"slug": "dentists", "label": "v2"})).status_code, 200)
        context = self.store.get("category", "dentists")
        self.assertEqual((context.version, context.raw_payload["label"]), (2, "v2"))
        self.assertEqual(self.store.count("category"), 1)

    def test_lower_and_same_versions_return_example_supported_409_and_preserve_value(self):
        self.post(self.envelope(version=2, payload={"slug": "dentists", "marker": "new"}))
        for version in (1, 2):
            with self.subTest(version=version):
                response = self.post(self.envelope(version=version, payload={"slug": "dentists", "marker": "replay"}))
                self.assertEqual(response.status_code, 409)
                self.assertEqual(response.get_json(), {
                    "accepted": False, "reason": "stale_version", "current_version": 2,
                })
        self.assertEqual(self.store.get("category", "dentists").raw_payload["marker"], "new")
        self.assertEqual(self.store.count("category"), 1)

    def test_same_id_in_different_scopes_isolated(self):
        self.assertEqual(self.post(self.envelope("category", "same", payload={"slug": "same"})).status_code, 200)
        self.assertEqual(self.post(self.envelope("merchant", "same", payload={"merchant_id": "same"})).status_code, 200)
        self.assertEqual(self.store.snapshot_counts(), {"category": 1, "merchant": 1, "customer": 0, "trigger": 0})

    def test_invalid_scope_returns_structured_error_and_is_not_stored(self):
        response = self.post(self.envelope(scope="bogus"))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["reason"], "invalid_scope")
        self.assertEqual(sum(self.store.snapshot_counts().values()), 0)

    def test_invalid_context_ids_are_rejected(self):
        for value in ("", "   ", None):
            with self.subTest(value=value):
                data = self.envelope()
                if value is None:
                    data.pop("context_id")
                else:
                    data["context_id"] = value
                response = self.post(data)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.get_json()["reason"], "invalid_context_id")

    def test_invalid_versions_are_rejected(self):
        for value in (-1, "1", True, None):
            with self.subTest(value=value):
                data = self.envelope()
                if value is None:
                    data.pop("version")
                else:
                    data["version"] = value
                response = self.post(data)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.get_json()["reason"], "invalid_version")

    def test_non_object_payloads_are_rejected(self):
        for value in (None, [], "text"):
            with self.subTest(value=value):
                data = self.envelope()
                data["payload"] = value
                response = self.post(data)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.get_json()["reason"], "invalid_payload")

    def test_invalid_or_naive_delivered_at_is_rejected_without_storage(self):
        for value in ("not-a-date", "2026-04-26T10:00:00", None, 42):
            with self.subTest(value=value):
                response = self.post(self.envelope(delivered_at=value))
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.get_json()["reason"], "invalid_delivered_at")
        self.assertEqual(self.store.count("category"), 0)

    def test_payload_over_500_kb_rejected_before_store_mutation(self):
        huge = self.envelope(payload={"slug": "dentists", "large": "x" * MAX_REQUEST_BYTES})
        response = self.post(huge)
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.get_json()["reason"], "payload_too_large")
        self.assertEqual(self.store.count("category"), 0)

    def test_rejected_inputs_do_not_change_health_counts(self):
        self.post(self.envelope())
        self.post(self.envelope(version=0, payload={"slug": "dentists", "marker": "stale"}))
        self.post(self.envelope(scope="invalid"))
        self.assertEqual(self.client.get("/v1/healthz").get_json()["contexts_loaded"]["category"], 1)

    def test_metadata_contains_configured_required_fields(self):
        response = self.client.get("/v1/metadata")
        self.assertEqual(response.status_code, 200)
        decoded = json.loads(response.data.decode("utf-8"))
        self.assertEqual(set(decoded), {
            "team_name", "team_members", "model", "approach", "contact_email", "version", "submitted_at",
        })
        self.assertEqual(decoded, META)

    def test_state_persists_across_requests_and_fresh_app_is_isolated(self):
        self.assertEqual(self.post(self.envelope()).status_code, 200)
        self.assertEqual(self.client.get("/v1/healthz").get_json()["contexts_loaded"]["category"], 1)
        fresh = create_app(metadata=META)
        fresh.config["TESTING"] = True
        self.assertEqual(fresh.test_client().get("/v1/healthz").get_json()["contexts_loaded"]["category"], 0)

    def test_success_and_error_responses_are_json_and_do_not_expose_payload(self):
        accepted = self.post(self.envelope())
        rejected = self.post(self.envelope(scope="invalid", payload={"secret": "not-for-error"}))
        self.assertIn("application/json", accepted.content_type)
        self.assertIn("application/json", rejected.content_type)
        self.assertNotIn("not-for-error", rejected.get_data(as_text=True))
        self.assertNotIn("CategoryContext", accepted.get_data(as_text=True))

    def test_malformed_json_returns_structured_error(self):
        response = self.client.post("/v1/context", data=b"{bad", content_type="application/json")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["reason"], "invalid_json")

    def test_real_seed_contexts_normalize_and_store_through_api(self):
        categories = [read_json(path) for path in (ROOT / "dataset" / "categories").glob("*.json")]
        merchants = read_json(ROOT / "dataset" / "merchants_seed.json")["merchants"]
        customers = read_json(ROOT / "dataset" / "customers_seed.json")["customers"]
        triggers = read_json(ROOT / "dataset" / "triggers_seed.json")["triggers"]
        groups = (
            ("category", [(item["slug"], item) for item in categories]),
            ("merchant", [(item["merchant_id"], item) for item in merchants]),
            ("customer", [(item["customer_id"], item) for item in customers]),
            ("trigger", [(item.get("trigger_id", item.get("id")), item) for item in triggers]),
        )
        for scope, items in groups:
            for context_id, payload in items:
                with self.subTest(scope=scope, context_id=context_id):
                    response = self.post(self.envelope(scope, context_id, payload=payload))
                    self.assertEqual(response.status_code, 200, response.get_json())
                    self.assertTrue(self.store.has(scope, context_id))
        self.assertEqual(self.store.snapshot_counts(), {
            "category": len(categories), "merchant": len(merchants),
            "customer": len(customers), "trigger": len(triggers),
        })

    def test_warmup_contract_flow_reflects_accepted_counts(self):
        self.assertEqual(self.client.get("/v1/healthz").status_code, 200)
        self.assertEqual(self.client.get("/v1/metadata").status_code, 200)
        entries = [
            self.envelope("category", "cat-a", payload={"slug": "cat-a"}),
            self.envelope("category", "cat-b", payload={"slug": "cat-b"}),
            self.envelope("merchant", "m-a", payload={"merchant_id": "m-a"}),
            self.envelope("customer", "c-a", payload={"customer_id": "c-a", "merchant_id": "m-a"}),
        ]
        for entry in entries:
            self.assertEqual(self.post(entry).status_code, 200)
        self.assertEqual(self.client.get("/v1/healthz").get_json()["contexts_loaded"], {
            "category": 2, "merchant": 1, "customer": 1, "trigger": 0,
        })

    def test_reply_route_is_registered_and_validates_required_fields(self):
        response = self.client.post("/v1/reply", json={})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "missing_required_fields")


if __name__ == "__main__":
    unittest.main()
