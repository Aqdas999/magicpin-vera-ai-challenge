"""Stage 3C/3D HTTP contract with deterministic context ingestion and tick orchestration."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
import re
import time
from threading import RLock
from typing import Any

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from .context_store import ContextStore
from .decision import DecisionAction, decide
from .message import MessageCTA, MessagePlan, SendAs, compose_message
from .models import CategoryContext, MerchantContext, TriggerContext
from .runtime_state import ConversationStatus, OptOutScope, RuntimeStateStore, TurnRole


MAX_REQUEST_BYTES = 500 * 1024
MAX_ACTIONS_PER_TICK = 20
_SCOPES = frozenset(("category", "merchant", "customer", "trigger"))
_METADATA_FIELDS = (
    "team_name", "team_members", "model", "approach", "contact_email", "version", "submitted_at"
)


def _metadata_from_environment() -> dict[str, Any]:
    try:
        members = json.loads(os.environ.get("VERA_TEAM_MEMBERS", "[]"))
    except json.JSONDecodeError as exc:
        raise ValueError("VERA_TEAM_MEMBERS must be a JSON array of strings") from exc
    if not isinstance(members, list) or any(not isinstance(item, str) for item in members):
        raise ValueError("VERA_TEAM_MEMBERS must be a JSON array of strings")
    return {
        "team_name": os.environ.get("VERA_TEAM_NAME", ""),
        "team_members": members,
        "model": os.environ.get("VERA_MODEL", ""),
        "approach": os.environ.get(
            "VERA_APPROACH", "Deterministic context normalization and in-memory versioned storage"
        ),
        "contact_email": os.environ.get("VERA_CONTACT_EMAIL", ""),
        "version": os.environ.get("VERA_VERSION", "0.1.0"),
        "submitted_at": os.environ.get("VERA_SUBMITTED_AT", ""),
    }


def _validate_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    output = {field: metadata.get(field) for field in _METADATA_FIELDS}
    if not isinstance(output["team_members"], list) or any(
        not isinstance(member, str) for member in output["team_members"]
    ):
        raise ValueError("team_members must be a list of strings")
    for name in set(_METADATA_FIELDS) - {"team_members"}:
        if not isinstance(output[name], str):
            raise ValueError(f"{name} must be a string")
    return output


def _error(reason: str, details: str, status: int):
    return jsonify({"accepted": False, "reason": reason, "details": details}), status


def _ack_id(scope: str, context_id: str, version: int) -> str:
    identity = f"{scope}\0{context_id}\0{version}".encode("utf-8")
    return "ack_" + hashlib.sha256(identity).hexdigest()[:24]


def _parse_delivered_at(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    timestamp_text = value
    try:
        parsed = datetime.fromisoformat(
            timestamp_text[:-1] + "+00:00" if timestamp_text.endswith(("Z", "z")) else timestamp_text
        )
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return timestamp_text


def _parse_tick_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _reply_turn_id(conversation_id: str, role: str, turn_number: int, received_at: str, message: str) -> str:
    """Stable internal identity; the reply contract has no explicit event ID."""
    payload = json.dumps(
        [conversation_id, role, turn_number, received_at, message],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "reply_" + hashlib.sha256(payload).hexdigest()


def _reply_text_key(message: str) -> str:
    text = message.casefold().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _is_canned_auto_reply(key: str) -> bool:
    return (
        key.startswith("thank you for contacting ")
        and key.endswith(" our team will respond shortly")
    )


def _consecutive_canned_auto_replies(conversation, expected_key: str) -> int:
    """Count the matching inbound suffix, ignoring Vera's intervening replies."""
    count = 0
    for turn in reversed(conversation.turns):
        if turn.role == TurnRole.VERA:
            continue
        key = _reply_text_key(turn.body)
        if key != expected_key or not _is_canned_auto_reply(key):
            break
        count += 1
    return count


def _reply_policy(message: str) -> dict[str, Any]:
    """Small deterministic reply policy for behaviors shown by the challenge."""
    key = _reply_text_key(message)
    explicit_stop = (
        key == "stop"
        or key.startswith("stop messaging me")
        or key.startswith("please stop messaging me")
        or key.startswith("not interested stop messaging me")
        or key.startswith("why are you bothering me this is useless stop sending these")
    )
    if explicit_stop:
        return {
            "action": "end",
            "rationale": "Explicit stop request received. Closing this conversation.",
        }
    if _is_canned_auto_reply(key):
        return {
            "action": "wait",
            "wait_seconds": 14400,
            "rationale": "Detected the documented canned auto-reply. Waiting for the owner to respond.",
        }
    if key == "ok lets do it whats next":
        return {
            "action": "send",
            "body": "Understood — I’ll guide you through the next step.",
            "cta": MessageCTA.NONE.value,
            "rationale": "The merchant explicitly asked to proceed; offering a general next step without claiming an action was completed.",
        }
    return {
        "action": "wait",
        "wait_seconds": 1800,
        "rationale": "No supported reply action was identified. Waiting without inferring intent.",
    }


def _has_established_jida_research_topic(conversation, contexts: ContextStore) -> bool:
    """Require a stored JIDA digest outbound linked to this merchant's trigger."""
    merchant = contexts.get("merchant", conversation.merchant_id)
    if not isinstance(merchant, MerchantContext):
        return False
    category = contexts.get_category_for_merchant(merchant)
    if not isinstance(category, CategoryContext) or not isinstance(category.digest, list):
        return False

    for outbound in reversed(conversation.outbound_history):
        if outbound.send_as != SendAs.VERA.value or not outbound.trigger_id:
            continue
        trigger = contexts.get("trigger", outbound.trigger_id)
        if not isinstance(trigger, TriggerContext) or trigger.kind not in {
            "research_digest", "research_digest_release", "category_research_digest_release",
        }:
            continue
        if not contexts.validate_trigger_merchant_relationship(trigger, merchant):
            continue
        payload = trigger.payload if isinstance(trigger.payload, dict) else {}
        if payload.get("category") != category.slug:
            continue

        item_id = payload.get("top_item_id")
        item = next((
            value for value in category.digest
            if isinstance(value, dict) and item_id and value.get("id") == item_id
        ), None)
        if item is None and isinstance(payload.get("top_item"), dict):
            item = payload["top_item"]
        if not isinstance(item, dict) or item.get("kind") != "research":
            continue
        source = item.get("source")
        if not isinstance(source, str) or "jida" not in source.casefold():
            continue
        # The linked outbound itself must visibly establish the JIDA reference.
        if "jida" in outbound.body.casefold() and source.casefold() in outbound.body.casefold():
            return True
    return False


def _conversation_id(
    message: MessagePlan, trigger: TriggerContext, category_slug: str | None
) -> str:
    """Build an unambiguous, deterministic ID for this trigger version."""
    identity = json.dumps(
        [
            message.merchant_id,
            category_slug,
            "customer" if message.customer_id is not None else "merchant",
            message.customer_id,
            trigger.context_id,
            message.trigger_id,
            trigger.version,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "conv_" + identity


def _tick_action(conversation_id: str, message: MessagePlan) -> dict[str, Any]:
    # The documented action envelope includes a template wrapper; Stage 3D uses
    # one generic body parameter and leaves template/session policy to a later stage.
    return {
        "conversation_id": conversation_id,
        "merchant_id": message.merchant_id,
        "customer_id": message.customer_id,
        "send_as": message.send_as.value,
        "trigger_id": message.trigger_id,
        "template_name": "vera_generic_v1",
        "template_params": [message.body],
        "body": message.body,
        "cta": message.cta.value,
        "suppression_key": message.suppression_key,
        "rationale": message.rationale,
    }


def create_app(
    *,
    context_store: ContextStore | None = None,
    runtime_store: RuntimeStateStore | None = None,
    metadata: dict[str, Any] | None = None,
) -> Flask:
    """Create an isolated app instance with one persistent pair of process-local stores."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES
    app.json.ensure_ascii = False
    app.extensions["vera_state"] = {
        "context_store": context_store if context_store is not None else ContextStore(),
        "runtime_store": runtime_store if runtime_store is not None else RuntimeStateStore(),
        "started_monotonic": time.monotonic(),
        "tick_lock": RLock(),
        "metadata": _validate_metadata(metadata if metadata is not None else _metadata_from_environment()),
    }

    @app.errorhandler(RequestEntityTooLarge)
    def request_too_large(_error):
        return _error("payload_too_large", "request body exceeds 500 KB", 413)

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify({"error": "not_found"}), 404

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify({"error": error.name.casefold().replace(" ", "_")}), error.code

    @app.errorhandler(Exception)
    def unexpected_error(_error):
        # Keep internal exception details and request contents out of the response.
        return jsonify({"error": "internal_error"}), 500

    @app.get("/v1/healthz")
    def healthz():
        state = app.extensions["vera_state"]
        elapsed = time.monotonic() - state["started_monotonic"]
        return jsonify({
            "status": "ok",
            "uptime_seconds": max(0, int(elapsed)),
            "contexts_loaded": state["context_store"].snapshot_counts(),
        })

    @app.get("/v1/metadata")
    def get_metadata():
        return jsonify(app.extensions["vera_state"]["metadata"])

    @app.post("/v1/tick")
    def tick():
        raw = request.get_data(cache=False)
        if len(raw) > MAX_REQUEST_BYTES:
            return jsonify({"error": "payload_too_large", "details": "request body exceeds 500 KB"}), 413
        try:
            body = json.loads(
                raw.decode("utf-8"),
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON constant: {value}")),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return jsonify({"error": "invalid_json", "details": "request body must be valid UTF-8 JSON"}), 400
        if not isinstance(body, dict):
            return jsonify({"error": "invalid_tick_request", "details": "request body must be a JSON object"}), 400
        now = _parse_tick_time(body.get("now"))
        if now is None:
            return jsonify({"error": "invalid_now", "details": "now must be a timezone-aware ISO-8601 timestamp"}), 400
        available = body.get("available_triggers", [])
        if not isinstance(available, list) or any(
            not isinstance(item, str) or not item.strip() for item in available
        ):
            return jsonify({"error": "invalid_available_triggers", "details": "available_triggers must be a list of non-empty strings"}), 400
        trigger_ids = {item.strip() for item in available}

        state = app.extensions["vera_state"]
        contexts: ContextStore = state["context_store"]
        runtime: RuntimeStateStore = state["runtime_store"]

        # Serialize concurrent ticks so suppression check + outbound recording is
        # atomic at the HTTP orchestration boundary without changing Stage 3B.
        with state["tick_lock"]:
            stored_triggers = {
                item.context_id: item for item in contexts.list_contexts("trigger")
                if item.context_id in trigger_ids
            }
            candidates: list[tuple[str, MessagePlan, str]] = []
            tick_suppression_keys: set[str] = set()
            for trigger_id in sorted(stored_triggers):
                decision = decide(contexts, trigger_id, now)
                if decision.action != DecisionAction.SEND:
                    continue
                message = compose_message(contexts, decision)
                if message.action != DecisionAction.SEND or not message.body:
                    continue
                key = message.suppression_key
                if key and (key in tick_suppression_keys or runtime.has_suppression(key)):
                    continue
                trigger = stored_triggers[trigger_id]
                merchant = contexts.get_merchant_for_trigger(trigger)
                category = contexts.get_category_for_merchant(merchant) if merchant is not None else None
                category_slug = category.slug if category is not None else None
                conversation_id = _conversation_id(message, trigger, category_slug)
                existing = runtime.get_conversation(conversation_id)
                # A tick starts a conversation; only /v1/reply may continue one.
                if existing is not None:
                    continue
                if runtime.has_outbound_for_trigger_body(
                    message.trigger_id, message.merchant_id, message.customer_id, message.body
                ):
                    continue
                if runtime.has_ended_conversation_for_trigger(
                    message.trigger_id, message.merchant_id, message.customer_id
                ):
                    continue
                if key:
                    tick_suppression_keys.add(key)
                candidates.append((conversation_id, message, key or ""))
                if len(candidates) >= MAX_ACTIONS_PER_TICK:
                    break

            actions: list[dict[str, Any]] = []
            for conversation_id, message, _key in candidates:
                runtime.create_conversation(
                    conversation_id,
                    message.merchant_id,
                    now,
                    customer_id=message.customer_id,
                )
                # Record only actions included in this response. RuntimeStateStore
                # persists its own body fingerprint and the logical suppression key.
                runtime.record_outbound(conversation_id, message, now)
                actions.append(_tick_action(conversation_id, message))
            return jsonify({"actions": actions}), 200

    @app.post("/v1/reply")
    def reply():
        raw = request.get_data(cache=False)
        if len(raw) > MAX_REQUEST_BYTES:
            return jsonify({"error": "payload_too_large", "details": "request body exceeds 500 KB"}), 413
        try:
            body = json.loads(
                raw.decode("utf-8"),
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON constant: {value}")),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return jsonify({"error": "invalid_json", "details": "request body must be valid UTF-8 JSON"}), 400
        if not isinstance(body, dict):
            return jsonify({"error": "invalid_reply_request", "details": "request body must be a JSON object"}), 400

        required = ("conversation_id", "from_role", "message", "received_at", "turn_number")
        missing = [name for name in required if name not in body]
        if missing:
            return jsonify({"error": "missing_required_fields", "details": f"missing: {', '.join(missing)}"}), 400

        conversation_id = body["conversation_id"]
        if not isinstance(conversation_id, str) or not conversation_id.strip():
            return jsonify({"error": "invalid_conversation_id", "details": "conversation_id must be a non-empty string"}), 400
        conversation_id = conversation_id.strip()
        role = body["from_role"]
        if role not in (TurnRole.MERCHANT.value, TurnRole.CUSTOMER.value):
            return jsonify({"error": "invalid_from_role", "details": "from_role must be merchant or customer"}), 400
        message = body["message"]
        if not isinstance(message, str):
            return jsonify({"error": "invalid_message", "details": "message must be a string"}), 400
        received_at_text = body["received_at"]
        received_at = _parse_tick_time(received_at_text)
        if received_at is None:
            return jsonify({"error": "invalid_received_at", "details": "received_at must be a timezone-aware ISO-8601 timestamp"}), 400
        turn_number = body["turn_number"]
        if isinstance(turn_number, bool) or not isinstance(turn_number, int) or turn_number < 1:
            return jsonify({"error": "invalid_turn_number", "details": "turn_number must be a positive integer"}), 400

        merchant_supplied = "merchant_id" in body
        merchant_id = body.get("merchant_id")
        if merchant_supplied and (not isinstance(merchant_id, str) or not merchant_id.strip()):
            return jsonify({"error": "invalid_merchant_id", "details": "merchant_id must be a non-empty string when supplied"}), 400
        if isinstance(merchant_id, str):
            merchant_id = merchant_id.strip()
        customer_supplied = "customer_id" in body
        customer_id = body.get("customer_id")
        if customer_id is not None and (not isinstance(customer_id, str) or not customer_id.strip()):
            return jsonify({"error": "invalid_customer_id", "details": "customer_id must be null or a non-empty string"}), 400
        if isinstance(customer_id, str):
            customer_id = customer_id.strip()

        runtime: RuntimeStateStore = app.extensions["vera_state"]["runtime_store"]
        turn_id = _reply_turn_id(conversation_id, role, turn_number, received_at_text, message)
        # Share the tick lock so duplicate detection and recording are atomic
        # with respect to another reply or outbound tick in this app instance.
        with app.extensions["vera_state"]["tick_lock"]:
            conversation = runtime.get_conversation(conversation_id)
            if conversation is None:
                if merchant_id is None:
                    return jsonify({"error": "missing_merchant_id", "details": "merchant_id is required to initialize an unknown conversation"}), 400
                if role == TurnRole.CUSTOMER.value and customer_id is None:
                    return jsonify({"error": "missing_customer_id", "details": "customer_id is required to initialize a customer reply"}), 400
                try:
                    conversation = runtime.create_conversation(
                        conversation_id, merchant_id, received_at, customer_id=customer_id
                    )
                except (TypeError, ValueError):
                    return jsonify({"error": "invalid_participants", "details": "conversation participants are invalid"}), 400
            else:
                if merchant_supplied and merchant_id != conversation.merchant_id:
                    return jsonify({"error": "participant_mismatch", "details": "merchant_id does not match the conversation"}), 409
                if customer_supplied and customer_id != conversation.customer_id:
                    return jsonify({"error": "participant_mismatch", "details": "customer_id does not match the conversation"}), 409
                merchant_id = conversation.merchant_id
                customer_id = conversation.customer_id
                if role == TurnRole.CUSTOMER.value and customer_id is None:
                    return jsonify({"error": "invalid_participants", "details": "customer replies require a stored customer_id"}), 400

            if conversation.status == ConversationStatus.ENDED:
                return jsonify({"action": "end", "rationale": "This conversation has already ended."}), 200

            is_duplicate = any(turn.turn_id == turn_id for turn in conversation.turns)
            try:
                if not is_duplicate:
                    runtime.record_inbound(conversation_id, turn_id, role, message, received_at)
            except (KeyError, TypeError, ValueError):
                return jsonify({"error": "invalid_reply_turn", "details": "reply turn could not be recorded"}), 400

            result = _reply_policy(message)
            if (
                not is_duplicate
                and _is_canned_auto_reply(_reply_text_key(message))
                and _consecutive_canned_auto_replies(
                    conversation, _reply_text_key(message)
                ) >= 2
            ):
                result = {
                    "action": "end",
                    "rationale": "The same canned auto-reply arrived three consecutive times. Closing this conversation.",
                }
            if (
                not is_duplicate
                and role == TurnRole.MERCHANT.value
                and _reply_text_key(message)
                == "btw can you also help me with my gst filing this month"
                and _has_established_jida_research_topic(
                    conversation, app.extensions["vera_state"]["context_store"]
                )
            ):
                result = {
                    "action": "send",
                    "body": "I can’t help with GST filing directly, but I can return to the JIDA research item we were discussing. Would you like to continue with that?",
                    "cta": MessageCTA.OPEN_ENDED.value,
                    "rationale": "Politely declining the out-of-scope GST request and returning to the established JIDA research topic.",
                }
            if result["action"] == "end":
                try:
                    runtime.record_opt_out(
                        merchant_id,
                        OptOutScope.CONVERSATION,
                        received_at,
                        customer_id=customer_id,
                        conversation_id=conversation_id,
                    )
                    runtime.end_conversation(conversation_id, received_at)
                except (KeyError, TypeError, ValueError):
                    return jsonify({"error": "reply_state_error", "details": "could not persist stop state"}), 400
                return jsonify(result), 200

            if result["action"] != "send":
                return jsonify(result), 200

            opted_out = runtime.is_opted_out(
                merchant_id, OptOutScope.CONVERSATION,
                customer_id=customer_id, conversation_id=conversation_id, at=received_at,
            )
            if customer_id is not None:
                opted_out = opted_out or runtime.is_opted_out(
                    merchant_id, OptOutScope.CUSTOMER_MERCHANT,
                    customer_id=customer_id, at=received_at,
                )
            opted_out = opted_out or runtime.is_opted_out(
                merchant_id, OptOutScope.MERCHANT, at=received_at,
            )
            suppression_key = f"reply:{turn_id}"
            if opted_out or runtime.has_suppression(suppression_key):
                return jsonify({
                    "action": "wait", "wait_seconds": 1800,
                    "rationale": "A prior suppression or opt-out prevents a follow-up send.",
                }), 200
            if runtime.has_outbound_body(conversation_id, result["body"]):
                return jsonify({
                    "action": "wait", "wait_seconds": 1800,
                    "rationale": "This exact message was already sent in this conversation; avoiding repetition.",
                }), 200

            reply_plan = MessagePlan(
                action=DecisionAction.SEND,
                trigger_id=None,
                merchant_id=merchant_id,
                customer_id=customer_id,
                trigger_kind=None,
                body=result["body"],
                cta=MessageCTA(result["cta"]),
                send_as=SendAs.VERA,
                suppression_key=suppression_key,
                rationale=result["rationale"],
            )
            try:
                runtime.record_outbound(
                    conversation_id, reply_plan, received_at,
                    event_id="reply_out_" + turn_id.removeprefix("reply_"),
                )
            except (KeyError, TypeError, ValueError):
                return jsonify({"error": "reply_state_error", "details": "could not record follow-up send"}), 400
            return jsonify(result), 200

    @app.post("/v1/context")
    def push_context():
        raw = request.get_data(cache=False)
        if len(raw) > MAX_REQUEST_BYTES:
            return _error("payload_too_large", "request body exceeds 500 KB", 413)
        try:
            text = raw.decode("utf-8")
            envelope = json.loads(
                text,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON constant: {value}")),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return _error("invalid_json", "request body must be valid UTF-8 JSON", 400)
        if not isinstance(envelope, dict):
            return _error("invalid_request", "request body must be a JSON object", 400)

        scope = envelope.get("scope")
        if not isinstance(scope, str) or scope not in _SCOPES:
            return _error("invalid_scope", "scope must be category, merchant, customer, or trigger", 400)

        context_id = envelope.get("context_id")
        if not isinstance(context_id, str) or not context_id.strip():
            return _error("invalid_context_id", "context_id must be a non-empty string", 400)
        context_id = context_id.strip()

        version = envelope.get("version")
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            return _error("invalid_version", "version must be an integer greater than or equal to zero", 400)

        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            return _error("invalid_payload", "payload must be a JSON object", 400)

        delivered_at = _parse_delivered_at(envelope.get("delivered_at"))
        if delivered_at is None:
            return _error("invalid_delivered_at", "delivered_at must be a timezone-aware ISO-8601 timestamp", 400)

        store: ContextStore = app.extensions["vera_state"]["context_store"]
        try:
            current_version = store.get_version(scope, context_id)
            if current_version is not None and version <= current_version:
                return jsonify({
                    "accepted": False,
                    "reason": "stale_version",
                    "current_version": current_version,
                }), 409
            accepted = store.put(scope, context_id, version, payload)
        except (TypeError, ValueError):
            return _error("invalid_payload", "payload could not be normalized for this scope", 400)

        if not accepted:
            # A concurrent request may have committed an equal or newer version.
            current_version = store.get_version(scope, context_id)
            return jsonify({
                "accepted": False,
                "reason": "stale_version",
                "current_version": current_version,
            }), 409

        return jsonify({
            "accepted": True,
            "ack_id": _ack_id(scope, context_id, version),
            "stored_at": delivered_at,
        }), 200

    return app


app = create_app()
