"""Deterministic, process-local runtime state primitives for later stages."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from threading import RLock
from typing import Any


class ConversationStatus(str, Enum):
    OPEN = "open"
    ENDED = "ended"


class TurnRole(str, Enum):
    MERCHANT = "merchant"
    CUSTOMER = "customer"
    VERA = "vera"


class OptOutScope(str, Enum):
    CONVERSATION = "conversation"
    CUSTOMER_MERCHANT = "customer_merchant"
    MERCHANT = "merchant"


@dataclass(frozen=True)
class Turn:
    turn_id: str
    role: TurnRole
    body: str
    timestamp: datetime


@dataclass(frozen=True)
class OutboundRecord:
    event_id: str | None
    body: str
    send_as: str
    suppression_key: str | None
    timestamp: datetime
    body_fingerprint: str
    action: str
    trigger_id: str | None
    rationale: str


@dataclass
class ConversationState:
    conversation_id: str
    merchant_id: str
    customer_id: str | None
    status: ConversationStatus
    created_at: datetime
    last_activity_at: datetime
    turns: list[Turn] = field(default_factory=list)
    outbound_history: list[OutboundRecord] = field(default_factory=list)
    contact_blocked: bool = False


@dataclass(frozen=True)
class OptOutRecord:
    merchant_id: str
    customer_id: str | None
    conversation_id: str | None
    scope: OptOutScope
    recorded_at: datetime
    expires_at: datetime | None = None


@dataclass(frozen=True)
class SuppressionRecord:
    suppression_key: str
    recorded_at: datetime


def _timestamp(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def body_fingerprint(body: str) -> str:
    """Return a stable SHA-256 fingerprint of exactly the UTF-8 body text."""
    if not isinstance(body, str):
        raise TypeError("body must be a string")
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


class RuntimeStateStore:
    """Thread-safe state store. Mutations are explicit; retrieved values are detached."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._conversations: dict[str, ConversationState] = {}
        self._suppressions: dict[str, SuppressionRecord] = {}
        self._opt_outs: dict[tuple[str, str, str, str], OptOutRecord] = {}

    @staticmethod
    def _required_id(value: str, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value

    def create_conversation(
        self,
        conversation_id: str,
        merchant_id: str,
        created_at: datetime,
        customer_id: str | None = None,
    ) -> ConversationState:
        conversation_id = self._required_id(conversation_id, "conversation_id")
        merchant_id = self._required_id(merchant_id, "merchant_id")
        if customer_id is not None:
            customer_id = self._required_id(customer_id, "customer_id")
        created_at = _timestamp(created_at)
        with self._lock:
            prior = self._conversations.get(conversation_id)
            if prior is not None:
                if prior.merchant_id != merchant_id or prior.customer_id != customer_id:
                    raise ValueError("conversation identity conflicts with existing state")
                return deepcopy(prior)
            state = ConversationState(
                conversation_id=conversation_id,
                merchant_id=merchant_id,
                customer_id=customer_id,
                status=ConversationStatus.OPEN,
                created_at=created_at,
                last_activity_at=created_at,
            )
            self._conversations[conversation_id] = state
            return deepcopy(state)

    def get_conversation(self, conversation_id: str) -> ConversationState | None:
        with self._lock:
            state = self._conversations.get(conversation_id)
            return deepcopy(state) if state is not None else None

    def add_turn(
        self,
        conversation_id: str,
        turn_id: str,
        role: TurnRole | str,
        body: str,
        timestamp: datetime,
    ) -> Turn:
        conversation_id = self._required_id(conversation_id, "conversation_id")
        turn_id = self._required_id(turn_id, "turn_id")
        if not isinstance(body, str):
            raise TypeError("body must be a string")
        try:
            role = TurnRole(role)
        except (ValueError, TypeError) as exc:
            raise ValueError("role must be merchant, customer, or vera") from exc
        timestamp = _timestamp(timestamp)
        turn = Turn(turn_id, role, body, timestamp)
        with self._lock:
            state = self._require_conversation(conversation_id)
            existing = next((item for item in state.turns if item.turn_id == turn_id), None)
            if existing is not None:
                if existing != turn:
                    raise ValueError("turn_id already exists with different content")
                return existing
            state.turns.append(turn)
            state.last_activity_at = max(state.last_activity_at, timestamp)
            return turn

    def record_inbound(
        self, conversation_id: str, turn_id: str, role: TurnRole | str, body: str, timestamp: datetime
    ) -> Turn:
        role = TurnRole(role)
        if role not in (TurnRole.MERCHANT, TurnRole.CUSTOMER):
            raise ValueError("inbound role must be merchant or customer")
        return self.add_turn(conversation_id, turn_id, role, body, timestamp)

    def record_outbound(
        self,
        conversation_id: str,
        message_plan: Any,
        timestamp: datetime,
        event_id: str | None = None,
    ) -> OutboundRecord:
        conversation_id = self._required_id(conversation_id, "conversation_id")
        if event_id is not None:
            event_id = self._required_id(event_id, "event_id")
        timestamp = _timestamp(timestamp)
        if getattr(getattr(message_plan, "action", None), "value", None) != "SEND":
            raise ValueError("only SEND MessagePlans can be recorded as outbound")
        body = getattr(message_plan, "body", None)
        send_as = getattr(message_plan, "send_as", None)
        if not isinstance(body, str) or not body.strip() or send_as is None:
            raise ValueError("MessagePlan must contain a body and send_as")
        send_as = getattr(send_as, "value", send_as)
        suppression_key = getattr(message_plan, "suppression_key", None)
        record = OutboundRecord(
            event_id=event_id,
            body=body,
            send_as=str(send_as),
            suppression_key=suppression_key,
            timestamp=timestamp,
            body_fingerprint=body_fingerprint(body),
            action="send",
            trigger_id=getattr(message_plan, "trigger_id", None),
            rationale=str(getattr(message_plan, "rationale", "")),
        )
        with self._lock:
            state = self._require_conversation(conversation_id)
            if getattr(message_plan, "merchant_id", None) != state.merchant_id:
                raise ValueError("MessagePlan merchant_id conflicts with conversation")
            if getattr(message_plan, "customer_id", None) != state.customer_id:
                raise ValueError("MessagePlan customer_id conflicts with conversation")
            if event_id is not None:
                existing = next((item for item in state.outbound_history if item.event_id == event_id), None)
                if existing is not None:
                    if existing != record:
                        raise ValueError("event_id already exists with different content")
                    return existing
                existing_turn = next((item for item in state.turns if item.turn_id == event_id), None)
                if existing_turn is not None and existing_turn != Turn(event_id, TurnRole.VERA, body, timestamp):
                    raise ValueError("event_id conflicts with an existing turn")
            if state.status != ConversationStatus.OPEN:
                raise ValueError("cannot record outbound in an ended conversation")
            state.outbound_history.append(record)
            state.last_activity_at = max(state.last_activity_at, timestamp)
            if event_id is not None and existing_turn is None:
                state.turns.append(Turn(event_id, TurnRole.VERA, body, timestamp))
            if suppression_key:
                self._record_suppression_locked(suppression_key, timestamp)
            return record

    def has_outbound(self, conversation_id: str) -> bool:
        with self._lock:
            return bool(self._require_conversation(conversation_id).outbound_history)

    def has_outbound_body(self, conversation_id: str, body: str) -> bool:
        """Return whether this conversation already sent the exact body."""
        fingerprint = body_fingerprint(body)
        with self._lock:
            state = self._require_conversation(conversation_id)
            return any(record.body_fingerprint == fingerprint for record in state.outbound_history)

    def has_outbound_for_trigger_body(
        self, trigger_id: str, merchant_id: str, customer_id: str | None, body: str
    ) -> bool:
        """Prevent a versioned tick for one trigger from repeating its prior body."""
        trigger_id = self._required_id(trigger_id, "trigger_id")
        merchant_id = self._required_id(merchant_id, "merchant_id")
        if customer_id is not None:
            customer_id = self._required_id(customer_id, "customer_id")
        fingerprint = body_fingerprint(body)
        with self._lock:
            return any(
                state.merchant_id == merchant_id
                and state.customer_id == customer_id
                and any(
                    record.trigger_id == trigger_id
                    and record.body_fingerprint == fingerprint
                    for record in state.outbound_history
                )
                for state in self._conversations.values()
            )

    def has_ended_conversation_for_trigger(
        self, trigger_id: str, merchant_id: str, customer_id: str | None
    ) -> bool:
        """Keep a later version of an ended tick trigger from reopening it."""
        trigger_id = self._required_id(trigger_id, "trigger_id")
        merchant_id = self._required_id(merchant_id, "merchant_id")
        if customer_id is not None:
            customer_id = self._required_id(customer_id, "customer_id")
        with self._lock:
            return any(
                state.status == ConversationStatus.ENDED
                and state.merchant_id == merchant_id
                and state.customer_id == customer_id
                and any(record.trigger_id == trigger_id for record in state.outbound_history)
                for state in self._conversations.values()
            )

    def is_first_outbound(self, conversation_id: str) -> bool:
        return not self.has_outbound(conversation_id)

    def end_conversation(self, conversation_id: str, timestamp: datetime) -> ConversationState:
        timestamp = _timestamp(timestamp)
        with self._lock:
            state = self._require_conversation(conversation_id)
            state.status = ConversationStatus.ENDED
            state.last_activity_at = max(state.last_activity_at, timestamp)
            return deepcopy(state)

    def record_opt_out(
        self,
        merchant_id: str,
        scope: OptOutScope | str,
        timestamp: datetime,
        *,
        customer_id: str | None = None,
        conversation_id: str | None = None,
        expires_at: datetime | None = None,
    ) -> OptOutRecord:
        merchant_id = self._required_id(merchant_id, "merchant_id")
        scope = OptOutScope(scope)
        timestamp = _timestamp(timestamp)
        if expires_at is not None:
            expires_at = _timestamp(expires_at)
            if expires_at < timestamp:
                raise ValueError("expires_at cannot precede recorded timestamp")
        if scope == OptOutScope.CUSTOMER_MERCHANT:
            customer_id = self._required_id(customer_id, "customer_id")
            conversation_id = None
        elif scope == OptOutScope.CONVERSATION:
            conversation_id = self._required_id(conversation_id, "conversation_id")
            if customer_id is not None:
                customer_id = self._required_id(customer_id, "customer_id")
        else:
            customer_id = None
            conversation_id = None
        record = OptOutRecord(merchant_id, customer_id, conversation_id, scope, timestamp, expires_at)
        key = (scope.value, merchant_id, customer_id or "", conversation_id or "")
        with self._lock:
            if scope == OptOutScope.CONVERSATION:
                state = self._require_conversation(conversation_id or "")
                if state.merchant_id != merchant_id or (customer_id is not None and state.customer_id != customer_id):
                    raise ValueError("opt-out identity conflicts with conversation")
                state.contact_blocked = True
            prior = self._opt_outs.get(key)
            if prior is not None:
                return prior
            self._opt_outs[key] = record
            return record

    def is_opted_out(
        self,
        merchant_id: str,
        scope: OptOutScope | str,
        *,
        customer_id: str | None = None,
        conversation_id: str | None = None,
        at: datetime | None = None,
    ) -> bool:
        merchant_id = self._required_id(merchant_id, "merchant_id")
        scope = OptOutScope(scope)
        if scope == OptOutScope.CUSTOMER_MERCHANT:
            customer_id = self._required_id(customer_id, "customer_id")
            conversation_id = None
        elif scope == OptOutScope.CONVERSATION:
            conversation_id = self._required_id(conversation_id, "conversation_id")
            if customer_id is not None:
                customer_id = self._required_id(customer_id, "customer_id")
        else:
            customer_id = conversation_id = None
        key = (scope.value, merchant_id, customer_id or "", conversation_id or "")
        with self._lock:
            record = self._opt_outs.get(key)
            if record is None:
                return False
            if record.expires_at is not None and at is not None:
                return _timestamp(at) < record.expires_at
            return True

    def clear_opt_out(
        self,
        merchant_id: str,
        scope: OptOutScope | str,
        *,
        customer_id: str | None = None,
        conversation_id: str | None = None,
    ) -> bool:
        merchant_id = self._required_id(merchant_id, "merchant_id")
        scope = OptOutScope(scope)
        if scope == OptOutScope.CUSTOMER_MERCHANT:
            customer_id = self._required_id(customer_id, "customer_id")
            conversation_id = None
        elif scope == OptOutScope.CONVERSATION:
            conversation_id = self._required_id(conversation_id, "conversation_id")
            if customer_id is not None:
                customer_id = self._required_id(customer_id, "customer_id")
        else:
            customer_id = conversation_id = None
        key = (scope.value, merchant_id, customer_id or "", conversation_id or "")
        with self._lock:
            removed = self._opt_outs.pop(key, None) is not None
            if scope == OptOutScope.CONVERSATION and conversation_id in self._conversations:
                self._conversations[conversation_id].contact_blocked = False
            return removed

    def has_suppression(self, suppression_key: str) -> bool:
        suppression_key = self._required_id(suppression_key, "suppression_key")
        with self._lock:
            return suppression_key in self._suppressions

    def record_suppression(self, suppression_key: str, timestamp: datetime) -> SuppressionRecord:
        suppression_key = self._required_id(suppression_key, "suppression_key")
        timestamp = _timestamp(timestamp)
        with self._lock:
            return self._record_suppression_locked(suppression_key, timestamp)

    def _record_suppression_locked(self, key: str, timestamp: datetime) -> SuppressionRecord:
        prior = self._suppressions.get(key)
        if prior is not None:
            return prior
        record = SuppressionRecord(key, timestamp)
        self._suppressions[key] = record
        return record

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return _jsonable({
                "conversations": self._conversations,
                "suppressions": list(self._suppressions.values()),
                "opt_outs": list(self._opt_outs.values()),
            })

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {
                "conversations": len(self._conversations),
                "turns": sum(len(item.turns) for item in self._conversations.values()),
                "outbounds": sum(len(item.outbound_history) for item in self._conversations.values()),
                "suppressions": len(self._suppressions),
                "opt_outs": len(self._opt_outs),
            }

    def clear(self) -> None:
        with self._lock:
            self._conversations.clear()
            self._suppressions.clear()
            self._opt_outs.clear()

    def _require_conversation(self, conversation_id: str) -> ConversationState:
        state = self._conversations.get(conversation_id)
        if state is None:
            raise KeyError(f"unknown conversation_id: {conversation_id}")
        return state


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
