from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class RoomMember:
    user_id: str
    display_name: str
    character: dict[str, Any]
    joined_turn: int
    last_origin: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RoomMember":
        return cls(
            user_id=str(raw.get("user_id") or ""),
            display_name=str(raw.get("display_name") or "玩家"),
            character=dict(raw.get("character") or {}),
            joined_turn=int(raw.get("joined_turn") or 0),
            last_origin=str(raw.get("last_origin") or ""),
        )


@dataclass(slots=True)
class StoryRoom:
    room_id: str
    owner_id: str
    story_config: dict[str, Any]
    bible: dict[str, Any]
    members: dict[str, RoomMember]
    created_at: float
    last_active_at: float
    turn: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    memory_summary: str = ""
    world_state: str = ""
    latest_discussion: list[dict[str, str]] = field(default_factory=list)
    dead_users: list[str] = field(default_factory=list)
    portraits: dict[str, dict[str, str]] = field(default_factory=dict)
    known_characters: dict[str, dict[str, Any]] = field(default_factory=dict)
    current_choices: list[str] = field(default_factory=list)
    conversation_character_id: str = ""
    image_trigger_history: list[str] = field(default_factory=list)
    last_response: dict[str, Any] = field(default_factory=dict)
    origins: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["members"] = {user_id: asdict(member) for user_id, member in self.members.items()}
        return data

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "StoryRoom":
        return cls(
            room_id=str(raw.get("room_id") or ""),
            owner_id=str(raw.get("owner_id") or ""),
            story_config=dict(raw.get("story_config") or {}),
            bible=dict(raw.get("bible") or {}),
            members={
                str(user_id): RoomMember.from_dict(member)
                for user_id, member in dict(raw.get("members") or {}).items()
                if isinstance(member, dict)
            },
            created_at=float(raw.get("created_at") or 0),
            last_active_at=float(raw.get("last_active_at") or 0),
            turn=int(raw.get("turn") or 0),
            history=list(raw.get("history") or []),
            memory_summary=str(raw.get("memory_summary") or ""),
            world_state=str(raw.get("world_state") or ""),
            latest_discussion=list(raw.get("latest_discussion") or []),
            dead_users=[str(item) for item in raw.get("dead_users", [])],
            portraits=dict(raw.get("portraits") or {}),
            known_characters=dict(raw.get("known_characters") or {}),
            current_choices=[str(item) for item in raw.get("current_choices", []) if str(item).strip()],
            conversation_character_id=str(raw.get("conversation_character_id") or ""),
            image_trigger_history=[
                str(item) for item in raw.get("image_trigger_history", []) if str(item).strip()
            ],
            last_response=dict(raw.get("last_response") or {}),
            origins=[str(item) for item in raw.get("origins", []) if str(item).strip()],
        )


@dataclass(slots=True)
class SaveRecord:
    room_id: str
    user_id: str
    slot: str
    created_at: float
    snapshot: dict[str, Any]
    automatic: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SaveRecord":
        return cls(
            room_id=str(raw.get("room_id") or ""),
            user_id=str(raw.get("user_id") or ""),
            slot=str(raw.get("slot") or ""),
            created_at=float(raw.get("created_at") or 0),
            snapshot=dict(raw.get("snapshot") or {}),
            automatic=bool(raw.get("automatic", False)),
        )
