from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from .models import SaveRecord, StoryRoom


class StateStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.state_path = data_dir / "state.json"
        self.images_dir = data_dir / "images"
        self.lock = asyncio.Lock()
        self.rooms: dict[str, StoryRoom] = {}
        self.player_rooms: dict[str, str] = {}
        self.saves: dict[str, dict[str, SaveRecord]] = {}
        self.rewound_users: dict[str, str] = {}
        self.pending_starts: dict[str, dict[str, Any]] = {}

    async def initialize(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        if not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(raw, dict):
            return
        self.rooms = {}
        for room_id, room_raw in _mapping(raw.get("rooms")).items():
            if not isinstance(room_raw, dict):
                continue
            try:
                room = StoryRoom.from_dict(room_raw)
            except (TypeError, ValueError, OverflowError):
                continue
            normalized_room_id = str(room_id)
            room.room_id = normalized_room_id
            if not room.owner_id or room.owner_id not in room.members:
                continue
            self.rooms[normalized_room_id] = room
        self.player_rooms = {str(k): str(v) for k, v in _mapping(raw.get("player_rooms")).items()}
        self.saves = {}
        for user_id, slots in _mapping(raw.get("saves")).items():
            if isinstance(slots, dict):
                parsed_slots: dict[str, SaveRecord] = {}
                for slot, record_raw in slots.items():
                    if not isinstance(record_raw, dict):
                        continue
                    try:
                        parsed_slots[str(slot)] = SaveRecord.from_dict(record_raw)
                    except (TypeError, ValueError, OverflowError):
                        continue
                if parsed_slots:
                    self.saves[str(user_id)] = parsed_slots
        self.rewound_users = {str(k): str(v) for k, v in _mapping(raw.get("rewound_users")).items()}
        self.pending_starts = {
            str(k): dict(v) for k, v in _mapping(raw.get("pending_starts")).items() if isinstance(v, dict)
        }
        self._repair_indexes()

    def _repair_indexes(self) -> None:
        valid: dict[str, str] = {}
        for room_id, room in self.rooms.items():
            for user_id in room.members:
                valid[user_id] = room_id
        self.player_rooms = valid
        self.saves = {
            user_id: {slot: save for slot, save in slots.items() if save.room_id in self.rooms}
            for user_id, slots in self.saves.items()
            if any(save.room_id in self.rooms for save in slots.values())
        }
        self.rewound_users = {
            user_id: room_id
            for user_id, room_id in self.rewound_users.items()
            if room_id in self.rooms and user_id not in valid
        }

    def _serialize(self) -> dict[str, Any]:
        return {
            "version": 1,
            "rooms": {room_id: room.to_dict() for room_id, room in self.rooms.items()},
            "player_rooms": self.player_rooms,
            "saves": {
                user_id: {slot: record.to_dict() for slot, record in slots.items()}
                for user_id, slots in self.saves.items()
            },
            "rewound_users": self.rewound_users,
            "pending_starts": self.pending_starts,
        }

    async def save(self) -> None:
        payload = self._serialize()
        await asyncio.to_thread(self._write_sync, payload)

    def _write_sync(self, payload: dict[str, Any]) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = self.state_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary_path, self.state_path)

    def room_for_user(self, user_id: str) -> StoryRoom | None:
        room_id = self.player_rooms.get(str(user_id))
        return self.rooms.get(room_id) if room_id else None

    def end_room(self, room_id: str) -> list[str]:
        room = self.rooms.pop(room_id, None)
        if room is None:
            return []
        removed_users = list(room.members)
        for user_id in removed_users:
            if self.player_rooms.get(user_id) == room_id:
                self.player_rooms.pop(user_id, None)
            self.rewound_users.pop(user_id, None)
        self.rewound_users = {
            user_id: rewound_room_id
            for user_id, rewound_room_id in self.rewound_users.items()
            if rewound_room_id != room_id
        }
        for user_id, slots in list(self.saves.items()):
            remaining = {slot: record for slot, record in slots.items() if record.room_id != room_id}
            if remaining:
                self.saves[user_id] = remaining
            else:
                self.saves.pop(user_id, None)
        room_image_dir = self.images_dir / room_id
        images_root = self.images_dir.resolve()
        resolved_room_dir = room_image_dir.resolve()
        if resolved_room_dir.parent == images_root and resolved_room_dir.exists():
            shutil.rmtree(room_image_dir, ignore_errors=True)
        return removed_users

    def cleanup_expired(self, days: int, now: float | None = None) -> list[str]:
        if days <= 0:
            return []
        cutoff = (now if now is not None else time.time()) - days * 86400
        expired = [room_id for room_id, room in self.rooms.items() if room.last_active_at < cutoff]
        for room_id in expired:
            self.end_room(room_id)
        self.pending_starts = {
            user_id: pending
            for user_id, pending in self.pending_starts.items()
            if float(pending.get("created_at") or 0) >= (now if now is not None else time.time()) - 1800
        }
        return expired


class RoomBusyRegistry:
    """An event-loop-local non-blocking room mutation lock."""

    def __init__(self) -> None:
        self._busy: set[str] = set()

    def try_begin(self, room_id: str) -> bool:
        if room_id in self._busy:
            return False
        self._busy.add(room_id)
        return True

    def finish(self, room_id: str) -> None:
        self._busy.discard(room_id)

    def is_busy(self, room_id: str) -> bool:
        return room_id in self._busy


def _mapping(value: Any) -> dict[Any, Any]:
    return value if isinstance(value, dict) else {}
