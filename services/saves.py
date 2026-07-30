from __future__ import annotations

import time
from typing import Iterable

from .models import SaveRecord, StoryRoom
from .storage import StateStore


class SaveService:
    MANUAL_SLOTS = ("1", "2", "3", "4")
    AUTO_SLOT = "auto"

    def __init__(self, store: StateStore):
        self.store = store

    def save_manual(self, user_id: str, room: StoryRoom, slot: int | None = None) -> SaveRecord:
        slots = self.store.saves.setdefault(user_id, {})
        if slot is None:
            selected = next((item for item in self.MANUAL_SLOTS if item not in slots), None)
            if selected is None:
                selected = min(
                    self.MANUAL_SLOTS,
                    key=lambda item: slots[item].created_at if item in slots else float("-inf"),
                )
        else:
            selected = str(slot)
        record = self._record(user_id, room, selected, automatic=False)
        slots[selected] = record
        return record

    def auto_save(self, room: StoryRoom, users: Iterable[str] | None = None) -> None:
        for user_id in users or room.members.keys():
            self.store.saves.setdefault(user_id, {})[self.AUTO_SLOT] = self._record(
                user_id, room, self.AUTO_SLOT, automatic=True
            )

    def load(self, user_id: str, room: StoryRoom, slot: int | None = None) -> tuple[StoryRoom | None, list[str]]:
        records = [record for record in self.store.saves.get(user_id, {}).values() if record.room_id == room.room_id]
        if slot is not None:
            record = self.store.saves.get(user_id, {}).get(str(slot))
            if record is None or record.room_id != room.room_id:
                return None, []
        else:
            if not records:
                return None, []
            record = max(records, key=lambda item: item.created_at)

        restored = StoryRoom.from_dict(record.snapshot)
        restored.last_active_at = room.last_active_at
        restored.origins = list(dict.fromkeys([*room.origins, *restored.origins]))
        current_member_ids = set(room.members)
        snapshot_member_ids = set(restored.members)
        removed = sorted(current_member_ids - snapshot_member_ids)

        # Loading rewinds existing participants; it never resurrects somebody who has already left.
        restored.members = {
            member_id: member
            for member_id, member in restored.members.items()
            if member_id in current_member_ids
        }
        valid_character_ids = set(restored.members) | set(restored.known_characters)
        restored.dead_users = [
            member_id for member_id in restored.dead_users if member_id in restored.members
        ]
        restored.portraits = {
            character_id: portrait
            for character_id, portrait in restored.portraits.items()
            if character_id in valid_character_ids
        }
        restored.character_stats = {
            character_id: stats
            for character_id, stats in restored.character_stats.items()
            if character_id in valid_character_ids
        }
        if restored.conversation_character_id not in valid_character_ids:
            restored.conversation_character_id = ""
        for member_id in removed:
            self.store.player_rooms.pop(member_id, None)
            self.store.rewound_users[member_id] = room.room_id
        for member_id in restored.members:
            self.store.player_rooms[member_id] = room.room_id
            self.store.rewound_users.pop(member_id, None)
        self.store.rooms[room.room_id] = restored
        return restored, removed

    @staticmethod
    def _record(user_id: str, room: StoryRoom, slot: str, automatic: bool) -> SaveRecord:
        return SaveRecord(
            room_id=room.room_id,
            user_id=user_id,
            slot=slot,
            created_at=time.time(),
            snapshot=room.to_dict(),
            automatic=automatic,
        )
