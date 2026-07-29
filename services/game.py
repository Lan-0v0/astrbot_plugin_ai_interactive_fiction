from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .config import DEFAULT_CONTENT_LIMIT, StoryConfig, WordLimits
from .llm import parse_json_object
from .memory import MemoryService
from .models import RoomMember, StoryRoom
from .prompts import action_task, join_character_task, story_creation_task
from .roundtable import RoundtableGenerationError, RoundtableOutput
from .story_generator import (
    PreparedStoryGeneration,
    StoryGenerationRequest,
    StoryGeneratorService,
)


@dataclass(slots=True)
class BuiltStory:
    story: StoryConfig
    bible: dict[str, Any]
    public_profile: dict[str, Any]
    full_character: dict[str, Any]
    opening_state: str
    opening_choices: list[str]
    discussion: list[dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class BuiltCharacter:
    public_profile: dict[str, Any]
    full_character: dict[str, Any]
    discussion: list[dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class ActionResult:
    narrative: str
    psychology: str
    choices: list[str]
    conversation_character_id: str
    state_summary: str
    death: bool
    story_ended: bool
    major_node: bool
    new_characters: list[dict[str, Any]]
    changed_characters: list[dict[str, Any]]
    cg_trigger: str
    cg_character_id: str
    discussion: list[dict[str, str]]


class GameService:
    def __init__(
        self,
        generator: StoryGeneratorService,
        memory: MemoryService,
        word_limits: WordLimits | None = None,
    ):
        self.generator = generator
        self.memory = memory
        self.word_limits = word_limits or WordLimits()

    async def build_story(
        self,
        story: StoryConfig | None,
        *,
        requirements: str,
        owner_name: str,
        content_type: str,
    ) -> BuiltStory:
        request = StoryGenerationRequest(
            task=story_creation_task(
                story,
                player_requirements=requirements,
                owner_name=owner_name,
            ),
            content_type=content_type,
            temperature=story.temperature if story else 1.0,
            output_validator=_is_story_payload,
            repair_instruction=(
                "顶层必须是JSON对象，且必须包含对象类型的story_bible和"
                "public_player_profile，以及恰好3项字符串的opening_choices；"
                "同时保留private_player_profile、opening_state和runtime。"
            ),
        )
        output = await self.generator.generate(
            request,
            trigger_types=self._trigger_types("draft", content_type),
        )
        parsed = parse_json_object(output.final_text)
        if not parsed:
            raise RoundtableGenerationError("生成失败，请配置或检查模型")
        bible = parsed.get("story_bible")
        public = parsed.get("public_player_profile")
        if not isinstance(bible, dict) or not isinstance(public, dict):
            raise RoundtableGenerationError("生成失败，请配置或检查模型")
        private = parsed.get("private_player_profile")
        runtime = parsed.get("runtime") if isinstance(parsed.get("runtime"), dict) else {}
        if story is None:
            mechanisms = {"save"} if _as_bool(runtime.get("save_enabled"), False) else set()
            story = StoryConfig(
                story_id=f"random-{uuid.uuid4().hex}",
                name=str(bible.get("title") or "全随机故事"),
                enabled=True,
                expected_minutes=_safe_int(runtime.get("expected_minutes"), 20, 5, 120),
                content_limit=DEFAULT_CONTENT_LIMIT,
                mechanisms=mechanisms,
                temperature=1.0,
                memory_mode="all",
                action_restriction=50,
                multiplayer=True,
            )
        full_character = {"public": public, "private": private if isinstance(private, dict) else {}}
        return BuiltStory(
            story=story,
            bible=bible,
            public_profile=public,
            full_character=full_character,
            opening_state=str(parsed.get("opening_state") or ""),
            opening_choices=_choice_list(parsed.get("opening_choices")),
            discussion=output.discussion,
        )

    async def build_join_character(
        self,
        room: StoryRoom,
        *,
        requirements: str,
        player_name: str,
        content_type: str,
    ) -> BuiltCharacter:
        owner = room.members.get(room.owner_id)
        if owner is None:
            raise RoundtableGenerationError("生成失败，请配置或检查模型")
        request = StoryGenerationRequest(
            task=join_character_task(
                bible=room.bible,
                owner_character=owner.character,
                requirements=requirements,
                player_name=player_name,
            ),
            content_type=content_type,
            temperature=StoryConfig.from_runtime_dict(room.story_config).temperature,
            output_validator=_is_join_character_payload,
            repair_instruction=(
                "顶层必须是JSON对象，且必须包含对象类型的public_player_profile；"
                "同时保留对象类型的private_player_profile。"
            ),
        )
        output = await self.generator.generate(
            request,
            trigger_types=self._trigger_types("character_creation", content_type),
        )
        parsed = parse_json_object(output.final_text)
        if not parsed or not isinstance(parsed.get("public_player_profile"), dict):
            raise RoundtableGenerationError("生成失败，请配置或检查模型")
        public = dict(parsed["public_player_profile"])
        private = parsed.get("private_player_profile")
        return BuiltCharacter(
            public_profile=public,
            full_character={"public": public, "private": private if isinstance(private, dict) else {}},
            discussion=output.discussion,
        )

    async def act(
        self,
        room: StoryRoom,
        *,
        actor_id: str,
        action: str,
        content_type: str,
        forbid_player_autonomy: bool,
        include_psychology: bool = False,
        trigger_types: set[str] | None = None,
        prepared: PreparedStoryGeneration | None = None,
    ) -> ActionResult:
        if prepared is None:
            prepared = await self.prepare_action(
                room,
                actor_id=actor_id,
                action=action,
                content_type=content_type,
                forbid_player_autonomy=forbid_player_autonomy,
                include_psychology=include_psychology,
            )
        output = await self.generator.complete(
            prepared,
            trigger_types=set(trigger_types or {"normal_action"}),
            content_type=content_type,
        )
        result = self._parse_action(output)
        if not include_psychology:
            result.psychology = ""
        return result

    async def prepare_action(
        self,
        room: StoryRoom,
        *,
        actor_id: str,
        action: str,
        content_type: str = "auto",
        forbid_player_autonomy: bool,
        include_psychology: bool | None = None,
    ) -> PreparedStoryGeneration:
        story = StoryConfig.from_runtime_dict(room.story_config)
        memory_context = await self.memory.context_for_action(room, story)
        characters = {
            "players": {user_id: member.character for user_id, member in room.members.items()},
            "known_npcs": room.known_characters,
        }
        request = StoryGenerationRequest(
            task=action_task(
                story=story,
                bible=room.bible,
                world_state=room.world_state,
                memory_context=memory_context,
                characters=characters,
                actor_id=actor_id,
                action=action,
                content_type=content_type,
                forbid_player_autonomy=forbid_player_autonomy,
                current_choices=room.current_choices,
                include_psychology=include_psychology,
                word_limits=self.word_limits,
            ),
            content_type=content_type,
            temperature=story.temperature,
            output_validator=_is_action_payload,
            repair_instruction=(
                "顶层必须是JSON对象并包含非空narrative。故事未死亡且未结束时choices必须恰好包含3项字符串；"
                "死亡或结束时choices可以为空。保留状态、人物、CG和conversation_character_id字段。"
            ),
        )
        return await self.generator.prepare(request)

    @staticmethod
    def _trigger_types(event_type: str, content_type: str) -> set[str]:
        result = {event_type}
        if content_type == "non_safe":
            result.add("non_safe")
        return result

    @staticmethod
    def _parse_action(output: RoundtableOutput) -> ActionResult:
        parsed = parse_json_object(output.final_text)
        if parsed is None:
            parsed = {"narrative": output.final_text}
        narrative = str(parsed.get("narrative") or "").strip()
        if not narrative:
            raise RoundtableGenerationError("生成失败，请配置或检查模型")
        new_characters = parsed.get("new_characters")
        changed_characters = parsed.get("changed_characters")
        choices = parsed.get("choices")
        trigger = str(parsed.get("cg_trigger") or "none").lower()
        if trigger not in {"none", "violation", "killing"}:
            trigger = "none"
        death = _as_bool(parsed.get("death"), False)
        return ActionResult(
            narrative=narrative,
            psychology=str(parsed.get("psychology") or "").strip(),
            choices=_choice_list(choices),
            conversation_character_id=str(parsed.get("conversation_character_id") or "").strip(),
            state_summary=str(parsed.get("state_summary") or ""),
            death=death,
            story_ended=_as_bool(parsed.get("story_ended"), False) and not death,
            major_node=_as_bool(parsed.get("major_node"), False),
            new_characters=[item for item in new_characters if isinstance(item, dict)] if isinstance(new_characters, list) else [],
            changed_characters=[item for item in changed_characters if isinstance(item, dict)] if isinstance(changed_characters, list) else [],
            cg_trigger=trigger,
            cg_character_id=str(parsed.get("cg_character_id") or ""),
            discussion=output.discussion,
        )

    @staticmethod
    def create_room(owner_id: str, owner_name: str, origin: str, built: BuiltStory) -> StoryRoom:
        now = time.time()
        room_id = uuid.uuid4().hex
        return StoryRoom(
            room_id=room_id,
            owner_id=owner_id,
            story_config=built.story.to_runtime_dict(),
            bible=built.bible,
            members={
                owner_id: RoomMember(
                    user_id=owner_id,
                    display_name=owner_name or "玩家",
                    character=built.full_character,
                    joined_turn=0,
                    last_origin=origin,
                )
            },
            created_at=now,
            last_active_at=now,
            world_state=built.opening_state,
            current_choices=list(built.opening_choices),
            latest_discussion=list(built.discussion),
            origins=[origin],
        )


def _safe_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(maximum, max(minimum, parsed))


def _is_story_payload(text: str) -> bool:
    parsed = parse_json_object(text)
    return bool(
        parsed
        and isinstance(parsed.get("story_bible"), dict)
        and isinstance(parsed.get("public_player_profile"), dict)
        and len(_choice_list(parsed.get("opening_choices"))) == 3
    )


def _is_join_character_payload(text: str) -> bool:
    parsed = parse_json_object(text)
    return bool(parsed and isinstance(parsed.get("public_player_profile"), dict))


def _is_action_payload(text: str) -> bool:
    parsed = parse_json_object(text)
    if not parsed or not isinstance(parsed.get("narrative"), str) or not parsed["narrative"].strip():
        return False
    if _as_bool(parsed.get("death"), False) or _as_bool(parsed.get("story_ended"), False):
        return True
    return len(_choice_list(parsed.get("choices"))) == 3


def _choice_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()][:3]


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "是"}:
            return True
        if normalized in {"false", "0", "no", "否"}:
            return False
    return default if value is None else bool(value)
