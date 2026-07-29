from __future__ import annotations

import asyncio
import copy
import json
import re
import time
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, Plain
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.core.message.message_event_result import MessageChain

from .services.config import PluginConfig, StoryConfig
from .services.game import ActionResult, GameService
from .services.images import ImageGenerationError, ImageService
from .services.llm import GlobalJudge, LLMService
from .services.memory import MemoryService
from .services.messaging import send_cached_image, send_generated_image, send_roundtable_forward
from .services.models import RoomMember, StoryRoom
from .services.roundtable import (
    RoundtableConfigurationError,
    RoundtableGenerationError,
    RoundtableService,
)
from .services.saves import SaveService
from .services.storage import RoomBusyRegistry, StateStore


PLUGIN_NAME = "astrbot_plugin_ai_interactive_fiction"
HELP_TEXT = """Game Start：
/故事 开始 [要求]
/故事 加入@房主 [角色要求]
/故事 继续
/故事 [选项或行动]
/故事 结束

存档与读档（有4个槽位）：
/存档 1 - 保存到1~4号槽
/读档 1 - 读取1~4号槽

查看最近AI之间的剧情讨论：
/圆桌会议

进入故事后，启用自然语言时也可直接 @我 用正常说话表达行动。"""
INVALID_SLOT_TEXT = "只有4个存档槽，请输入数字1~4"
SAVE_DISABLED_TEXT = "该故事未开启存档/读档功能"
REWOUND_TEXT = "角色因世界回溯消失，请重新加入"
CG_FAILED_TEXT = "CG生成失败，请配置或检查模型"
GENERATION_FAILED_TEXT = "生成失败，请配置或检查模型"


@register(
    PLUGIN_NAME,
    "Lan",
    "基于多模型圆桌会议的单人及群聊互动故事插件",
    "0.1.0",
)
class AIInteractiveFictionPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.astrbot_config = config
        self.raw_config = dict(config or {})
        self._normalize_roundtable_display_fields()
        self.config = PluginConfig(self.raw_config)
        self.store: StateStore | None = None
        self.busy = RoomBusyRegistry()
        self._starting_users: set[str] = set()
        self._last_cleanup_at = 0.0
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._image_triggers_inflight: set[str] = set()
        self.llm = LLMService(
            context,
            streaming=self.config.streaming,
            default_timeout=self.config.global_timeout_seconds,
            logger=logger,
        )
        self.judge = GlobalJudge(
            self.llm,
            self.config.global_judge_provider_id,
            self.config.global_judge_persona,
        )
        self.roundtable = RoundtableService(
            self.llm,
            self.config.roundtable_models,
            mode=self.config.discussion_mode,
            rounds=self.config.discussion_rounds,
            logger=logger,
        )
        self.memory = MemoryService(self.llm, logger)
        self.game = GameService(self.roundtable, self.memory, self.config.word_limits)
        self.images = ImageService(
            self.llm,
            self.config.image_generators,
            self.config.workflow_mappings,
            default_timeout=self.config.global_timeout_seconds,
            logger=logger,
        )
        self.saves: SaveService | None = None

    def _normalize_roundtable_display_fields(self) -> None:
        entries = self.raw_config.get("roundtable_models")
        if not isinstance(entries, list):
            return
        changed = False
        role_labels = {"proposal": "提案", "reviewer": "评审"}
        content_labels = {"regular": "常规", "non_safe": "非安全"}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            role_display = role_labels.get(str(entry.get("role") or "proposal"), "提案")
            content_display = content_labels.get(
                str(entry.get("content_type") or "regular"),
                "常规",
            )
            if entry.get("role_display") != role_display:
                entry["role_display"] = role_display
                changed = True
            if entry.get("content_type_display") != content_display:
                entry["content_type_display"] = content_display
                changed = True
        if not changed:
            return
        save_config = getattr(self.astrbot_config, "save_config", None)
        if callable(save_config):
            save_config(replace_config=self.raw_config)
        elif isinstance(self.astrbot_config, dict):
            self.astrbot_config.clear()
            self.astrbot_config.update(self.raw_config)

    async def initialize(self) -> None:
        data_dir = StarTools.get_data_dir(PLUGIN_NAME)
        self.store = StateStore(data_dir)
        await self.store.initialize()
        self.saves = SaveService(self.store)
        await self._cleanup_if_due(force=True)

    async def terminate(self) -> None:
        tasks = list(self._background_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self.store is not None:
            async with self.store.lock:
                await self.store.save()

    @filter.command("故事", priority=1100)
    async def story_command(self, event: AstrMessageEvent):
        """互动故事帮助，以及开始、加入和结束故事"""
        await self._dispatch_registered_command(event, fallback={"name": "help"})

    @filter.command("存档", priority=1100)
    async def save_command(self, event: AstrMessageEvent):
        """将当前故事保存到个人1至4号存档槽"""
        await self._dispatch_registered_command(event, fallback={"name": "save", "slot": None})

    @filter.command("读档", priority=1100)
    async def load_command(self, event: AstrMessageEvent):
        """从个人1至4号存档槽读取故事，不填槽位时读取最新存档"""
        await self._dispatch_registered_command(event, fallback={"name": "load", "slot": None})

    @filter.command("圆桌会议", priority=1100)
    async def roundtable_command(self, event: AstrMessageEvent):
        """查看当前群聊最近一次故事行动的圆桌讨论"""
        await self._dispatch_registered_command(event, fallback={"name": "roundtable"})

    @filter.llm_tool(name="interactive_fiction")
    async def interactive_fiction_tool(self, event: AstrMessageEvent, request: str):
        """执行互动故事操作或玩家行动。仅当用户明确想开始、加入、结束、存读档、查看圆桌会议或推进正在游玩的故事时调用；普通聊天不要调用。

        Args:
            request(string): 用户完整的互动故事请求或玩家行动原文
        """
        if self.store is None or self.saves is None:
            yield "互动故事插件尚未初始化，请稍后重试。"
            return
        if not self.config.enable_natural_language:
            yield "自然语言互动未启用，请使用 /故事 查看指令。"
            return
        if not self.config.global_judge_provider_id:
            yield "未配置全局判断LLM，无法执行自然语言互动故事请求。"
            return

        user_id = str(event.get_sender_id() or "").strip()
        text = str(request or "").strip()
        if not user_id or not text:
            yield "缺少玩家身份或互动故事请求。"
            return
        await self._cleanup_if_due()
        room = self.store.room_for_user(user_id)
        if room is None and user_id not in self.store.pending_starts:
            yield "请先使用 /故事 开始 进入故事；未开局时不会调用自然语言判断。"
            return
        story = StoryConfig.from_runtime_dict(room.story_config) if room else None
        try:
            route = await self.judge.route(
                text,
                active_room=room is not None,
                pending_story_choice=user_id in self.store.pending_starts,
                action_restriction=story.action_restriction if story else 50,
                room_context=self._judge_room_context(room, user_id),
            )
        except Exception as exc:
            logger.warning(f"互动故事函数工具调用全局判断LLM失败: {exc}")
            yield "全局判断LLM调用失败，未执行互动故事操作。"
            return
        if not route or str(route.get("intent") or "chat") == "chat":
            yield "该请求属于普通聊天，未执行互动故事操作；请直接正常回复用户。"
            return

        await self._handle_natural_route(event, user_id, text, route, room)
        yield "互动故事操作已执行，结果已直接发送给用户，无需重复回复。"

    @filter.event_message_type(filter.EventMessageType.ALL, priority=1000)
    async def on_message(self, event: AstrMessageEvent):
        if self.store is None or self.saves is None:
            return
        user_id = str(event.get_sender_id() or "").strip()
        text = str(event.get_message_str() or "").strip()
        if not user_id or not text or user_id == str(event.get_self_id() or ""):
            return
        await self._cleanup_if_due()

        command = self._parse_command(text)
        if command is not None:
            event.stop_event()
            await self._handle_command(event, user_id, command)
            return

        if not self.config.enable_natural_language or not self.config.global_judge_provider_id:
            return

        room = self.store.room_for_user(user_id)
        if room is None and user_id not in self.store.pending_starts:
            return
        story = StoryConfig.from_runtime_dict(room.story_config) if room else None
        try:
            route = await self.judge.route(
                text,
                active_room=room is not None,
                pending_story_choice=user_id in self.store.pending_starts,
                action_restriction=story.action_restriction if story else 50,
                room_context=self._judge_room_context(room, user_id),
            )
        except Exception as exc:
            logger.warning(f"全局判断LLM调用失败，消息交回AstrBot普通聊天: {exc}")
            return
        if not route or str(route.get("intent") or "chat") == "chat":
            return
        await self._handle_natural_route(event, user_id, text, route, room)

    async def _dispatch_registered_command(
        self,
        event: AstrMessageEvent,
        *,
        fallback: dict[str, Any],
    ) -> None:
        """Bridge AstrBot command registration to the existing command implementation."""
        if self.store is None or self.saves is None:
            return
        user_id = str(event.get_sender_id() or "").strip()
        if not user_id or user_id == str(event.get_self_id() or ""):
            return
        await self._cleanup_if_due()
        raw = str(event.get_message_str() or "").strip()
        normalized = raw if raw.startswith(("/", "／")) else "/" + raw
        command = self._parse_command(normalized) or fallback
        event.stop_event()
        await self._handle_command(event, user_id, command)

    async def _handle_command(self, event: AstrMessageEvent, user_id: str, command: dict[str, Any]) -> None:
        name = command["name"]
        if name == "help":
            await self._send_text(event, HELP_TEXT)
            return
        if name == "start":
            await self._start_request(event, user_id, str(command.get("args") or ""))
            return
        if name == "join":
            owner_id = self._mentioned_owner(event) or str(command.get("owner_id") or "")
            await self._join_room(event, user_id, owner_id, str(command.get("args") or ""))
            return
        if name == "end":
            await self._end_or_leave(event, user_id)
            return
        if name == "continue":
            await self._continue_last_response(event, user_id)
            return
        if name == "action":
            await self._perform_command_action(event, user_id, str(command.get("args") or ""))
            return
        if name in {"save", "load"}:
            slot_raw = command.get("slot")
            if slot_raw is not None and not self._valid_slot(slot_raw):
                await self._send_text(event, INVALID_SLOT_TEXT)
                return
            await self._save_or_load(event, user_id, name, int(slot_raw) if slot_raw is not None else None)
            return
        if name == "roundtable":
            await self._show_roundtable(event, user_id)

    async def _handle_natural_route(
        self,
        event: AstrMessageEvent,
        user_id: str,
        text: str,
        route: dict[str, Any],
        room: StoryRoom | None,
    ) -> None:
        intent = str(route.get("intent") or "chat")
        if intent == "start":
            if room is None and user_id not in self.store.pending_starts:
                return
            event.stop_event()
            await self._start_request(event, user_id, str(route.get("requirements") or ""))
            return
        if intent == "select_story":
            event.stop_event()
            await self._select_pending_story(event, user_id, str(route.get("story_choice") or text))
            return
        if intent == "join":
            if room is None:
                return
            event.stop_event()
            owner_id = self._mentioned_owner(event) or str(route.get("owner_id") or "")
            await self._join_room(event, user_id, owner_id, str(route.get("requirements") or ""))
            return
        if intent == "end":
            event.stop_event()
            await self._end_or_leave(event, user_id)
            return
        if intent in {"save", "load"}:
            event.stop_event()
            slot = route.get("slot")
            if slot is not None and not self._valid_slot(slot):
                await self._send_text(event, INVALID_SLOT_TEXT)
                return
            await self._save_or_load(event, user_id, intent, int(slot) if slot is not None else None)
            return
        if intent == "roundtable":
            event.stop_event()
            await self._show_roundtable(event, user_id)
            return
        if intent != "action":
            return

        if room is None:
            if user_id in self.store.rewound_users:
                event.stop_event()
                await self._send_text(event, REWOUND_TEXT)
            return
        event.stop_event()
        if not self._route_bool(route.get("reasonable"), True):
            await self._send_text(event, self.config.unreasonable_action_message)
            return
        content_type = "non_safe" if str(route.get("content_type")) == "non_safe" else "regular"
        resolved_action = self._resolve_command_action(room, text)
        if text.strip().isdigit() and not resolved_action:
            await self._send_text(event, "请输入1~3号选项，或直接描述行动")
            return
        await self._perform_action(
            event,
            user_id,
            room,
            resolved_action or text,
            content_type,
            include_psychology=self._route_bool(route.get("include_psychology"), False),
        )

    async def _perform_command_action(
        self,
        event: AstrMessageEvent,
        user_id: str,
        requested_action: str,
    ) -> None:
        room = self.store.room_for_user(user_id)
        if room is None:
            await self._send_text(event, REWOUND_TEXT if user_id in self.store.rewound_users else "你当前不在故事房间内")
            return
        action = self._resolve_command_action(room, requested_action)
        if not action:
            await self._send_text(event, "请输入1~3号选项，或直接使用 /故事 [行动]")
            return
        story = StoryConfig.from_runtime_dict(room.story_config)
        try:
            route = await self.judge.route(
                action,
                active_room=True,
                pending_story_choice=False,
                action_restriction=story.action_restriction,
                room_context=self._judge_room_context(room, user_id),
            )
        except Exception as exc:
            logger.warning(f"指令行动调用全局判断LLM失败: {exc}")
            await self._send_text(event, "全局判断LLM调用失败，未执行故事行动。")
            return
        if route and not self._route_bool(route.get("reasonable"), True):
            await self._send_text(event, self.config.unreasonable_action_message)
            return
        content_type = "non_safe" if route and str(route.get("content_type")) == "non_safe" else "regular"
        await self._perform_action(
            event,
            user_id,
            room,
            action,
            content_type,
            include_psychology=self._route_bool((route or {}).get("include_psychology"), False),
        )

    async def _continue_last_response(self, event: AstrMessageEvent, user_id: str) -> None:
        room = self.store.room_for_user(user_id)
        if room is None:
            await self._send_text(event, "你当前不在故事房间内")
            return
        cached = dict(room.last_response or {})
        text = str(cached.get("text") or "").strip()
        if not text:
            await self._send_text(event, "暂无可重新发送的故事回复")
            return
        await self._send_text(event, text)
        for item in list(cached.get("images") or []):
            if not isinstance(item, dict):
                continue
            path = Path(str(item.get("path") or ""))
            if path.is_file():
                await send_cached_image(event, path)

    async def _start_request(self, event: AstrMessageEvent, user_id: str, args: str) -> None:
        if self.store.room_for_user(user_id) is not None or user_id in self._starting_users:
            await self._send_text(event, "你已经在一个故事房间内")
            return
        enabled = self.config.enabled_stories()
        direct_story, requirements, is_random = self._resolve_start_args(args, enabled)
        if direct_story is not None or is_random or not enabled:
            await self._create_game(event, user_id, direct_story, requirements or args, forced_random=is_random or not enabled)
            return
        async with self.store.lock:
            self.store.pending_starts[user_id] = {
                "requirements": args.strip(),
                "created_at": time.time(),
            }
            await self.store.save()
        lines = ["请选择使用已有故事条目还是全随机：", "0. 全随机"]
        lines.extend(f"{index}. {story.name}" for index, story in enumerate(enabled, start=1))
        await self._send_text(event, "\n".join(lines))

    async def _select_pending_story(self, event: AstrMessageEvent, user_id: str, choice: str) -> None:
        pending = self.store.pending_starts.get(user_id)
        if pending is None:
            await self._send_text(event, "当前没有待选择的故事")
            return
        enabled = self.config.enabled_stories()
        selected, _unused, is_random = self._resolve_start_args(choice, enabled, choice_only=True)
        if selected is None and not is_random:
            await self._send_text(event, "未找到该故事，请回复编号、故事名称或“全随机”")
            return
        requirements = str(pending.get("requirements") or "")
        async with self.store.lock:
            self.store.pending_starts.pop(user_id, None)
            await self.store.save()
        await self._create_game(event, user_id, selected, requirements, forced_random=is_random)

    async def _create_game(
        self,
        event: AstrMessageEvent,
        user_id: str,
        story: StoryConfig | None,
        requirements: str,
        *,
        forced_random: bool,
    ) -> None:
        if not self.config.global_judge_provider_id:
            await self._send_text(event, "未配置全局判断LLM")
            return
        if user_id in self._starting_users:
            return
        self._starting_users.add(user_id)
        try:
            classification_text = "\n".join(
                [requirements, story.world if story else "", story.protagonist if story else "", ",".join(story.required_tags) if story else ""]
            )
            content_type = await self.judge.classify_content(classification_text)
            built = await self.game.build_story(
                None if forced_random else story,
                requirements=requirements,
                owner_name=event.get_sender_name(),
                content_type=content_type,
            )
            public_text, environment_text = await self._render_opening_texts(
                bible=built.bible,
                opening_state=built.opening_state,
                public_profile=built.public_profile,
            )
            room = self.game.create_room(
                user_id,
                event.get_sender_name(),
                event.unified_msg_origin,
                built,
            )
            already_joined = False
            async with self.store.lock:
                if self.store.room_for_user(user_id) is not None:
                    already_joined = True
                else:
                    self.store.rooms[room.room_id] = room
                    self.store.player_rooms[user_id] = room.room_id
                    self.store.rewound_users.pop(user_id, None)
                    self.store.pending_starts.pop(user_id, None)
                    await self.store.save()
            if already_joined:
                await self._send_text(event, "你已经在一个故事房间内")
                return
            await self._send_text(event, f"环境：{environment_text}")
            await self._send_text(
                event,
                f"{public_text}\n\n{self._format_choices(built.opening_choices, False)}",
            )
            if "first_appearance" in self.config.image_generation_triggers:
                self._spawn_background(
                    self._generate_initial_portrait(
                        event,
                        room,
                        user_id,
                        built.public_profile,
                        history_key=f"first_appearance:{user_id}",
                    )
                )
        except RoundtableConfigurationError as exc:
            await self._send_text(event, str(exc))
        except Exception as exc:
            logger.warning(f"创建故事失败: {exc}")
            await self._send_text(event, GENERATION_FAILED_TEXT)
        finally:
            self._starting_users.discard(user_id)

    async def _join_room(
        self,
        event: AstrMessageEvent,
        user_id: str,
        owner_id: str,
        requirements: str,
    ) -> None:
        if self.store.room_for_user(user_id) is not None or user_id in self._starting_users:
            await self._send_text(event, "你已经在一个故事房间内")
            return
        if not self.config.global_judge_provider_id:
            await self._send_text(event, "未配置全局判断LLM")
            return
        owner_id = "".join(ch for ch in str(owner_id) if ch.isdigit())
        target = self.store.room_for_user(owner_id) if owner_id else None
        if target is None or target.owner_id != owner_id:
            await self._send_text(event, "未找到该房主正在进行的故事")
            return
        story = StoryConfig.from_runtime_dict(target.story_config)
        if not story.multiplayer:
            await self._send_text(event, "该故事未开启群聊多人同游")
            return
        room_id = target.room_id
        if not self.busy.try_begin(room_id):
            await self._send_text(event, "故事正在生成中，请稍后重试")
            return
        owns_room_lock = True
        try:
            target = self.store.rooms.get(room_id)
            if target is None:
                await self._send_text(event, "未找到该房主正在进行的故事")
                return
            owner = target.members.get(target.owner_id)
            if owner is None:
                raise RoundtableGenerationError(GENERATION_FAILED_TEXT)
            if requirements:
                allowed, reason = await self.judge.check_join_requirements(requirements, target.bible, owner.character)
                if not allowed:
                    await self._send_text(event, reason or "角色要求与当前故事不匹配，请修改后重试")
                    return
            content_type = await self.judge.classify_content(
                f"{requirements}\n{json.dumps(target.bible, ensure_ascii=False)}"
            )
            built = await self.game.build_join_character(
                target,
                requirements=requirements,
                player_name=event.get_sender_name(),
                content_type=content_type,
            )
            public_text, environment_text = await self._render_opening_texts(
                bible=target.bible,
                opening_state=target.world_state,
                public_profile=built.public_profile,
            )
            target.members[user_id] = RoomMember(
                user_id=user_id,
                display_name=event.get_sender_name() or "玩家",
                character=built.full_character,
                joined_turn=target.turn,
                last_origin=event.unified_msg_origin,
            )
            if event.unified_msg_origin not in target.origins:
                target.origins.append(event.unified_msg_origin)
            target.last_active_at = time.time()
            async with self.store.lock:
                self.store.player_rooms[user_id] = target.room_id
                self.store.rewound_users.pop(user_id, None)
                self.store.pending_starts.pop(user_id, None)
                await self.store.save()
            await self._send_text(event, f"环境：{environment_text}")
            await self._send_text(
                event,
                f"{public_text}\n\n{self._format_choices(target.current_choices, bool(target.conversation_character_id))}",
            )
            self.busy.finish(target.room_id)
            owns_room_lock = False
            if "first_appearance" in self.config.image_generation_triggers:
                self._spawn_background(
                    self._generate_initial_portrait(
                        event,
                        target,
                        user_id,
                        built.public_profile,
                        history_key=f"first_appearance:{user_id}",
                    )
                )
        except RoundtableConfigurationError as exc:
            await self._send_text(event, str(exc))
        except Exception as exc:
            logger.warning(f"加入故事失败: {exc}")
            await self._send_text(event, GENERATION_FAILED_TEXT)
        finally:
            if owns_room_lock:
                self.busy.finish(room_id)

    async def _render_opening_texts(
        self,
        *,
        bible: dict[str, Any],
        opening_state: str,
        public_profile: dict[str, Any],
    ) -> tuple[str, str]:
        public_text = (
            await self.judge.render_public_profile(
                public_profile,
                max_chars=self.config.word_limits.profile_chars,
            )
        ).strip()
        environment_text = (
            await self.judge.render_opening_environment(
                bible=bible,
                opening_state=opening_state,
                profile=public_profile,
                max_chars=self.config.word_limits.environment_chars,
            )
        ).strip()
        environment_text = re.sub(r"^环境\s*[：:]\s*", "", environment_text).strip()
        if not public_text or not environment_text:
            raise RoundtableGenerationError(GENERATION_FAILED_TEXT)
        return public_text, environment_text

    async def _end_or_leave(self, event: AstrMessageEvent, user_id: str) -> None:
        room = self.store.room_for_user(user_id)
        if room is None:
            await self._send_text(event, "你当前不在故事房间内")
            return
        room_id = room.room_id
        if not self.busy.try_begin(room_id):
            await self._send_text(event, "故事正在生成中，请稍后重试")
            return
        try:
            room = self.store.rooms.get(room_id)
            if room is None:
                await self._send_text(event, "你当前不在故事房间内")
                return
            async with self.store.lock:
                if room.owner_id == user_id:
                    self.store.end_room(room.room_id)
                    message = "故事已结束"
                else:
                    room.members.pop(user_id, None)
                    if user_id in room.dead_users:
                        room.dead_users.remove(user_id)
                    self.store.player_rooms.pop(user_id, None)
                    self.store.rewound_users.pop(user_id, None)
                    self.store.saves.pop(user_id, None)
                    message = "已退出故事房间"
                await self.store.save()
            await self._send_text(event, message)
        finally:
            self.busy.finish(room_id)

    async def _save_or_load(
        self,
        event: AstrMessageEvent,
        user_id: str,
        operation: str,
        slot: int | None,
    ) -> None:
        room = self.store.room_for_user(user_id)
        if room is None:
            await self._send_text(event, "你当前不在故事房间内")
            return
        story = StoryConfig.from_runtime_dict(room.story_config)
        if not story.save_enabled:
            await self._send_text(event, SAVE_DISABLED_TEXT)
            return
        room_id = room.room_id
        if not self.busy.try_begin(room_id):
            await self._send_text(event, "故事正在生成中，请稍后重试")
            return
        try:
            room = self.store.rooms.get(room_id)
            if room is None:
                await self._send_text(event, "你当前不在故事房间内")
                return
            async with self.store.lock:
                if operation == "save":
                    record = self.saves.save_manual(user_id, room, slot)
                    message = f"已保存到{record.slot}号槽"
                else:
                    restored, _removed = self.saves.load(user_id, room, slot)
                    if restored is None:
                        message = "未找到可读取的存档"
                    else:
                        restored_member = restored.members.get(user_id)
                        if restored_member:
                            restored_member.last_origin = event.unified_msg_origin
                        if event.unified_msg_origin not in restored.origins:
                            restored.origins.append(event.unified_msg_origin)
                        message = f"已读取{slot}号槽" if slot is not None else "已读取最新存档"
                await self.store.save()
            await self._send_text(event, message)
        finally:
            self.busy.finish(room_id)

    async def _show_roundtable(self, event: AstrMessageEvent, user_id: str) -> None:
        room = self.store.room_for_user(user_id)
        if room is None:
            visible_rooms = [
                candidate
                for candidate in self.store.rooms.values()
                if candidate.latest_discussion
                and event.unified_msg_origin in candidate.origins
            ]
            room = max(visible_rooms, key=lambda candidate: candidate.last_active_at) if visible_rooms else None
        if room is None or not room.latest_discussion:
            await self._send_text(event, "暂无圆桌会议记录")
            return
        sent = await send_roundtable_forward(event, self.context, room.latest_discussion)
        if not sent:
            await self._send_text(event, "圆桌会议记录发送失败")

    async def _perform_action(
        self,
        event: AstrMessageEvent,
        user_id: str,
        room: StoryRoom,
        action: str,
        content_type: str,
        *,
        include_psychology: bool,
    ) -> None:
        room_id = room.room_id
        if not self.busy.try_begin(room_id):
            await self._send_text(event, "已有玩家的行动正在处理中，请等待故事回复")
            return
        owns_room_lock = True
        try:
            current = self.store.rooms.get(room_id)
            if current is None or user_id not in current.members:
                return
            room = current
            if user_id in room.dead_users:
                await self._send_text(event, "你已死亡，请读档或使用 /故事 结束")
                return
            pre_action = StoryRoom.from_dict(copy.deepcopy(room.to_dict()))
            result = await self.game.act(
                room,
                actor_id=user_id,
                action=action,
                content_type=content_type,
                forbid_player_autonomy=self.config.forbid_player_autonomy,
                include_psychology=include_psychology,
            )
            room.turn += 1
            room.last_active_at = time.time()
            room.world_state = result.state_summary or room.world_state
            room.latest_discussion = result.discussion
            room.current_choices = [] if result.death or result.story_ended else result.choices
            room.conversation_character_id = (
                "" if result.death or result.story_ended else result.conversation_character_id
            )
            room.history.append(
                {
                    "turn": room.turn,
                    "actor_id": user_id,
                    "action": action,
                    "result": result.narrative,
                }
            )
            member = room.members.get(user_id)
            if member:
                member.last_origin = event.unified_msg_origin
            if event.unified_msg_origin not in room.origins:
                room.origins.append(event.unified_msg_origin)
            if result.death and user_id not in room.dead_users:
                room.dead_users.append(user_id)
            for character in result.new_characters:
                character_id = str(character.get("id") or character.get("name") or "").strip()
                if character_id:
                    room.known_characters[character_id] = character
            for changed in result.changed_characters:
                character_id = str(changed.get("id") or "").strip()
                if not character_id:
                    continue
                if character_id in room.members:
                    public_profile = room.members[character_id].character.setdefault("public", {})
                    if isinstance(public_profile, dict):
                        public_profile.update(changed)
                elif character_id in room.known_characters:
                    room.known_characters[character_id].update(changed)
            story = StoryConfig.from_runtime_dict(room.story_config)
            await self.memory.compress_if_needed(room, story)

            response_parts = [result.narrative]
            if result.psychology:
                response_parts.append(result.psychology)
            if result.choices and not result.death and not result.story_ended:
                response_parts.append(
                    self._format_choices(result.choices, bool(result.conversation_character_id))
                )
            if result.death:
                response_parts.append("You are dead")
            response = "\n\n".join(response_parts)
            room.last_response = {
                "turn": room.turn,
                "text": response,
                "images": [],
            }
            async with self.store.lock:
                if story.save_enabled and (result.major_node or result.death):
                    self.saves.auto_save(pre_action, users=pre_action.members.keys())
                await self.store.save()

            text_sent = asyncio.Event()
            if not result.story_ended:
                self._spawn_background(
                    self._detect_and_handle_action_images(
                        event,
                        room_id=room.room_id,
                        expected_turn=room.turn,
                        action=action,
                        result=result,
                        text_sent=text_sent,
                    )
                )
            await self._send_text(event, response)
            text_sent.set()
            if result.story_ended:
                async with self.store.lock:
                    self.store.end_room(room.room_id)
                    await self.store.save()
                self.busy.finish(room_id)
                owns_room_lock = False
                return

            self.busy.finish(room_id)
            owns_room_lock = False
        except RoundtableConfigurationError as exc:
            await self._send_text(event, str(exc))
        except Exception as exc:
            logger.warning(f"故事行动生成失败: {exc}")
            await self._send_text(event, GENERATION_FAILED_TEXT)
        finally:
            if owns_room_lock:
                self.busy.finish(room_id)

    async def _generate_initial_portrait(
        self,
        event: AstrMessageEvent,
        room: StoryRoom,
        character_id: str,
        character: dict[str, Any],
        *,
        response_turn: int | None = None,
        history_key: str = "",
    ) -> bool:
        expected_turn = room.turn
        try:
            generated = await self.images.generate_character_image(
                bible=room.bible,
                character=character,
                output_dir=self._room_image_dir(room.room_id),
            )
        except Exception as exc:
            logger.warning(f"首次登场立绘生成失败: {exc}")
            return False
        should_send = False
        async with self.store.lock:
            current = self.store.rooms.get(room.room_id)
            character_exists = bool(
                current
                and (character_id in current.members or character_id in current.known_characters)
            )
            if current and current.turn >= expected_turn and character_exists:
                current.portraits[character_id] = {
                    "path": str(generated.path),
                    "prompt": generated.prompt,
                }
                await self.store.save()
                should_send = True
        if not should_send:
            self._discard_generated_path(generated.path)
            return False
        try:
            sent = await send_generated_image(event, self.context, generated.path, generated.generator)
        except Exception as exc:
            logger.warning(f"人物立绘发送失败: {exc}")
            return False
        if not sent:
            return False
        if response_turn is not None:
            await self._cache_generated_image(room.room_id, response_turn, generated)
        if history_key:
            await self._mark_image_trigger(room.room_id, history_key)
        return True

    async def _detect_and_handle_action_images(
        self,
        event: AstrMessageEvent,
        *,
        room_id: str,
        expected_turn: int,
        action: str,
        result: ActionResult,
        text_sent: asyncio.Event,
    ) -> None:
        try:
            room = self.store.rooms.get(room_id)
            if room is None:
                return
            known_characters = {
                **room.known_characters,
                **{
                    member_id: member.character.get("public", {})
                    for member_id, member in room.members.items()
                },
            }
            triggers = await self.judge.detect_image_triggers(
                action=action,
                narrative=result.narrative,
                known_characters=known_characters,
                new_characters=result.new_characters,
                changed_characters=result.changed_characters,
                enabled_triggers=self.config.image_generation_triggers,
            )
            await text_sent.wait()
            order = {
                "first_appearance": 0,
                "first_conversation": 1,
                "scene_change": 2,
                "battle_damage": 3,
                "killing": 4,
                "violation": 5,
            }
            for trigger in sorted(triggers, key=lambda item: order.get(item["type"], 99)):
                await self._handle_image_trigger(
                    event,
                    room_id=room_id,
                    expected_turn=expected_turn,
                    trigger=trigger,
                    result=result,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"故事文字已发送，但后台图像时机处理失败: {exc}")

    async def _handle_image_trigger(
        self,
        event: AstrMessageEvent,
        *,
        room_id: str,
        expected_turn: int,
        trigger: dict[str, str],
        result: ActionResult,
    ) -> None:
        trigger_type = trigger["type"]
        character_id = trigger.get("character_id") or ""
        if not character_id:
            if trigger_type == "first_conversation":
                character_id = result.conversation_character_id
            elif trigger_type in {"killing", "violation"}:
                character_id = result.cg_character_id
        history_key = (
            f"{trigger_type}:{character_id}"
            if character_id
            else f"{trigger_type}:turn-{expected_turn}"
        )
        inflight_key = f"{room_id}:{history_key}"
        room = self.store.rooms.get(room_id)
        if room is None or history_key in room.image_trigger_history or inflight_key in self._image_triggers_inflight:
            return
        self._image_triggers_inflight.add(inflight_key)
        succeeded = False
        try:
            if trigger_type == "scene_change":
                generated = await self.images.generate_scene_image(
                    bible=room.bible,
                    event_context=trigger.get("description") or result.narrative,
                    output_dir=self._room_image_dir(room_id),
                )
                succeeded = await send_generated_image(event, self.context, generated.path, generated.generator)
                if succeeded:
                    await self._cache_generated_image(room_id, expected_turn, generated)
            else:
                character = self._room_character(room, character_id)
                if not character:
                    return
                cached = room.portraits.get(character_id)
                if trigger_type in {"first_appearance", "first_conversation"} and not cached:
                    succeeded = await self._generate_initial_portrait(
                        event,
                        room,
                        character_id,
                        character,
                        response_turn=expected_turn,
                    )
                else:
                    if not cached or not Path(str(cached.get("path") or "")).is_file():
                        if trigger_type in {"killing", "violation", "battle_damage"}:
                            await self._send_text(event, CG_FAILED_TEXT)
                        return
                    update_portrait = trigger_type in {"first_conversation", "battle_damage"}
                    succeeded = await self._generate_character_edit(
                        event,
                        room_id=room_id,
                        expected_turn=expected_turn,
                        character_id=character_id,
                        character=character,
                        cached=cached,
                        event_context=trigger.get("description") or result.narrative,
                        non_safe=trigger_type in {"killing", "violation", "battle_damage"},
                        update_portrait=update_portrait,
                    )
            if succeeded:
                await self._mark_image_trigger(room_id, history_key)
        except ImageGenerationError as exc:
            logger.warning(f"{trigger_type}图像生成失败: {exc}")
            if trigger_type in {"killing", "violation", "battle_damage"}:
                await self._send_text(event, CG_FAILED_TEXT)
        finally:
            self._image_triggers_inflight.discard(inflight_key)

    async def _generate_character_edit(
        self,
        event: AstrMessageEvent,
        *,
        room_id: str,
        expected_turn: int,
        character_id: str,
        character: dict[str, Any],
        cached: dict[str, str],
        event_context: str,
        non_safe: bool,
        update_portrait: bool,
    ) -> bool:
        source_path = str(cached.get("path") or "")
        room = self.store.rooms.get(room_id)
        if room is None:
            return False
        generated = await self.images.generate_character_image(
            bible=room.bible,
            character=character,
            output_dir=self._room_image_dir(room_id),
            event_context=event_context,
            input_image=Path(source_path),
            original_prompt=str(cached.get("prompt") or ""),
            non_safe=non_safe,
        )
        current = self.store.rooms.get(room_id)
        current_cached = current.portraits.get(character_id) if current else None
        if current is None or str((current_cached or {}).get("path") or "") != source_path:
            self._discard_generated_path(generated.path)
            return False
        if update_portrait:
            async with self.store.lock:
                current = self.store.rooms.get(room_id)
                if current is None:
                    self._discard_generated_path(generated.path)
                    return False
                current.portraits[character_id] = {
                    "path": str(generated.path),
                    "prompt": generated.prompt,
                }
                await self.store.save()
        sent = await send_generated_image(event, self.context, generated.path, generated.generator)
        if sent:
            await self._cache_generated_image(room_id, expected_turn, generated)
        return sent

    async def _cache_generated_image(self, room_id: str, turn: int, generated: Any) -> None:
        async with self.store.lock:
            room = self.store.rooms.get(room_id)
            if room is None or int(room.last_response.get("turn") or -1) != turn:
                return
            images = room.last_response.setdefault("images", [])
            if isinstance(images, list) and not any(
                isinstance(item, dict) and str(item.get("path") or "") == str(generated.path)
                for item in images
            ):
                images.append(
                    {
                        "path": str(generated.path),
                        "generator": generated.generator.name,
                        "kind": generated.generator.kind,
                    }
                )
                await self.store.save()

    async def _mark_image_trigger(self, room_id: str, history_key: str) -> None:
        async with self.store.lock:
            room = self.store.rooms.get(room_id)
            if room is not None and history_key not in room.image_trigger_history:
                room.image_trigger_history.append(history_key)
                await self.store.save()

    @staticmethod
    def _room_character(room: StoryRoom, character_id: str) -> dict[str, Any]:
        if character_id in room.known_characters:
            return room.known_characters[character_id]
        member = room.members.get(character_id)
        if member is not None:
            public = member.character.get("public")
            return public if isinstance(public, dict) else {}
        return {}

    def _spawn_background(self, coroutine: Any) -> None:
        task = asyncio.create_task(coroutine)
        self._background_tasks.add(task)

        def completed(done: asyncio.Task[Any]) -> None:
            self._background_tasks.discard(done)
            if done.cancelled():
                return
            try:
                error = done.exception()
            except asyncio.CancelledError:
                return
            if error is not None:
                logger.warning(f"互动故事后台任务失败: {error}")

        task.add_done_callback(completed)

    async def _cleanup_if_due(self, force: bool = False) -> None:
        if self.store is None:
            return
        now = time.time()
        if not force and now - self._last_cleanup_at < 3600:
            return
        async with self.store.lock:
            pending_count = len(self.store.pending_starts)
            expired = self.store.cleanup_expired(self.config.cache_cleanup_days, now)
            if expired or len(self.store.pending_starts) != pending_count:
                await self.store.save()
                logger.info(f"已清理 {len(expired)} 个过期互动故事缓存")
        self._last_cleanup_at = now

    def _room_image_dir(self, room_id: str) -> Path:
        assert self.store is not None
        return self.store.images_dir / room_id

    def _discard_generated_path(self, path: Path) -> None:
        assert self.store is not None
        try:
            resolved = path.resolve()
            if resolved.parent.parent == self.store.images_dir.resolve() and resolved.is_file():
                resolved.unlink()
        except OSError:
            return

    @staticmethod
    async def _send_text(event: AstrMessageEvent, text: str) -> None:
        await event.send(MessageChain([Plain(str(text))]))

    @staticmethod
    def _judge_room_context(room: StoryRoom | None, user_id: str) -> str:
        if room is None:
            return ""
        return (
            f"当前轮次：{room.turn}；行动者是否死亡：{user_id in room.dead_users}；"
            f"最近客观状态：{room.world_state[-1000:]}；"
            f"当前候选行动：{room.current_choices}"
        )

    @staticmethod
    def _format_choices(choices: list[str], in_conversation: bool) -> str:
        normalized = [str(item).strip() for item in choices if str(item).strip()][:3]
        fallbacks = ["观察当前环境", "确认自身状态", "暂不行动并继续观察"]
        while len(normalized) < 3:
            normalized.append(fallbacks[len(normalized)])
        lines = [f"{index}. {choice}" for index, choice in enumerate(normalized, start=1)]
        if in_conversation:
            lines.extend(
                [
                    "4. 杀害",
                    "5. 侵犯",
                    "6. 可自由使用“/故事 [行动]”或 @我 行动",
                ]
            )
        else:
            lines.append("4. 可自由使用“/故事 [行动]”或 @我 行动")
        return "\n".join(lines)

    @staticmethod
    def _resolve_command_action(room: StoryRoom, requested: str) -> str:
        cleaned = str(requested or "").strip()
        if not cleaned:
            return ""
        if cleaned.isdigit():
            option = int(cleaned)
            if 1 <= option <= 3 and option <= len(room.current_choices):
                return room.current_choices[option - 1]
            if room.conversation_character_id and option == 4:
                return f"杀害正在交谈的角色（ID：{room.conversation_character_id}）"
            if room.conversation_character_id and option == 5:
                return f"侵犯正在交谈的角色（ID：{room.conversation_character_id}）"
            return ""
        if cleaned in {"杀害", "侵犯"} and room.conversation_character_id:
            return f"{cleaned}正在交谈的角色（ID：{room.conversation_character_id}）"
        return cleaned

    @staticmethod
    def _route_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() not in {"false", "0", "no", "否"}
        return default if value is None else bool(value)

    @staticmethod
    def _valid_slot(value: Any) -> bool:
        try:
            return int(value) in {1, 2, 3, 4}
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _mentioned_owner(event: AstrMessageEvent) -> str:
        self_id = str(event.get_self_id() or "")
        for component in event.get_messages():
            if isinstance(component, At) and str(component.qq) != self_id:
                return str(component.qq)
        return ""

    @staticmethod
    def _resolve_start_args(
        args: str,
        stories: list[StoryConfig],
        *,
        choice_only: bool = False,
    ) -> tuple[StoryConfig | None, str, bool]:
        cleaned = args.strip()
        random_match = re.match(r"^(?:0|全随机(?:模式)?|随机|random)(?:\s+(.*))?$", cleaned, re.IGNORECASE)
        if random_match:
            return None, str(random_match.group(1) or "").strip(), True
        number_match = re.match(r"^(\d+)(?:\s+(.*))?$", cleaned)
        if number_match:
            index = int(number_match.group(1)) - 1
            if 0 <= index < len(stories):
                return stories[index], str(number_match.group(2) or "").strip(), False
        for story in stories:
            if cleaned == story.name:
                return story, "", False
            if cleaned.startswith(story.name + " "):
                return story, cleaned[len(story.name) :].strip(), False
        return None, "" if choice_only else cleaned, False

    @staticmethod
    def _parse_command(text: str) -> dict[str, Any] | None:
        normalized = text.replace("／", "/").strip()
        if re.fullmatch(r"/故事\s*", normalized):
            return {"name": "help"}
        story_match = re.match(r"^/故事\s*(开始|加入|结束|继续)(.*)$", normalized, re.DOTALL)
        if story_match:
            action = story_match.group(1)
            rest = story_match.group(2).strip()
            if action == "开始":
                return {"name": "start", "args": rest}
            if action == "结束":
                return {"name": "end"}
            if action == "继续":
                return {"name": "continue"}
            owner_match = re.search(r"(?:@|\[At:)(\d+)", rest)
            owner_id = owner_match.group(1) if owner_match else ""
            args = re.sub(r"(?:@|\[At:)\d+\]?", "", rest).strip()
            return {"name": "join", "owner_id": owner_id, "args": args}
        slot_match = re.match(r"^/(存档|读档)(?:\s+(\S+))?\s*$", normalized)
        if slot_match:
            return {
                "name": "save" if slot_match.group(1) == "存档" else "load",
                "slot": slot_match.group(2),
            }
        if re.fullmatch(r"/圆桌会议\s*", normalized):
            return {"name": "roundtable"}
        action_match = re.match(r"^/故事\s+(.+)$", normalized, re.DOTALL)
        if action_match:
            return {"name": "action", "args": action_match.group(1).strip()}
        return None
