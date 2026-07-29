from __future__ import annotations

import asyncio
import importlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


ROOT = Path(__file__).resolve().parent
PACKAGE_NAME = "_astrbot_plugin_ai_interactive_fiction_integration"

try:
    import astrbot  # noqa: F401
except ImportError:
    ASTRBOT_AVAILABLE = False
else:
    ASTRBOT_AVAILABLE = True


def load_plugin_module():
    if PACKAGE_NAME not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            PACKAGE_NAME,
            ROOT / "__init__.py",
            submodule_search_locations=[str(ROOT)],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("无法创建插件包导入规范")
        package = importlib.util.module_from_spec(spec)
        sys.modules[PACKAGE_NAME] = package
        spec.loader.exec_module(package)
    return importlib.import_module(PACKAGE_NAME + ".main")


@unittest.skipUnless(ASTRBOT_AVAILABLE, "AstrBot runtime is not installed")
class AstrBotRegistrationTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_plugin_module()

    def test_commands_are_registered_with_higher_priority_than_router(self) -> None:
        from astrbot.core.star.filter.command import CommandFilter
        from astrbot.core.star.star_handler import EventType, star_handlers_registry

        expected = {"故事", "存档", "读档", "圆桌会议", "查看故事"}
        found: dict[str, int] = {}
        handlers = star_handlers_registry.get_handlers_by_event_type(EventType.AdapterMessageEvent)
        for handler in handlers:
            if handler.handler_module_path != self.module.__name__:
                continue
            for filter_ref in handler.event_filters:
                if isinstance(filter_ref, CommandFilter):
                    found[filter_ref.command_name] = int(handler.extras_configs.get("priority", 0))
        self.assertEqual(set(found), expected)
        self.assertTrue(all(priority > 1000 for priority in found.values()))

    def test_interactive_fiction_llm_tool_is_registered(self) -> None:
        from astrbot.core.provider.register import llm_tools

        tool = llm_tools.get_func("interactive_fiction")
        self.assertIsNotNone(tool)
        self.assertIn("普通聊天不要调用", tool.description)
        properties = dict((tool.parameters or {}).get("properties") or {})
        self.assertEqual(properties["request"]["type"], "string")

    def test_command_parser_keeps_documented_surface(self) -> None:
        parse = self.module.AIInteractiveFictionPlugin._parse_command
        self.assertEqual(parse("/故事"), {"name": "help"})
        self.assertEqual(parse("/故事 开始 主角是女性"), {"name": "start", "args": "主角是女性"})
        self.assertEqual(parse("/故事 继续"), {"name": "action", "args": "继续"})
        self.assertEqual(parse("/查看故事"), {"name": "view_story"})
        self.assertEqual(parse("/故事 2"), {"name": "action", "args": "2"})
        self.assertEqual(parse("/故事 杀害"), {"name": "action", "args": "杀害"})
        self.assertEqual(parse("/存档 5"), {"name": "save", "slot": "5"})
        self.assertEqual(parse("/读档 1"), {"name": "load", "slot": "1"})
        self.assertEqual(parse("/圆桌会议"), {"name": "roundtable"})

    def test_story_help_menu_matches_public_copy(self) -> None:
        self.assertEqual(
            self.module.HELP_TEXT,
            """Game Start：
/故事 开始 [要求]
/故事 加入@房主 [角色要求]
/故事 [选项或行动]
/故事 结束

查看最近一次故事回复：
/查看故事

存档与读档（有4个槽位）：
/存档 1 - 保存到1~4号槽
/读档 1 - 读取1~4号槽

查看最近AI之间的剧情讨论：
/圆桌会议

进入故事后，启用自然语言时也可直接 @我 用正常说话表达行动。""",
        )

    async def test_registered_command_bridge_handles_astrbot_stripped_prefix(self) -> None:
        class Event:
            stopped = False

            def get_sender_id(self):
                return "10001"

            def get_self_id(self):
                return "20002"

            def get_message_str(self):
                return "故事 开始 主角是女性"

            def stop_event(self):
                self.stopped = True

        plugin = object.__new__(self.module.AIInteractiveFictionPlugin)
        plugin.store = object()
        plugin.saves = object()
        event = Event()
        with (
            patch.object(plugin, "_cleanup_if_due", AsyncMock()),
            patch.object(plugin, "_handle_command", AsyncMock()) as handle,
        ):
            await plugin._dispatch_registered_command(event, fallback={"name": "help"})
        self.assertTrue(event.stopped)
        handle.assert_awaited_once_with(
            event,
            "10001",
            {"name": "start", "args": "主角是女性"},
        )

    async def test_invalid_registered_slot_uses_fixed_error_message(self) -> None:
        plugin = object.__new__(self.module.AIInteractiveFictionPlugin)
        event = object()
        with patch.object(plugin, "_send_text", AsyncMock()) as send:
            await plugin._handle_command(event, "10001", {"name": "save", "slot": "5"})
        send.assert_awaited_once_with(event, self.module.INVALID_SLOT_TEXT)

    async def test_story_action_command_selects_pending_story(self) -> None:
        plugin = object.__new__(self.module.AIInteractiveFictionPlugin)
        plugin.store = SimpleNamespace(pending_starts={"10001": {}})
        plugin._select_pending_story = AsyncMock()
        plugin._perform_command_action = AsyncMock()

        await plugin._handle_command(object(), "10001", {"name": "action", "args": "1"})

        plugin._select_pending_story.assert_awaited_once_with(unittest.mock.ANY, "10001", "1")
        plugin._perform_command_action.assert_not_awaited()

    async def test_llm_tool_respects_natural_language_switch(self) -> None:
        plugin = object.__new__(self.module.AIInteractiveFictionPlugin)
        plugin.store = object()
        plugin.saves = object()
        plugin.config = SimpleNamespace(enable_natural_language=False)
        output = [item async for item in plugin.interactive_fiction_tool(object(), "开始故事")]
        self.assertEqual(output, ["自然语言互动未启用，请使用 /故事 查看指令。"])

    async def test_llm_tool_does_not_route_before_game_starts(self) -> None:
        class Store:
            pending_starts = {}
            rewound_users = {}

            @staticmethod
            def room_for_user(_user_id):
                return None

        plugin = object.__new__(self.module.AIInteractiveFictionPlugin)
        plugin.store = Store()
        plugin.saves = object()
        plugin.config = SimpleNamespace(
            enable_natural_language=True,
            global_judge_provider_id="judge",
        )
        plugin.judge = SimpleNamespace(route=AsyncMock())
        plugin._cleanup_if_due = AsyncMock()
        event = SimpleNamespace(get_sender_id=lambda: "10001")
        output = [item async for item in plugin.interactive_fiction_tool(event, "开始一局故事")]
        self.assertEqual(
            output,
            ["请先使用 /故事 开始 进入故事；未开局时不会调用自然语言判断。"],
        )
        plugin.judge.route.assert_not_awaited()

    async def test_llm_tool_routes_numeric_choice_without_intent_judgment(self) -> None:
        from _astrbot_plugin_ai_interactive_fiction_integration.services.models import StoryRoom

        room = StoryRoom(
            room_id="room-1",
            owner_id="10001",
            story_config={},
            bible={},
            members={},
            created_at=1,
            last_active_at=1,
            current_choices=["甲", "乙", "丙"],
        )
        plugin = object.__new__(self.module.AIInteractiveFictionPlugin)
        plugin.store = SimpleNamespace(
            pending_starts={},
            rewound_users={},
            room_for_user=lambda _user_id: room,
        )
        plugin.saves = object()
        plugin.config = SimpleNamespace(
            enable_natural_language=True,
            global_judge_provider_id="judge",
        )
        plugin.judge = SimpleNamespace(route=AsyncMock(return_value={"intent": "chat"}))
        plugin._cleanup_if_due = AsyncMock()
        plugin._perform_command_action = AsyncMock()
        event = SimpleNamespace(
            get_sender_id=lambda: "10001",
            get_messages=lambda: [],
        )

        output = [item async for item in plugin.interactive_fiction_tool(event, "第三项")]

        self.assertEqual(output, ["互动故事操作已执行，结果已直接发送给用户，无需重复回复。"])
        plugin._perform_command_action.assert_awaited_once_with(event, "10001", "3")
        plugin.judge.route.assert_not_awaited()

    async def test_message_router_skips_judge_before_game_starts(self) -> None:
        class Store:
            pending_starts = {}
            rewound_users = {}

            @staticmethod
            def room_for_user(_user_id):
                return None

        plugin = object.__new__(self.module.AIInteractiveFictionPlugin)
        plugin.store = Store()
        plugin.saves = object()
        plugin.config = SimpleNamespace(
            enable_natural_language=True,
            global_judge_provider_id="judge",
        )
        plugin.judge = SimpleNamespace(route=AsyncMock())
        plugin._cleanup_if_due = AsyncMock()
        event = SimpleNamespace(
            get_sender_id=lambda: "10001",
            get_self_id=lambda: "20002",
            get_message_str=lambda: "今天天气怎么样",
        )
        await plugin.on_message(event)
        plugin.judge.route.assert_not_awaited()

    async def test_rewound_player_natural_action_gets_rejoin_notice(self) -> None:
        class Store:
            pending_starts = {}
            rewound_users = {"10001": "room-1"}
            rooms = {}

            @staticmethod
            def room_for_user(_user_id):
                return None

        plugin = object.__new__(self.module.AIInteractiveFictionPlugin)
        plugin.store = Store()
        plugin.saves = object()
        plugin.config = SimpleNamespace(
            enable_natural_language=True,
            global_judge_provider_id="judge",
        )
        plugin.judge = SimpleNamespace(
            route=AsyncMock(return_value={"intent": "action", "reasonable": True})
        )
        plugin._cleanup_if_due = AsyncMock()
        plugin._send_text = AsyncMock()
        event = SimpleNamespace(
            stopped=False,
            get_sender_id=lambda: "10001",
            get_self_id=lambda: "20002",
            get_message_str=lambda: "我继续向前走",
            stop_event=lambda: setattr(event, "stopped", True),
        )

        await plugin.on_message(event)

        self.assertTrue(event.stopped)
        plugin.judge.route.assert_awaited_once()
        plugin._send_text.assert_awaited_once_with(event, self.module.REWOUND_TEXT)

    async def test_at_bot_numeric_choice_bypasses_ambiguous_intent_judgment(self) -> None:
        from astrbot.api.message_components import At, Plain
        from _astrbot_plugin_ai_interactive_fiction_integration.services.models import StoryRoom

        room = StoryRoom(
            room_id="room-1",
            owner_id="10001",
            story_config={},
            bible={},
            members={},
            created_at=1,
            last_active_at=1,
            current_choices=["退开并倾听", "检查练习纸", "打开教室门"],
        )
        store = SimpleNamespace(
            pending_starts={},
            rewound_users={},
            room_for_user=lambda _user_id: room,
        )
        plugin = object.__new__(self.module.AIInteractiveFictionPlugin)
        plugin.store = store
        plugin.saves = object()
        plugin.config = SimpleNamespace(
            enable_natural_language=True,
            global_judge_provider_id="judge",
        )
        plugin.judge = SimpleNamespace(route=AsyncMock(return_value={"intent": "chat"}))
        plugin._cleanup_if_due = AsyncMock()
        plugin._perform_command_action = AsyncMock()
        event = SimpleNamespace(
            stopped=False,
            get_sender_id=lambda: "10001",
            get_self_id=lambda: "20002",
            # OneBot excludes the first @self component from message_str.
            get_message_str=lambda: "3",
            get_messages=lambda: [At(qq="20002", name="阿米娅"), Plain("3")],
        )
        event.stop_event = lambda: setattr(event, "stopped", True)

        await plugin.on_message(event)

        self.assertTrue(event.stopped)
        plugin._perform_command_action.assert_awaited_once_with(event, "10001", "3")
        plugin.judge.route.assert_not_awaited()

    def test_direct_natural_choice_accepts_common_forms_and_validates_room_options(self) -> None:
        from _astrbot_plugin_ai_interactive_fiction_integration.services.models import StoryRoom

        room = StoryRoom(
            room_id="room-1",
            owner_id="10001",
            story_config={},
            bible={},
            members={},
            created_at=1,
            last_active_at=1,
            current_choices=["甲", "乙", "丙"],
        )
        event = SimpleNamespace(get_messages=lambda: [])
        parse = self.module.AIInteractiveFictionPlugin._direct_natural_choice
        self.assertEqual(parse(event, "3", room), "3")
        self.assertEqual(parse(event, "选3", room), "3")
        self.assertEqual(parse(event, "第三项", room), "3")
        self.assertEqual(parse(event, "4", room), "")
        self.assertEqual(parse(event, "今天有3个人", room), "")

    async def test_rewound_player_can_naturally_rejoin_original_room(self) -> None:
        room = SimpleNamespace(owner_id="owner")
        store = SimpleNamespace(
            rewound_users={"10001": "room-1"},
            rooms={"room-1": room},
        )
        plugin = object.__new__(self.module.AIInteractiveFictionPlugin)
        plugin.store = store
        plugin._mentioned_owner = MagicMock(return_value="")
        plugin._join_room = AsyncMock()
        event = SimpleNamespace(stop_event=MagicMock())

        await plugin._handle_natural_route(
            event,
            "10001",
            "重新加入",
            {"intent": "join", "requirements": ""},
            None,
        )

        event.stop_event.assert_called_once()
        plugin._join_room.assert_awaited_once_with(event, "10001", "owner", "")

    def test_choice_format_and_numeric_resolution(self) -> None:
        from _astrbot_plugin_ai_interactive_fiction_integration.services.models import StoryRoom

        plugin_type = self.module.AIInteractiveFictionPlugin
        self.assertEqual(
            plugin_type._format_choices(["甲", "乙", "丙"], False),
            "1. 甲\n2. 乙\n3. 丙\n4. 可自由使用“/故事 [行动]”或 @我 行动",
        )
        self.assertEqual(
            plugin_type._format_choices(["甲", "乙", "丙"], True),
            "1. 甲\n2. 乙\n3. 丙\n4. 杀害\n5. 侵犯\n6. 可自由使用“/故事 [行动]”或 @我 行动",
        )
        room = StoryRoom(
            room_id="room",
            owner_id="owner",
            story_config={},
            bible={},
            members={},
            created_at=1,
            last_active_at=1,
            current_choices=["甲", "乙", "丙"],
            conversation_character_id="npc-1",
        )
        self.assertEqual(plugin_type._resolve_command_action(room, "2"), "乙")
        self.assertIn("npc-1", plugin_type._resolve_command_action(room, "4"))
        self.assertIn("npc-1", plugin_type._resolve_command_action(room, "侵犯"))

    async def test_continue_replays_cached_text(self) -> None:
        from _astrbot_plugin_ai_interactive_fiction_integration.services.models import StoryRoom

        room = StoryRoom(
            room_id="room",
            owner_id="owner",
            story_config={},
            bible={},
            members={},
            created_at=1,
            last_active_at=1,
            last_response={"turn": 1, "text": "结果\n\n1. 前进", "images": []},
        )
        plugin = object.__new__(self.module.AIInteractiveFictionPlugin)
        plugin.store = SimpleNamespace(room_for_user=lambda _user_id: room)
        event = object()
        with patch.object(plugin, "_send_text", AsyncMock()) as send:
            await plugin._continue_last_response(event, "owner")
        send.assert_awaited_once_with(event, "结果\n\n1. 前进")

    async def test_continue_replays_completed_images(self) -> None:
        from _astrbot_plugin_ai_interactive_fiction_integration.services.models import StoryRoom

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "result.png"
            image_path.write_bytes(b"image")
            room = StoryRoom(
                room_id="room",
                owner_id="owner",
                story_config={},
                bible={},
                members={},
                created_at=1,
                last_active_at=1,
                last_response={
                    "turn": 1,
                    "text": "结果",
                    "images": [{"path": str(image_path)}],
                },
            )
            plugin = object.__new__(self.module.AIInteractiveFictionPlugin)
            plugin.store = SimpleNamespace(room_for_user=lambda _user_id: room)
            with (
                patch.object(plugin, "_send_text", AsyncMock()),
                patch.object(self.module, "send_cached_image", AsyncMock(return_value=True)) as send_image,
            ):
                await plugin._continue_last_response(object(), "owner")
            send_image.assert_awaited_once_with(unittest.mock.ANY, image_path)

    async def test_action_caches_reply_and_spawns_image_work_without_waiting(self) -> None:
        from _astrbot_plugin_ai_interactive_fiction_integration.services.config import StoryConfig
        from _astrbot_plugin_ai_interactive_fiction_integration.services.game import ActionResult
        from _astrbot_plugin_ai_interactive_fiction_integration.services.models import RoomMember, StoryRoom
        from _astrbot_plugin_ai_interactive_fiction_integration.services.storage import RoomBusyRegistry

        member = RoomMember("owner", "玩家", {"public": {"name": "玩家"}}, 0, "origin")
        room = StoryRoom(
            room_id="room",
            owner_id="owner",
            story_config=StoryConfig("story", "测试", enabled=True).to_runtime_dict(),
            bible={},
            members={"owner": member},
            created_at=1,
            last_active_at=1,
        )

        class Store:
            def __init__(self):
                self.rooms = {"room": room}
                self.lock = asyncio.Lock()
                self.save = AsyncMock()

        result = ActionResult(
            narrative="你推开门。",
            psychology="",
            choices=["进入", "观察", "返回"],
            conversation_character_id="npc-1",
            state_summary="门已打开",
            death=False,
            story_ended=False,
            major_node=False,
            new_characters=[],
            changed_characters=[],
            cg_trigger="none",
            cg_character_id="",
            discussion=[],
        )
        plugin = object.__new__(self.module.AIInteractiveFictionPlugin)
        plugin.store = Store()
        plugin.busy = RoomBusyRegistry()
        plugin.game = SimpleNamespace(act=AsyncMock(return_value=result))
        plugin.memory = SimpleNamespace(compress_if_needed=AsyncMock(return_value=False))
        plugin.saves = SimpleNamespace(auto_save=MagicMock())
        plugin.config = SimpleNamespace(
            forbid_player_autonomy=True,
            image_generation_triggers={"scene_change"},
        )
        plugin.images = SimpleNamespace(has_enabled_generators=lambda: True)
        plugin._send_text = AsyncMock()
        spawned: list[object] = []

        def capture_background(coroutine):
            spawned.append(coroutine)
            coroutine.close()

        plugin._spawn_background = capture_background
        event = SimpleNamespace(unified_msg_origin="origin")
        await plugin._perform_action(
            event,
            "owner",
            room,
            "推门",
            "regular",
            include_psychology=False,
        )
        self.assertFalse(plugin.busy.is_busy("room"))
        self.assertEqual(len(spawned), 1)
        self.assertEqual(room.conversation_character_id, "npc-1")
        self.assertIn("4. 杀害", room.last_response["text"])
        self.assertIn("6. 可自由使用", room.last_response["text"])
        plugin._send_text.assert_awaited_once_with(event, room.last_response["text"])

    async def test_command_action_checks_room_lock_before_judge(self) -> None:
        from _astrbot_plugin_ai_interactive_fiction_integration.services.models import StoryRoom
        from _astrbot_plugin_ai_interactive_fiction_integration.services.storage import RoomBusyRegistry

        room = StoryRoom(
            room_id="room-1",
            owner_id="10001",
            story_config={},
            bible={},
            members={},
            created_at=1,
            last_active_at=1,
        )
        plugin = object.__new__(self.module.AIInteractiveFictionPlugin)
        plugin.store = SimpleNamespace(room_for_user=lambda _user_id: room, rewound_users={})
        plugin.busy = RoomBusyRegistry()
        plugin.busy.try_begin(room.room_id)
        plugin.judge = SimpleNamespace(route=AsyncMock())
        plugin._send_text = AsyncMock()

        await plugin._perform_command_action(object(), "10001", "前进")

        plugin.judge.route.assert_not_awaited()
        plugin._send_text.assert_awaited_once_with(
            unittest.mock.ANY,
            "已有玩家的行动正在处理中，请等待故事回复",
        )

    async def test_command_action_runs_judge_and_story_draft_concurrently(self) -> None:
        from _astrbot_plugin_ai_interactive_fiction_integration.services.config import StoryConfig
        from _astrbot_plugin_ai_interactive_fiction_integration.services.models import StoryRoom
        from _astrbot_plugin_ai_interactive_fiction_integration.services.storage import RoomBusyRegistry

        room = StoryRoom(
            room_id="room-1",
            owner_id="10001",
            story_config=StoryConfig("story", "测试", enabled=True).to_runtime_dict(),
            bible={},
            members={},
            created_at=1,
            last_active_at=1,
        )
        judge_started = asyncio.Event()
        story_started = asyncio.Event()
        release_judge = asyncio.Event()
        prepared = object()

        async def judge_route(*_args, **_kwargs):
            judge_started.set()
            await asyncio.wait_for(story_started.wait(), 1)
            await asyncio.wait_for(release_judge.wait(), 1)
            return {
                "reasonable": True,
                "content_type": "regular",
                "action_level": "high_risk_complex",
                "requests_story_change": False,
            }

        async def prepare_story(*_args, **_kwargs):
            story_started.set()
            await asyncio.wait_for(judge_started.wait(), 1)
            return prepared

        plugin = object.__new__(self.module.AIInteractiveFictionPlugin)
        plugin.store = SimpleNamespace(room_for_user=lambda _user_id: room, rewound_users={})
        plugin.busy = RoomBusyRegistry()
        plugin.config = SimpleNamespace(forbid_player_autonomy=True)
        plugin.judge = SimpleNamespace(route=AsyncMock(side_effect=judge_route))
        plugin.game = SimpleNamespace(prepare_action=AsyncMock(side_effect=prepare_story))
        plugin._perform_action = AsyncMock()
        plugin._send_text = AsyncMock()

        first_action = asyncio.create_task(
            plugin._perform_command_action(object(), "10001", "打开机关门")
        )
        await asyncio.wait_for(judge_started.wait(), 1)
        await asyncio.wait_for(story_started.wait(), 1)
        await plugin._perform_command_action(object(), "10001", "抢先穿过门")
        plugin._send_text.assert_awaited_once_with(
            unittest.mock.ANY,
            "已有玩家的行动正在处理中，请等待故事回复",
        )
        release_judge.set()
        await first_action

        plugin._perform_action.assert_awaited_once()
        kwargs = plugin._perform_action.await_args.kwargs
        self.assertIs(kwargs["prepared"], prepared)
        self.assertEqual(kwargs["trigger_types"], {"high_risk_complex_action"})
        self.assertTrue(kwargs["lock_already_held"])
        plugin.busy.finish(room.room_id)

    async def test_non_safe_action_is_not_rejected_when_reason_is_none(self) -> None:
        from _astrbot_plugin_ai_interactive_fiction_integration.services.config import StoryConfig
        from _astrbot_plugin_ai_interactive_fiction_integration.services.models import StoryRoom
        from _astrbot_plugin_ai_interactive_fiction_integration.services.storage import RoomBusyRegistry

        room = StoryRoom(
            room_id="room-1",
            owner_id="10001",
            story_config=StoryConfig("story", "测试", enabled=True).to_runtime_dict(),
            bible={},
            members={},
            created_at=1,
            last_active_at=1,
        )
        prepared = object()
        plugin = object.__new__(self.module.AIInteractiveFictionPlugin)
        plugin.store = SimpleNamespace(room_for_user=lambda _user_id: room, rewound_users={})
        plugin.busy = RoomBusyRegistry()
        plugin.config = SimpleNamespace(
            forbid_player_autonomy=True,
            unreasonable_action_message="你想搁这开挂呢？哒咩！",
        )
        plugin.judge = SimpleNamespace(
            route=AsyncMock(
                return_value={
                    "intent": "action",
                    "reasonable": False,
                    "unreasonable_reason": "none",
                    "content_type": "non_safe",
                }
            )
        )
        plugin.game = SimpleNamespace(prepare_action=AsyncMock(return_value=prepared))
        plugin._perform_action = AsyncMock()
        plugin._send_text = AsyncMock()

        await plugin._perform_command_action(
            object(),
            "10001",
            "把旁边的血肉当做性玩具来自慰",
        )

        plugin._send_text.assert_not_awaited()
        plugin._perform_action.assert_awaited_once()
        self.assertEqual(plugin._perform_action.await_args.args[4], "non_safe")
        self.assertIs(plugin._perform_action.await_args.kwargs["prepared"], prepared)
        plugin.busy.finish(room.room_id)

    def test_only_supported_unreasonable_reasons_reject_actions(self) -> None:
        classify = self.module.AIInteractiveFictionPlugin._route_action_is_reasonable
        self.assertTrue(classify({"reasonable": False, "unreasonable_reason": "none"}))
        self.assertFalse(classify({"reasonable": True, "unreasonable_reason": "physically_impossible"}))
        self.assertFalse(classify({"reasonable": True, "unreasonable_reason": "claims_result"}))
        self.assertFalse(classify({"reasonable": False}))

    def test_roundtable_action_trigger_categories_are_exclusive_except_non_safe(self) -> None:
        classify = self.module.AIInteractiveFictionPlugin._roundtable_action_triggers
        self.assertEqual(classify({}, "regular"), {"normal_action"})
        self.assertEqual(
            classify({"action_level": "high_risk_complex"}, "regular"),
            {"high_risk_complex_action"},
        )
        self.assertEqual(
            classify({"requests_story_change": True}, "non_safe"),
            {"story_change_request", "non_safe"},
        )

    def test_roundtable_display_fields_are_chinese(self) -> None:
        plugin = object.__new__(self.module.AIInteractiveFictionPlugin)
        config = MagicMock()
        plugin.astrbot_config = config
        plugin.raw_config = {
            "roundtable_models": [
                {"name": "智谱", "role": "proposal", "content_type": "regular"},
                {"name": "评审", "role": "reviewer", "content_type": "non_safe"},
            ]
        }
        plugin._normalize_roundtable_display_fields()
        first, second = plugin.raw_config["roundtable_models"]
        self.assertEqual(first["display_name"], "智谱——提案——常规")
        self.assertEqual(second["display_name"], "评审——评审——非安全")
        config.save_config.assert_called_once_with(replace_config=plugin.raw_config)

    async def test_action_lock_rejects_later_action_with_message(self) -> None:
        from _astrbot_plugin_ai_interactive_fiction_integration.services.models import StoryRoom
        from _astrbot_plugin_ai_interactive_fiction_integration.services.storage import RoomBusyRegistry

        plugin = object.__new__(self.module.AIInteractiveFictionPlugin)
        plugin.busy = RoomBusyRegistry()
        plugin.busy.try_begin("room-1")
        room = StoryRoom(
            room_id="room-1",
            owner_id="10001",
            story_config={},
            bible={},
            members={},
            created_at=1,
            last_active_at=1,
        )
        event = object()
        with patch.object(plugin, "_send_text", AsyncMock()) as send:
            await plugin._perform_action(
                event,
                "10001",
                room,
                "前进",
                "regular",
                include_psychology=False,
            )
        send.assert_awaited_once_with(event, "已有玩家的行动正在处理中，请等待故事回复")

    async def test_start_sends_profile_environment_then_portrait(self) -> None:
        from _astrbot_plugin_ai_interactive_fiction_integration.services.config import WordLimits
        from _astrbot_plugin_ai_interactive_fiction_integration.services.game import BuiltStory
        from _astrbot_plugin_ai_interactive_fiction_integration.services.models import StoryRoom

        class Store:
            def __init__(self) -> None:
                self.lock = asyncio.Lock()
                self.rooms = {}
                self.player_rooms = {}
                self.rewound_users = {}
                self.pending_starts = {}
                self.save = AsyncMock()

            def room_for_user(self, user_id: str):
                room_id = self.player_rooms.get(user_id)
                return self.rooms.get(room_id) if room_id else None

        built = BuiltStory(
            story=SimpleNamespace(),
            bible={"world": "hidden"},
            public_profile={"name": "岚"},
            full_character={"public": {"name": "岚"}},
            opening_state="醒在房间",
            opening_choices=["观察", "等待", "前进"],
        )
        room = StoryRoom(
            room_id="room-1",
            owner_id="10001",
            story_config={},
            bible=built.bible,
            members={},
            created_at=1,
            last_active_at=1,
        )
        plugin = object.__new__(self.module.AIInteractiveFictionPlugin)
        plugin.config = SimpleNamespace(
            global_judge_provider_id="judge",
            word_limits=WordLimits(profile_chars=111, environment_chars=77),
        )
        plugin.store = Store()
        plugin._starting_users = set()
        plugin.judge = SimpleNamespace(
            classify_content=AsyncMock(return_value="regular"),
            render_public_profile=AsyncMock(return_value="个人信息"),
            render_opening_environment=AsyncMock(return_value="环境：月光照进房间"),
        )
        plugin.game = SimpleNamespace(
            build_story=AsyncMock(return_value=built),
            create_room=MagicMock(return_value=room),
        )
        event = SimpleNamespace(
            get_sender_name=lambda: "测试者",
            unified_msg_origin="origin",
        )
        sent: list[tuple[str, str]] = []

        async def send_text(_event, content: str) -> None:
            sent.append(("text", content))

        async def send_portrait(_event, _room, character_id: str, _profile, **_kwargs) -> None:
            sent.append(("image", character_id))

        plugin._send_text = send_text
        plugin._generate_initial_portrait = send_portrait
        plugin.config.image_generation_triggers = set()
        await plugin._create_game(
            event,
            "10001",
            None,
            "",
            forced_random=True,
        )
        self.assertEqual(
            sent,
            [
                ("text", "环境：月光照进房间"),
                ("text", "个人信息\n\n1. 观察\n2. 等待\n3. 前进\n4. 可自由使用“/故事 [行动]”或 @我 行动"),
            ],
        )
        plugin.judge.render_public_profile.assert_awaited_once_with(
            built.public_profile,
            max_chars=111,
        )
        plugin.judge.render_opening_environment.assert_awaited_once_with(
            bible=built.bible,
            opening_state=built.opening_state,
            profile=built.public_profile,
            max_chars=77,
        )


if __name__ == "__main__":
    unittest.main()
