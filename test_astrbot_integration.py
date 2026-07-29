from __future__ import annotations

import importlib
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


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

        expected = {"故事", "存档", "读档", "圆桌会议"}
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
        self.assertEqual(parse("/存档 5"), {"name": "save", "slot": "5"})
        self.assertEqual(parse("/读档 1"), {"name": "load", "slot": "1"})
        self.assertEqual(parse("/圆桌会议"), {"name": "roundtable"})

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

    async def test_llm_tool_respects_natural_language_switch(self) -> None:
        plugin = object.__new__(self.module.AIInteractiveFictionPlugin)
        plugin.store = object()
        plugin.saves = object()
        plugin.config = SimpleNamespace(enable_natural_language=False)
        output = [item async for item in plugin.interactive_fiction_tool(object(), "开始故事")]
        self.assertEqual(output, ["自然语言互动未启用，请使用 /故事 查看指令。"])

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


if __name__ == "__main__":
    unittest.main()
