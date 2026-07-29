from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from services.config import (
    DEFAULT_GLOBAL_JUDGE_PERSONA,
    DEFAULT_NON_SAFE_ROUNDTABLE_PERSONA,
    DEFAULT_REGULAR_ROUNDTABLE_PERSONA,
    LEGACY_COMBINED_ROUNDTABLE_PERSONA,
    ImageGeneratorConfig,
    PluginConfig,
    RoundtableModelConfig,
    StoryConfig,
    WorkflowNodeMapping,
)
from services.game import GameService
from services.images import ImageService
from services.llm import GlobalJudge
from services.memory import MemoryService
from services.models import RoomMember, StoryRoom
from services.prompts import action_task
from services.roundtable import (
    RoundtableConfigurationError,
    RoundtableOutput,
    RoundtableService,
)
from services.saves import SaveService
from services.storage import RoomBusyRegistry, StateStore
from services.workflows import merge_workflow, set_dot_path


ROOT = Path(__file__).resolve().parent


def make_room(*, members: tuple[str, ...] = ("owner",), turn: int = 0) -> StoryRoom:
    return StoryRoom(
        room_id="room-1",
        owner_id="owner",
        story_config=StoryConfig(story_id="s", name="test", enabled=True, mechanisms={"save"}).to_runtime_dict(),
        bible={"world": "hidden"},
        members={
            user_id: RoomMember(user_id, user_id, {"public": {"name": user_id}}, turn, "origin")
            for user_id in members
        },
        created_at=1.0,
        last_active_at=1.0,
        turn=turn,
    )


class ConfigTests(unittest.TestCase):
    def test_final_story_fields_and_defaults(self) -> None:
        cfg = PluginConfig(
            {
                "stories": [
                    {
                        "name": "A",
                        "enabled": True,
                        "mechanisms": ["存档", "一条命"],
                        "memory_mode": "compressed",
                        "compression_provider_id": "compressor",
                    }
                ]
            }
        )
        story = cfg.enabled_stories()[0]
        self.assertTrue(story.save_enabled)
        self.assertEqual(story.memory_mode, "compressed")
        self.assertEqual(story.compression_provider_id, "compressor")
        self.assertEqual(story.mechanisms, {"save"})

    def test_template_key_parses_image_kind(self) -> None:
        cfg = PluginConfig(
            {"image_generators": [{"__template_key": "comfyui", "name": "wf", "enabled": True}]}
        )
        self.assertEqual(cfg.image_generators[0].kind, "comfyui")

    def test_roundtable_defaults_and_explicit_disable(self) -> None:
        cfg = PluginConfig(
            {
                "roundtable_models": [
                    {"name": "unsafe", "role": "reviewer", "content_type": "non_safe"},
                    {"name": "off", "enabled": False},
                ],
                "global_judge_persona": "",
            }
        )
        unsafe, disabled = cfg.roundtable_models
        self.assertTrue(unsafe.enabled)
        self.assertIn("最终评审", unsafe.persona)
        self.assertIn("暴力", unsafe.persona)
        self.assertIn("血腥", unsafe.persona)
        self.assertIn("性内容", unsafe.persona)
        self.assertFalse(disabled.enabled)
        self.assertEqual(cfg.global_judge_persona, DEFAULT_GLOBAL_JUDGE_PERSONA)

    def test_roundtable_uses_persona_for_selected_content_type(self) -> None:
        regular = PluginConfig(
            {
                "roundtable_models": [
                    {
                        "content_type": "regular",
                        "regular_persona": "regular-custom",
                        "non_safe_persona": "non-safe-custom",
                    }
                ]
            }
        ).roundtable_models[0]
        non_safe = PluginConfig(
            {
                "roundtable_models": [
                    {
                        "content_type": "non_safe",
                        "regular_persona": "regular-custom",
                        "non_safe_persona": "non-safe-custom",
                    }
                ]
            }
        ).roundtable_models[0]
        self.assertEqual(regular.persona, "regular-custom")
        self.assertEqual(non_safe.persona, "non-safe-custom")

    def test_custom_v002_persona_is_preserved_during_migration(self) -> None:
        custom = PluginConfig(
            {
                "roundtable_models": [
                    {
                        "content_type": "non_safe",
                        "persona": "legacy-custom",
                        "non_safe_persona": DEFAULT_NON_SAFE_ROUNDTABLE_PERSONA,
                    }
                ]
            }
        ).roundtable_models[0]
        old_default = PluginConfig(
            {
                "roundtable_models": [
                    {
                        "content_type": "regular",
                        "persona": LEGACY_COMBINED_ROUNDTABLE_PERSONA,
                        "regular_persona": DEFAULT_REGULAR_ROUNDTABLE_PERSONA,
                    }
                ]
            }
        ).roundtable_models[0]
        self.assertEqual(custom.persona, "legacy-custom")
        self.assertEqual(old_default.persona, DEFAULT_REGULAR_ROUNDTABLE_PERSONA)

    def test_configuration_schema_defaults_and_item_subtitles(self) -> None:
        schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        templates = {
            "story": schema["stories"]["templates"]["story"],
            "roundtable": schema["roundtable_models"]["templates"]["model"],
            "openai": schema["image_generators"]["templates"]["openai"],
            "comfyui": schema["image_generators"]["templates"]["comfyui"],
            "mapping": schema["workflow_node_mappings"]["templates"]["mapping"],
        }
        for template in templates.values():
            self.assertEqual(template["display_item"], "name")
            self.assertTrue(template["hide_hint_in_list"])
            self.assertEqual(template["items"]["name"]["type"], "string")
        roundtable_items = templates["roundtable"]["items"]
        self.assertTrue(roundtable_items["enabled"]["default"])
        self.assertEqual(
            roundtable_items["regular_persona"]["default"],
            DEFAULT_REGULAR_ROUNDTABLE_PERSONA,
        )
        self.assertEqual(
            roundtable_items["non_safe_persona"]["default"],
            DEFAULT_NON_SAFE_ROUNDTABLE_PERSONA,
        )
        self.assertEqual(
            roundtable_items["regular_persona"]["condition"],
            {"content_type": "regular"},
        )
        self.assertEqual(
            roundtable_items["non_safe_persona"]["condition"],
            {"content_type": "non_safe"},
        )
        self.assertTrue(roundtable_items["persona"]["invisible"])
        item_order = list(roundtable_items)
        self.assertLess(item_order.index("provider_id"), item_order.index("content_type"))
        self.assertLess(item_order.index("content_type"), item_order.index("regular_persona"))
        self.assertEqual(
            schema["image_generators"]["hint"],
            "人物首次生成使用文生图，后续CG使用缓存图改图以确保人物一致性；按优先级从高到低失败切换模型",
        )
        self.assertEqual(schema["global_judge_persona"]["default"], DEFAULT_GLOBAL_JUDGE_PERSONA)

    def test_natural_slot_routing_preserves_out_of_range_numbers_for_validation(self) -> None:
        self.assertIn("即使超出1至4", GlobalJudge.ROUTE_SYSTEM)


class LockTests(unittest.TestCase):
    def test_room_busy_is_non_blocking(self) -> None:
        lock = RoomBusyRegistry()
        self.assertTrue(lock.try_begin("room"))
        self.assertFalse(lock.try_begin("room"))
        lock.finish("room")
        self.assertTrue(lock.try_begin("room"))


class SaveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = StateStore(Path(self.temp.name))
        self.service = SaveService(self.store)
        self.room = make_room()
        self.store.rooms[self.room.room_id] = self.room
        self.store.player_rooms["owner"] = self.room.room_id

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_natural_save_fills_empty_then_oldest(self) -> None:
        for expected in ("1", "2", "3", "4"):
            self.assertEqual(self.service.save_manual("owner", self.room).slot, expected)
        self.store.saves["owner"]["1"].created_at = 1
        self.store.saves["owner"]["2"].created_at = 2
        self.assertEqual(self.service.save_manual("owner", self.room).slot, "1")

    def test_load_removes_players_created_after_snapshot(self) -> None:
        self.service.save_manual("owner", self.room, 1)
        self.room.members["late"] = RoomMember("late", "late", {}, 1, "origin")
        self.store.player_rooms["late"] = self.room.room_id
        self.room.last_active_at = 99
        restored, removed = self.service.load("owner", self.room, 1)
        self.assertIsNotNone(restored)
        self.assertEqual(removed, ["late"])
        self.assertNotIn("late", restored.members)
        self.assertEqual(self.store.rewound_users["late"], self.room.room_id)
        self.assertEqual(restored.last_active_at, 99)

    def test_latest_load_includes_auto_slot(self) -> None:
        manual = self.service.save_manual("owner", self.room, 1)
        manual.created_at = 1
        self.room.turn = 2
        self.service.auto_save(self.room)
        restored, _ = self.service.load("owner", self.room)
        self.assertEqual(restored.turn, 2)

    def test_ending_room_clears_rewind_marker_and_saves(self) -> None:
        self.service.save_manual("owner", self.room, 1)
        self.store.rewound_users["late"] = self.room.room_id
        self.store.end_room(self.room.room_id)
        self.assertNotIn("late", self.store.rewound_users)
        self.assertNotIn("owner", self.store.saves)

    def test_cleanup_expired_uses_room_last_activity(self) -> None:
        self.room.last_active_at = 100
        self.assertEqual(self.store.cleanup_expired(7, now=100 + 7 * 86400 - 1), [])
        self.assertEqual(self.store.cleanup_expired(7, now=100 + 7 * 86400 + 1), ["room-1"])
        self.assertNotIn("owner", self.store.player_rooms)


class WorkflowTests(unittest.TestCase):
    def test_dot_path_and_workflow_mapping(self) -> None:
        data = {"inputs": {"texts": ["old"]}}
        set_dot_path(data, "inputs.texts.0", "new")
        self.assertEqual(data["inputs"]["texts"][0], "new")

        generator = ImageGeneratorConfig(
            kind="comfyui",
            name="wf",
            enabled=True,
            priority=1,
            prompt_provider_id="p",
            raw={"workflow_content": json.dumps({"6": {"inputs": {"text": "old"}}})},
        )
        mapping = WorkflowNodeMapping("prompt", "wf", "6", "inputs.text", "positive_prompt")
        merged = merge_workflow(generator, [mapping], prompt="new")
        self.assertEqual(merged["6"]["inputs"]["text"], "new")

    def test_image_candidates_route_non_safe_edits_to_free_or_comfyui(self) -> None:
        generators = [
            ImageGeneratorConfig("openai", "regular", True, 90, "p", {"content_type": "regular"}),
            ImageGeneratorConfig("openai", "free", True, 80, "p", {"content_type": "free"}),
            ImageGeneratorConfig("comfyui", "local", True, 70, "p", {"support_mode": "edit"}),
        ]
        service = ImageService(FakeLLM(), generators, [], default_timeout=300, logger=Logger())
        names = [item.name for item in service._candidates(mode="edit", non_safe=True)]
        self.assertEqual(names, ["free", "local"])


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.prompts: list[str] = []
        self.system_prompts: list[str] = []

    async def generate(self, provider_id: str, prompt: str, **kwargs) -> str:
        self.calls.append(provider_id)
        self.prompts.append(prompt)
        self.system_prompts.append(str(kwargs.get("system_prompt") or ""))
        return f"output-{provider_id}"

    async def describe_provider(self, provider_id: str):
        class Description:
            label = provider_id + "＋model"
        return Description()


class FailingProposalLLM(FakeLLM):
    async def generate(self, provider_id: str, prompt: str, **kwargs) -> str:
        self.calls.append(provider_id)
        self.prompts.append(prompt)
        self.system_prompts.append(str(kwargs.get("system_prompt") or ""))
        if provider_id.startswith("p"):
            raise RuntimeError("failed")
        return f"output-{provider_id}"


class Logger:
    def warning(self, *_args, **_kwargs) -> None:
        return None


class FakeRoundtable:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    async def run(self, task: str, **kwargs) -> RoundtableOutput:
        self.calls.append({"task": task, **kwargs})
        return RoundtableOutput(json.dumps(self.payload, ensure_ascii=False), [])


class GameTests(unittest.IsolatedAsyncioTestCase):
    async def test_random_story_runtime_fields_are_built_from_roundtable(self) -> None:
        roundtable = FakeRoundtable(
            {
                "story_bible": {"title": "随机故事"},
                "public_player_profile": {"name": "玩家"},
                "private_player_profile": {"secret": "x"},
                "opening_state": "醒来",
                "runtime": {"expected_minutes": 26, "save_enabled": True},
            }
        )
        game = GameService(roundtable, object())
        built = await game.build_story(
            None,
            requirements="主角是女性",
            owner_name="测试者",
            content_type="regular",
        )
        self.assertEqual(built.story.name, "随机故事")
        self.assertEqual(built.story.expected_minutes, 26)
        self.assertTrue(built.story.save_enabled)
        self.assertEqual(built.opening_state, "醒来")
        self.assertIn("主角是女性", roundtable.calls[0]["task"])


class RoundtableTests(unittest.IsolatedAsyncioTestCase):
    async def test_high_priority_runs_last_and_last_reviewer_wins(self) -> None:
        models = [
            RoundtableModelConfig("p-high", True, 90, "proposal", "regular", "p-high", "", -1),
            RoundtableModelConfig("p-low", True, 10, "proposal", "regular", "p-low", "", -1),
            RoundtableModelConfig("r-high", True, 90, "reviewer", "regular", "r-high", "", -1),
            RoundtableModelConfig("r-low", True, 10, "reviewer", "regular", "r-low", "", -1),
        ]
        llm = FakeLLM()
        service = RoundtableService(llm, models, mode="sequential", rounds=1, logger=Logger())
        with patch("services.roundtable.random.shuffle", side_effect=lambda value: None):
            output = await service.run("task")
        self.assertEqual(llm.calls, ["p-low", "p-high", "r-low", "r-high"])
        self.assertEqual(output.final_text, "output-r-high")
        self.assertTrue(all(prompt.startswith("task") for prompt in llm.prompts))

    async def test_non_safe_falls_back_to_regular_roles(self) -> None:
        models = [
            RoundtableModelConfig("p", True, 1, "proposal", "regular", "p", "", -1),
            RoundtableModelConfig("r", True, 1, "reviewer", "regular", "r", "", -1),
        ]
        output = await RoundtableService(FakeLLM(), models, mode="sequential", rounds=1, logger=Logger()).run(
            "task", content_type="non_safe"
        )
        self.assertEqual(output.final_text, "output-r")

    async def test_all_proposals_failing_stops_later_rounds(self) -> None:
        models = [
            RoundtableModelConfig("p1", True, 1, "proposal", "regular", "p1", "", -1),
            RoundtableModelConfig("p2", True, 2, "proposal", "regular", "p2", "", -1),
            RoundtableModelConfig("r", True, 1, "reviewer", "regular", "r", "", -1),
        ]
        llm = FailingProposalLLM()
        output = await RoundtableService(llm, models, mode="sequential", rounds=2, logger=Logger()).run("task")
        self.assertEqual(llm.calls, ["p1", "p2", "r"])
        self.assertEqual(output.final_text, "output-r")

    async def test_missing_role_configuration_is_reported(self) -> None:
        models = [RoundtableModelConfig("p", True, 1, "proposal", "regular", "p", "", -1)]
        with self.assertRaisesRegex(RoundtableConfigurationError, "缺少常规/非安全模型配置"):
            await RoundtableService(FakeLLM(), models, mode="sequential", rounds=1, logger=Logger()).run("task")

    async def test_empty_legacy_persona_gets_non_safe_role_fallback(self) -> None:
        models = [
            RoundtableModelConfig("p", True, 1, "proposal", "non_safe", "p", "", -1),
            RoundtableModelConfig("r", True, 1, "reviewer", "non_safe", "r", "", -1),
        ]
        llm = FakeLLM()
        await RoundtableService(llm, models, mode="sequential", rounds=1, logger=Logger()).run(
            "task", content_type="non_safe"
        )
        self.assertIn("暴力", llm.system_prompts[0])
        self.assertIn("性内容", llm.system_prompts[0])
        self.assertIn("最终评审", llm.system_prompts[-1])


class MemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_long_memory_compression_keeps_latest_turn(self) -> None:
        room = make_room(turn=3)
        room.history = [
            {"turn": 1, "result": "a"},
            {"turn": 2, "result": "b"},
            {"turn": 3, "result": "c"},
        ]
        story = StoryConfig(
            story_id="s",
            name="n",
            enabled=True,
            memory_mode="compressed",
            compress_after_turns=2,
            compression_provider_id="compressor",
        )
        service = MemoryService(FakeLLM(), Logger())
        self.assertTrue(await service.compress_if_needed(room, story))
        self.assertEqual(room.memory_summary, "output-compressor")
        self.assertEqual(room.history, [{"turn": 3, "result": "c"}])


class PromptTests(unittest.TestCase):
    def test_content_limit_is_action_prompt_prefix_only(self) -> None:
        story = StoryConfig(story_id="s", name="n", enabled=True, content_limit="LIMIT")
        prompt = action_task(
            story=story,
            bible={},
            world_state="",
            memory_context="",
            characters={},
            actor_id="u",
            action="move",
            content_type="regular",
            forbid_player_autonomy=True,
            current_choices=[],
            include_psychology=False,
        )
        self.assertTrue(prompt.startswith("LIMIT"))

    def test_action_result_keeps_choices_and_string_booleans(self) -> None:
        output = RoundtableOutput(
            final_text=json.dumps(
                {
                    "narrative": "结果",
                    "choices": ["前进", "返回"],
                    "death": "true",
                    "story_ended": "true",
                    "major_node": "true",
                },
                ensure_ascii=False,
            ),
            discussion=[],
        )
        parsed = GameService._parse_action(output)
        self.assertEqual(parsed.choices, ["前进", "返回"])
        self.assertTrue(parsed.death)
        self.assertFalse(parsed.story_ended)
        self.assertTrue(parsed.major_node)


if __name__ == "__main__":
    unittest.main()
