from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from services.config import (
    DEFAULT_CONTENT_LIMIT,
    DEFAULT_GLOBAL_STORY_PERSONA,
    DEFAULT_GLOBAL_JUDGE_PERSONA,
    DEFAULT_NON_SAFE_ROUNDTABLE_PERSONA,
    DEFAULT_REGULAR_ROUNDTABLE_PERSONA,
    LEGACY_DEFAULT_CONTENT_LIMIT,
    LEGACY_COMBINED_ROUNDTABLE_PERSONA,
    ImageGeneratorConfig,
    PluginConfig,
    RoundtableModelConfig,
    StoryConfig,
    WordLimits,
    WorkflowNodeMapping,
)
from services.game import GameService, _choice_list, _is_action_payload
from services.images import ImageService, apply_art_style
from services.llm import GlobalJudge, LLMService, parse_json_object
from services.memory import MemoryService
from services.models import RoomMember, StoryRoom
from services.prompts import action_task
from services.roundtable import (
    RoundtableConfigurationError,
    RoundtableGenerationError,
    RoundtableOutput,
    RoundtableService,
)
from services.story_generator import StoryGenerationRequest, StoryGeneratorService
from services.saves import SaveService
from services.storage import RoomBusyRegistry, StateStore
from services.workflows import ImageGenerationError, merge_workflow, set_dot_path


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

    def test_word_limits_and_legacy_story_limit_are_migrated(self) -> None:
        cfg = PluginConfig(
            {
                "regular_content_chars": 41,
                "non_safe_content_chars": 260,
                "profile_chars": 130,
                "environment_chars": 90,
                "psychology_chars": 45,
                "stories": [{"name": "旧配置", "content_limit": LEGACY_DEFAULT_CONTENT_LIMIT}],
            }
        )
        self.assertEqual(
            cfg.word_limits,
            WordLimits(
                regular_content_chars=41,
                non_safe_content_chars=260,
                profile_chars=130,
                environment_chars=90,
                psychology_chars=45,
            ),
        )
        self.assertEqual(cfg.stories[0].content_limit, DEFAULT_CONTENT_LIMIT)

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

    def test_global_story_defaults_and_roundtable_triggers(self) -> None:
        cfg = PluginConfig({})
        self.assertEqual(cfg.global_story_persona, DEFAULT_GLOBAL_STORY_PERSONA)
        self.assertEqual(cfg.roundtable_triggers, {"non_safe", "draft"})
        custom = PluginConfig(
            {
                "global_story_provider_id": "story-model",
                "global_story_persona": "custom",
                "roundtable_triggers": ["normal_action", "unknown"],
            }
        )
        self.assertEqual(custom.global_story_provider_id, "story-model")
        self.assertEqual(custom.global_story_persona, "custom")
        self.assertEqual(custom.roundtable_triggers, {"normal_action"})

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
        for key, template in templates.items():
            if key == "roundtable":
                self.assertEqual(template["display_item"], "display_name")
                self.assertEqual(template["items"]["display_name"]["type"], "string")
                self.assertTrue(template["items"]["display_name"]["invisible"])
            else:
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
        self.assertEqual(schema["global_story_persona"]["default"], DEFAULT_GLOBAL_STORY_PERSONA)
        self.assertEqual(schema["roundtable_triggers"]["default"], ["non_safe", "draft"])
        self.assertEqual(
            list(schema)[list(schema).index("stories") + 1 : list(schema).index("roundtable_models")],
            ["global_story_provider_id", "global_story_persona", "roundtable_triggers"],
        )
        self.assertEqual(
            schema["image_generation_triggers"]["default"],
            [
                "first_appearance",
                "first_conversation",
                "killing",
                "violation",
                "scene_change",
                "battle_damage",
            ],
        )
        self.assertEqual(schema["stories"]["templates"]["story"]["items"]["content_limit"]["default"], DEFAULT_CONTENT_LIMIT)
        root_order = list(schema)
        self.assertEqual(root_order[root_order.index("streaming") + 1], "word_limits_panel")
        for key, default in {
            "regular_content_chars": 30,
            "non_safe_content_chars": 200,
            "profile_chars": 120,
            "environment_chars": 80,
            "psychology_chars": 50,
        }.items():
            self.assertEqual(schema[key]["default"], default)
            self.assertEqual(schema[key]["condition"], {"word_limits_panel": "expanded"})
        for image_template in (templates["openai"], templates["comfyui"]):
            image_items = list(image_template["items"])
            self.assertEqual(image_items[image_items.index("prompt_provider_id") + 1], "style")

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

    def test_load_does_not_restore_departed_player_state(self) -> None:
        self.room.members["departed"] = RoomMember("departed", "departed", {}, 0, "origin")
        self.room.dead_users.append("departed")
        self.room.portraits["departed"] = {"path": "old.png", "prompt": "old"}
        self.room.conversation_character_id = "departed"
        self.store.player_rooms["departed"] = self.room.room_id
        self.service.save_manual("owner", self.room, 1)
        self.room.members.pop("departed")
        self.store.player_rooms.pop("departed")

        restored, removed = self.service.load("owner", self.room, 1)

        self.assertEqual(removed, [])
        self.assertNotIn("departed", restored.members)
        self.assertNotIn("departed", restored.dead_users)
        self.assertNotIn("departed", restored.portraits)
        self.assertEqual(restored.conversation_character_id, "")

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


class StorageTests(unittest.TestCase):
    def test_load_skips_malformed_rooms_and_save_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            good_room = make_room()
            payload = {
                "rooms": {
                    "room-1": good_room.to_dict(),
                    "broken": {**good_room.to_dict(), "turn": "not-a-number"},
                    "ownerless": {**good_room.to_dict(), "owner_id": "missing"},
                },
                "saves": {
                    "owner": {
                        "1": {
                            "room_id": "room-1",
                            "user_id": "owner",
                            "slot": "1",
                            "created_at": "bad-time",
                            "snapshot": {},
                        }
                    }
                },
                "rewound_users": {"ghost": "missing-room"},
            }
            (data_dir / "state.json").write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            store = StateStore(data_dir)

            store._load_sync()

            self.assertEqual(set(store.rooms), {"room-1"})
            self.assertEqual(store.player_rooms, {"owner": "room-1"})
            self.assertEqual(store.saves, {})
            self.assertEqual(store.rewound_users, {})


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

    def test_workflow_requires_prompt_and_edit_image_mappings(self) -> None:
        generator = ImageGeneratorConfig(
            kind="comfyui",
            name="wf",
            enabled=True,
            priority=1,
            prompt_provider_id="p",
            raw={"workflow_content": json.dumps({"6": {"inputs": {"text": "old"}}})},
        )
        with self.assertRaisesRegex(ImageGenerationError, "正向提示词"):
            merge_workflow(generator, [], prompt="new")
        prompt_mapping = WorkflowNodeMapping("prompt", "wf", "6", "inputs.text", "positive_prompt")
        with self.assertRaisesRegex(ImageGenerationError, "图像输入"):
            merge_workflow(generator, [prompt_mapping], prompt="new", uploaded_image="input.png")

    def test_image_candidates_route_non_safe_edits_to_free_or_comfyui(self) -> None:
        generators = [
            ImageGeneratorConfig("openai", "regular", True, 90, "p", {"content_type": "regular"}),
            ImageGeneratorConfig("openai", "free", True, 80, "p", {"content_type": "free"}),
            ImageGeneratorConfig("comfyui", "local", True, 70, "p", {"support_mode": "edit"}),
        ]
        service = ImageService(FakeLLM(), generators, [], default_timeout=300, logger=Logger())
        names = [item.name for item in service._candidates(mode="edit", non_safe=True)]
        self.assertEqual(names, ["free", "local"])

    def test_art_style_prefix_is_normalized(self) -> None:
        self.assertEqual(apply_art_style("一个穿着JK制服的女生", "日系二次元"), "日系二次元画风，一个穿着JK制服的女生")
        self.assertEqual(apply_art_style("厚涂画风，持剑角色", "厚涂画风"), "厚涂画风，持剑角色")
        self.assertEqual(apply_art_style("角色立绘", ""), "角色立绘")


class ImageTests(unittest.IsolatedAsyncioTestCase):
    async def test_configured_style_reaches_every_final_image_prompt(self) -> None:
        for kind in ("openai", "comfyui"):
            with self.subTest(kind=kind):
                llm = FakeLLM()
                raw = {"content_type": "regular", "style": "日系二次元"}
                if kind == "comfyui":
                    raw["support_mode"] = "generate"
                generator = ImageGeneratorConfig(
                    kind,
                    "styled",
                    True,
                    50,
                    "prompt-model",
                    raw,
                )
                service = ImageService(llm, [generator], [], default_timeout=300, logger=Logger())
                runner = service.openai if kind == "openai" else service.comfyui
                runner.generate = unittest.mock.AsyncMock(return_value=Path("generated.png"))
                result = await service.generate_character_image(
                    bible={"tone": "青春"},
                    character={"appearance": "JK制服"},
                    output_dir=Path("unused"),
                    environment_context="月光照进旧教室",
                )
                self.assertEqual(
                    result.prompt,
                    "日系二次元画风，output-prompt-model，当前环境背景：月光照进旧教室",
                )
                self.assertIn("用户指定画风：日系二次元", llm.prompts[0])
                self.assertIn("月光照进旧教室", llm.prompts[0])
                self.assertEqual(runner.generate.await_args.kwargs["prompt"], result.prompt)

    async def test_scene_image_uses_scene_prompt_and_generate_model(self) -> None:
        llm = FakeLLM()
        generator = ImageGeneratorConfig(
            "openai",
            "scene",
            True,
            50,
            "prompt-model",
            {"content_type": "regular", "style": "电影感", "api_keys": ["test"]},
        )
        service = ImageService(llm, [generator], [], default_timeout=300, logger=Logger())
        service.openai.generate = unittest.mock.AsyncMock(return_value=Path("scene.png"))
        result = await service.generate_scene_image(
            bible={"tone": "悬疑"},
            event_context="你从走廊进入大厅",
            output_dir=Path("unused"),
        )
        self.assertEqual(result.path, Path("scene.png"))
        self.assertIn("环境全景", llm.prompts[0])
        self.assertIn("你从走廊进入大厅", llm.prompts[0])


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


class StoryGeneratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_direct_result_is_used_when_no_trigger_matches(self) -> None:
        llm = FakeLLM()
        roundtable = unittest.mock.AsyncMock()
        service = StoryGeneratorService(
            llm,
            roundtable,
            provider_id="story",
            persona="persona",
            roundtable_triggers={"draft"},
            logger=Logger(),
        )
        output = await service.generate(
            StoryGenerationRequest("task"),
            trigger_types={"normal_action"},
        )
        self.assertEqual(output.final_text, "output-story")
        self.assertEqual(output.discussion, [])
        roundtable.run.assert_not_awaited()

    async def test_matching_trigger_sends_global_draft_to_roundtable(self) -> None:
        llm = FakeLLM()
        roundtable = unittest.mock.AsyncMock()
        roundtable.run.return_value = RoundtableOutput(
            "roundtable-final",
            [{"label": "reviewer", "content": "roundtable-final"}],
        )
        service = StoryGeneratorService(
            llm,
            roundtable,
            provider_id="story",
            persona="persona",
            roundtable_triggers={"non_safe"},
            logger=Logger(),
        )
        output = await service.generate(
            StoryGenerationRequest("task", content_type="non_safe"),
            trigger_types={"non_safe", "normal_action"},
        )
        self.assertEqual(output.final_text, "roundtable-final")
        self.assertEqual(roundtable.run.await_args.kwargs["initial_draft"], "output-story")
        self.assertEqual(roundtable.run.await_args.kwargs["content_type"], "non_safe")

    async def test_missing_global_story_provider_falls_back_to_roundtable(self) -> None:
        roundtable = unittest.mock.AsyncMock()
        roundtable.run.return_value = RoundtableOutput("fallback", [])
        service = StoryGeneratorService(
            FakeLLM(),
            roundtable,
            provider_id="",
            persona="persona",
            roundtable_triggers=set(),
            logger=Logger(),
        )
        output = await service.generate(
            StoryGenerationRequest("task"),
            trigger_types={"normal_action"},
        )
        self.assertEqual(output.final_text, "fallback")
        self.assertEqual(roundtable.run.await_args.kwargs["initial_draft"], "")


class JudgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_opening_render_prompts_apply_visible_character_limits(self) -> None:
        llm = FakeLLM()
        judge = GlobalJudge(llm, "judge", "custom persona")
        await judge.render_public_profile({"name": "岚"}, max_chars=111)
        await judge.render_opening_environment(
            bible={"secret": "hidden"},
            opening_state="醒在房间",
            profile={"name": "岚"},
            max_chars=77,
        )
        self.assertIn("不超过111字", llm.prompts[0])
        self.assertIn("不超过77字", llm.prompts[1])
        self.assertIn("视野所及", llm.prompts[1])
        self.assertIn("不得透露视野外NPC", llm.prompts[1])
        self.assertIn("不加“环境：”前缀", llm.prompts[1])
        self.assertNotIn("只输出JSON", llm.system_prompts[0])

    async def test_image_trigger_detection_filters_unconfigured_types(self) -> None:
        class TriggerLLM(FakeLLM):
            async def generate(self, provider_id: str, prompt: str, **kwargs) -> str:
                self.prompts.append(prompt)
                return json.dumps(
                    {
                        "triggers": [
                            {"type": "scene_change", "character_id": "", "description": "进入大厅"},
                            {"type": "killing", "character_id": "npc", "description": "不应保留"},
                        ]
                    },
                    ensure_ascii=False,
                )

        llm = TriggerLLM()
        judge = GlobalJudge(llm, "judge")
        triggers = await judge.detect_image_triggers(
            action="进入大厅",
            narrative="你推门进入大厅",
            known_characters={},
            new_characters=[],
            changed_characters=[],
            enabled_triggers={"scene_change"},
        )
        self.assertEqual(
            triggers,
            [{"type": "scene_change", "character_id": "", "description": "进入大厅"}],
        )
        self.assertIn("场景变换", llm.prompts[0])

    def test_route_prompt_defines_roundtable_action_classification(self) -> None:
        self.assertIn('"action_level":"normal|high_risk_complex"', GlobalJudge.ROUTE_SYSTEM)
        self.assertIn('"requests_story_change":false', GlobalJudge.ROUTE_SYSTEM)


class StructuredOutputTests(unittest.IsolatedAsyncioTestCase):
    def test_json_parser_extracts_largest_balanced_object(self) -> None:
        parsed = parse_json_object(
            '说明 {"status":"draft"}\n最终结果：'
            '{"story_bible":{"title":"测试"},"public_player_profile":{"name":"玩家"}}\n结束'
        )
        self.assertEqual(parsed["story_bible"]["title"], "测试")
        self.assertEqual(parsed["public_player_profile"]["name"], "玩家")

    async def test_stream_accepts_cumulative_chunks_and_trailing_delta(self) -> None:
        class Response:
            def __init__(self, text: str, is_chunk: bool) -> None:
                self.completion_text = text
                self.is_chunk = is_chunk

        class Provider:
            async def text_chat_stream(self, **_kwargs):
                yield Response('{"story_bible"', True)
                yield Response('{"story_bible":{}', True)
                yield Response(',"public_player_profile":{}}', False)

        class ProviderManager:
            async def get_provider_by_id(self, _provider_id: str):
                return Provider()

        class Context:
            provider_manager = ProviderManager()

        service = LLMService(Context(), streaming=True, default_timeout=30, logger=Logger())
        output = await service._generate_stream("provider", "prompt", "", {})
        self.assertEqual(
            parse_json_object(output),
            {"story_bible": {}, "public_player_profile": {}},
        )


class FakeGenerator:
    def __init__(self, payload: dict, discussion: list[dict[str, str]] | None = None) -> None:
        self.payload = payload
        self.discussion = discussion or []
        self.calls: list[dict] = []

    async def generate(self, request: StoryGenerationRequest, **kwargs) -> RoundtableOutput:
        self.calls.append({"task": request.task, **kwargs})
        return RoundtableOutput(json.dumps(self.payload, ensure_ascii=False), self.discussion)


class GameTests(unittest.IsolatedAsyncioTestCase):
    async def test_random_story_runtime_fields_are_built_from_roundtable(self) -> None:
        discussion = [{"label": "model", "content": "proposal"}]
        generator = FakeGenerator(
            {
                "story_bible": {"title": "随机故事"},
                "public_player_profile": {"name": "玩家"},
                "private_player_profile": {"secret": "x"},
                "opening_state": "醒来",
                "opening_choices": ["观察", "等待", "前进"],
                "runtime": {"expected_minutes": 26, "save_enabled": True},
            },
            discussion,
        )
        game = GameService(generator, object())
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
        self.assertEqual(built.opening_choices, ["观察", "等待", "前进"])
        self.assertEqual(built.discussion, discussion)
        room = game.create_room("owner", "玩家", "origin", built)
        self.assertEqual(room.latest_discussion, discussion)
        self.assertIn("主角是女性", generator.calls[0]["task"])

    async def test_join_character_keeps_roundtable_discussion(self) -> None:
        discussion = [{"label": "reviewer", "content": "character"}]
        generator = FakeGenerator(
            {
                "public_player_profile": {"name": "新玩家"},
                "private_player_profile": {"secret": "x"},
            },
            discussion,
        )
        game = GameService(generator, object())
        room = make_room()

        built = await game.build_join_character(
            room,
            requirements="",
            player_name="新玩家",
            content_type="regular",
        )

        self.assertEqual(built.discussion, discussion)


class RoundtableTests(unittest.IsolatedAsyncioTestCase):
    async def test_sequential_roundtable_starts_from_global_story_draft(self) -> None:
        models = [
            RoundtableModelConfig("p", True, 1, "proposal", "regular", "p", "", -1),
            RoundtableModelConfig("r", True, 1, "reviewer", "regular", "r", "", -1),
        ]
        llm = FakeLLM()
        output = await RoundtableService(
            llm,
            models,
            mode="sequential",
            rounds=1,
            logger=Logger(),
        ).run("task", initial_draft="global-draft")
        self.assertIn("全局故事生成LLM底稿：global-draft", llm.prompts[0])
        self.assertNotIn(
            "global-draft",
            "\n".join(item["content"] for item in output.discussion),
        )

    async def test_independent_roundtable_only_gives_global_draft_to_reviewers(self) -> None:
        models = [
            RoundtableModelConfig("p", True, 1, "proposal", "regular", "p", "", -1),
            RoundtableModelConfig("r", True, 1, "reviewer", "regular", "r", "", -1),
        ]
        llm = FakeLLM()
        await RoundtableService(
            llm,
            models,
            mode="independent",
            rounds=1,
            logger=Logger(),
        ).run("task", initial_draft="global-draft")
        self.assertNotIn("global-draft", llm.prompts[0])
        self.assertIn("全局故事生成LLM底稿：global-draft", llm.prompts[1])

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
        with self.assertRaisesRegex(RoundtableGenerationError, "生成失败"):
            await RoundtableService(llm, models, mode="sequential", rounds=2, logger=Logger()).run("task")
        self.assertEqual(llm.calls, ["p1", "p2"])

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

    async def test_invalid_final_reviewer_is_repaired_once(self) -> None:
        class RepairingLLM(FakeLLM):
            async def generate(self, provider_id: str, prompt: str, **kwargs) -> str:
                self.calls.append(provider_id)
                self.prompts.append(prompt)
                self.system_prompts.append(str(kwargs.get("system_prompt") or ""))
                if provider_id == "p":
                    return "proposal"
                if "上一次的输出未通过" in prompt:
                    self.repair_temperature = kwargs.get("temperature")
                    return '{"story_bible":{},"public_player_profile":{}}'
                return '{"story_bible":{}}'

        models = [
            RoundtableModelConfig("p", True, 1, "proposal", "regular", "p", "", -1),
            RoundtableModelConfig("r", True, 1, "reviewer", "regular", "r", "", -1),
        ]
        llm = RepairingLLM()
        service = RoundtableService(llm, models, mode="sequential", rounds=1, logger=Logger())
        validator = lambda text: bool(
            (parsed := parse_json_object(text))
            and isinstance(parsed.get("story_bible"), dict)
            and isinstance(parsed.get("public_player_profile"), dict)
        )
        output = await service.run(
            "task",
            output_validator=validator,
            repair_instruction="required schema",
        )
        self.assertTrue(validator(output.final_text))
        self.assertEqual(llm.calls, ["p", "r", "r"])
        self.assertEqual(llm.repair_temperature, 0.0)
        self.assertIn("结构要求：\nrequired schema", llm.prompts[-1])
        self.assertTrue(output.discussion[-1]["label"].endswith("（格式修复）"))

    async def test_failed_repair_falls_back_to_prior_valid_reviewer(self) -> None:
        class FallbackLLM(FakeLLM):
            async def generate(self, provider_id: str, prompt: str, **kwargs) -> str:
                self.calls.append(provider_id)
                self.prompts.append(prompt)
                if provider_id == "p":
                    return "proposal"
                if provider_id == "r-low":
                    return '{"required":true}'
                return '{"wrong":true}'

        models = [
            RoundtableModelConfig("p", True, 1, "proposal", "regular", "p", "", -1),
            RoundtableModelConfig("r-low", True, 10, "reviewer", "regular", "r-low", "", -1),
            RoundtableModelConfig("r-high", True, 90, "reviewer", "regular", "r-high", "", -1),
        ]
        llm = FallbackLLM()
        service = RoundtableService(llm, models, mode="sequential", rounds=1, logger=Logger())
        with patch("services.roundtable.random.shuffle", side_effect=lambda value: None):
            output = await service.run(
                "task",
                output_validator=lambda text: bool((parse_json_object(text) or {}).get("required")),
                repair_instruction="required schema",
            )
        self.assertEqual(output.final_text, '{"required":true}')
        self.assertEqual(llm.calls, ["p", "r-low", "r-high", "r-high"])


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
        self.assertIn("不超过30字", prompt)

    def test_action_prompt_uses_content_type_and_psychology_limits(self) -> None:
        story = StoryConfig(story_id="s", name="n", enabled=True)
        limits = WordLimits(
            regular_content_chars=36,
            non_safe_content_chars=240,
            profile_chars=100,
            environment_chars=70,
            psychology_chars=42,
        )
        regular = action_task(
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
            word_limits=limits,
        )
        non_safe = action_task(
            story=story,
            bible={},
            world_state="",
            memory_context="",
            characters={},
            actor_id="u",
            action="attack",
            content_type="non_safe",
            forbid_player_autonomy=True,
            current_choices=[],
            include_psychology=True,
            word_limits=limits,
        )
        self.assertIn("常规内容（移动、道具介绍等一般性质内容）：行为＋场景介绍", regular)
        self.assertIn("不超过36字", regular)
        self.assertNotIn("不超过240字", regular)
        self.assertIn("非安全内容：具体行动＋与对方的过程＋结果", non_safe)
        self.assertIn("不超过240字", non_safe)
        self.assertIn("不超过42字", non_safe)

    def test_action_result_keeps_choices_and_string_booleans(self) -> None:
        output = RoundtableOutput(
            final_text=json.dumps(
                {
                    "narrative": "结果",
                    "choices": ["前进", "返回"],
                    "death": "true",
                    "story_ended": "true",
                    "major_node": "true",
                    "conversation_character_id": "npc-1",
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
        self.assertEqual(parsed.conversation_character_id, "npc-1")

    def test_structured_actions_reject_non_string_choices_and_narrative(self) -> None:
        self.assertEqual(_choice_list(["前进", {"action": "返回"}, 3, "观察"]), ["前进", "观察"])
        self.assertFalse(
            _is_action_payload(
                json.dumps(
                    {"narrative": {"text": "结果"}, "choices": ["一", "二", "三"]},
                    ensure_ascii=False,
                )
            )
        )


class NewFeatureTests(unittest.TestCase):
    def test_image_trigger_defaults_and_explicit_empty_disable(self) -> None:
        default = PluginConfig({})
        self.assertEqual(
            default.image_generation_triggers,
            {
                "first_appearance",
                "first_conversation",
                "killing",
                "violation",
                "scene_change",
                "battle_damage",
            },
        )
        self.assertEqual(PluginConfig({"image_generation_triggers": []}).image_generation_triggers, set())

    def test_disabled_image_entries_skip_background_judgment_gate(self) -> None:
        disabled = ImageGeneratorConfig("openai", "off", False, 50, "prompt", {})
        service = ImageService(FakeLLM(), [disabled], [], default_timeout=300, logger=Logger())
        self.assertFalse(service.has_enabled_generators())



if __name__ == "__main__":
    unittest.main()
