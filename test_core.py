from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from services.config import PluginConfig, RoundtableModelConfig, StoryConfig
from services.game import GameService
from services.models import RoomMember, StoryRoom
from services.prompts import action_task
from services.roundtable import RoundtableOutput, RoundtableService
from services.saves import SaveService
from services.storage import RoomBusyRegistry, StateStore
from services.workflows import merge_workflow, set_dot_path
from services.config import ImageGeneratorConfig, WorkflowNodeMapping


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


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.prompts: list[str] = []

    async def generate(self, provider_id: str, prompt: str, **kwargs) -> str:
        self.calls.append(provider_id)
        self.prompts.append(prompt)
        return f"output-{provider_id}"

    async def describe_provider(self, provider_id: str):
        class Description:
            label = provider_id + "＋model"
        return Description()


class FailingProposalLLM(FakeLLM):
    async def generate(self, provider_id: str, prompt: str, **kwargs) -> str:
        self.calls.append(provider_id)
        self.prompts.append(prompt)
        if provider_id.startswith("p"):
            raise RuntimeError("failed")
        return f"output-{provider_id}"


class Logger:
    def warning(self, *_args, **_kwargs) -> None:
        return None


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
