from __future__ import annotations

import json
from typing import Any

from .config import StoryConfig
from .llm import LLMService
from .models import StoryRoom
from .prompts import compression_prompt


class MemoryService:
    TEMPORARY_COMPRESS_CHARS = 30000

    def __init__(self, llm: LLMService, logger: Any):
        self.llm = llm
        self.logger = logger

    async def context_for_action(self, room: StoryRoom, story: StoryConfig) -> str:
        raw = self._format_memory(room.memory_summary, room.history)
        if len(raw) <= self.TEMPORARY_COMPRESS_CHARS or not story.compression_provider_id:
            return raw
        try:
            temporary = await self.llm.generate(
                story.compression_provider_id,
                compression_prompt(room.memory_summary, room.history),
            )
            if temporary:
                return self._format_memory(temporary, room.history[-1:])
        except Exception as exc:
            self.logger.warning(f"临时记忆压缩失败，继续使用原始记忆: {exc}")
        return raw

    async def compress_if_needed(self, room: StoryRoom, story: StoryConfig) -> bool:
        if story.memory_mode != "compressed":
            return False
        if len(room.history) <= story.compress_after_turns:
            return False
        if not story.compression_provider_id:
            self.logger.warning("故事需要长记忆压缩，但未配置文本压缩LLM")
            return False
        try:
            summary = await self.llm.generate(
                story.compression_provider_id,
                compression_prompt(room.memory_summary, room.history[:-1]),
            )
        except Exception as exc:
            self.logger.warning(f"长记忆自动压缩失败: {exc}")
            return False
        if not summary:
            return False
        room.memory_summary = summary
        room.history = room.history[-1:]
        return True

    @staticmethod
    def _format_memory(summary: str, history: list[dict[str, Any]]) -> str:
        return (
            f"长期摘要：{summary or '无'}\n"
            f"近期行动：{json.dumps(history, ensure_ascii=False)}"
        )

