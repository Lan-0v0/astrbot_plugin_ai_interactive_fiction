from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .llm import LLMService
from .roundtable import RoundtableGenerationError, RoundtableOutput, RoundtableService


@dataclass(slots=True)
class StoryGenerationRequest:
    task: str
    content_type: str = "regular"
    temperature: float = 1.0
    output_validator: Callable[[str], bool] | None = None
    repair_instruction: str = ""


@dataclass(slots=True)
class PreparedStoryGeneration:
    request: StoryGenerationRequest
    draft: str | None


class StoryGeneratorService:
    """Generate once with the global story model, then optionally invoke the roundtable."""

    def __init__(
        self,
        llm: LLMService,
        roundtable: RoundtableService,
        *,
        provider_id: str,
        persona: str,
        non_safe_provider_id: str = "",
        non_safe_persona: str = "",
        roundtable_triggers: set[str],
        logger: Any,
    ) -> None:
        self.llm = llm
        self.roundtable = roundtable
        self.provider_id = provider_id
        self.persona = persona
        self.non_safe_provider_id = non_safe_provider_id
        self.non_safe_persona = non_safe_persona
        self.roundtable_triggers = set(roundtable_triggers)
        self.logger = logger

    async def prepare(self, request: StoryGenerationRequest) -> PreparedStoryGeneration:
        try:
            draft = await self._generate_direct(request)
        except Exception as exc:
            model_name = "全局非安全内容LLM" if request.content_type == "non_safe" else "全局故事生成LLM"
            self.logger.warning(f"{model_name}生成失败，回退圆桌会议: {exc}")
            draft = None
        return PreparedStoryGeneration(request=request, draft=draft)

    async def generate(
        self,
        request: StoryGenerationRequest,
        *,
        trigger_types: set[str],
    ) -> RoundtableOutput:
        return await self.complete(
            await self.prepare(request),
            trigger_types=trigger_types,
            content_type=request.content_type,
        )

    async def complete(
        self,
        prepared: PreparedStoryGeneration,
        *,
        trigger_types: set[str],
        content_type: str,
    ) -> RoundtableOutput:
        request = prepared.request
        should_use_roundtable = bool(self.roundtable_triggers.intersection(trigger_types))
        if prepared.draft is not None and not should_use_roundtable:
            return RoundtableOutput(final_text=prepared.draft, discussion=[])

        return await self.roundtable.run(
            request.task,
            content_type=content_type,
            temperature=request.temperature,
            output_validator=request.output_validator,
            repair_instruction=request.repair_instruction,
            initial_draft=prepared.draft or "",
        )

    async def _generate_direct(self, request: StoryGenerationRequest) -> str:
        is_non_safe = request.content_type == "non_safe"
        provider_id = self.non_safe_provider_id if is_non_safe else self.provider_id
        persona = self.non_safe_persona if is_non_safe else self.persona
        model_name = "全局非安全内容LLM" if is_non_safe else "全局故事生成LLM"
        if not provider_id:
            raise RoundtableGenerationError(f"未配置{model_name}")
        output = await self.llm.generate(
            provider_id,
            request.task,
            system_prompt=persona,
            temperature=request.temperature,
        )
        output = str(output or "").strip()
        if not output:
            raise RoundtableGenerationError("全局故事生成LLM返回空内容")
        if request.output_validator is None or self._is_valid(request.output_validator, output):
            return output

        repaired = await self.llm.generate(
            provider_id,
            self._repair_prompt(request, output),
            system_prompt=persona,
            temperature=0.0,
        )
        repaired = str(repaired or "").strip()
        if repaired and self._is_valid(request.output_validator, repaired):
            return repaired
        raise RoundtableGenerationError("全局故事生成LLM未返回符合要求的结构化内容")

    @staticmethod
    def _is_valid(validator: Callable[[str], bool], text: str) -> bool:
        try:
            return bool(validator(text))
        except Exception:
            return False

    @staticmethod
    def _repair_prompt(request: StoryGenerationRequest, invalid_text: str) -> str:
        return (
            f"{request.task}\n\n"
            "你上一次的输出未通过插件结构校验。只修复格式和缺失字段，保留原有故事事实，"
            "重新输出完整JSON对象，不使用Markdown代码块，不解释。\n"
            f"结构要求：\n{request.repair_instruction or '严格遵循原任务规定的JSON结构。'}\n\n"
            f"上一次输出：\n{invalid_text}"
        )
