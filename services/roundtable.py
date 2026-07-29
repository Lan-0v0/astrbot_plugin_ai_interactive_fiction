from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Any

from .config import RoundtableModelConfig, default_roundtable_persona
from .llm import LLMService


class RoundtableConfigurationError(RuntimeError):
    pass


class RoundtableGenerationError(RuntimeError):
    pass


@dataclass(slots=True)
class RoundtableOutput:
    final_text: str
    discussion: list[dict[str, str]]


class RoundtableService:
    def __init__(
        self,
        llm: LLMService,
        models: list[RoundtableModelConfig],
        *,
        mode: str,
        rounds: int,
        logger: Any,
    ) -> None:
        self.llm = llm
        self.models = models
        self.mode = mode
        self.rounds = max(1, rounds)
        self.logger = logger

    def _ordered(self, models: list[RoundtableModelConfig]) -> list[RoundtableModelConfig]:
        grouped: dict[int, list[RoundtableModelConfig]] = {}
        for model in models:
            grouped.setdefault(model.priority, []).append(model)
        result: list[RoundtableModelConfig] = []
        for priority in sorted(grouped):
            group = grouped[priority]
            random.shuffle(group)
            result.extend(group)
        return result

    def _select_role(self, role: str, requested_type: str) -> list[RoundtableModelConfig]:
        enabled = [
            model for model in self.models
            if model.enabled and model.provider_id and model.role == role
        ]
        if requested_type == "non_safe":
            preferred = [model for model in enabled if model.content_type == "non_safe"]
            if preferred:
                return self._ordered(preferred)
        regular = [model for model in enabled if model.content_type == "regular"]
        return self._ordered(regular)

    async def run(
        self,
        task: str,
        *,
        content_type: str = "regular",
        temperature: float = 1.0,
    ) -> RoundtableOutput:
        proposers = self._select_role("proposal", content_type)
        reviewers = self._select_role("reviewer", content_type)
        if not proposers or not reviewers:
            raise RoundtableConfigurationError("缺少常规/非安全模型配置")

        discussion: list[dict[str, str]] = []
        proposal_context: list[str] = []
        if self.mode == "independent":
            results = await asyncio.gather(
                *[
                    self._call_model(model, self._proposal_prompt(task, [], 1), temperature)
                    for model in proposers
                ],
                return_exceptions=True,
            )
            for model, result in zip(proposers, results):
                if isinstance(result, Exception) or not str(result).strip():
                    await self._log_failure(model, result if isinstance(result, Exception) else None)
                    continue
                text = str(result).strip()
                label = await self._label(model)
                discussion.append({"label": label, "content": text})
                proposal_context.append(f"{label}：{text}")
        else:
            for round_number in range(1, self.rounds + 1):
                successes_before = len(proposal_context)
                for model in proposers:
                    try:
                        text = await self._call_model(
                            model,
                            self._proposal_prompt(task, proposal_context, round_number),
                            temperature,
                        )
                    except Exception as exc:
                        await self._log_failure(model, exc)
                        continue
                    if not text:
                        await self._log_failure(model, None)
                        continue
                    label = await self._label(model)
                    discussion.append({"label": label, "content": text})
                    proposal_context.append(f"第{round_number}轮 {label}：{text}")
                if len(proposal_context) == successes_before:
                    break

        review_context = list(proposal_context)
        final_text = ""
        reviewer_success = False
        for model in reviewers:
            try:
                text = await self._call_model(
                    model,
                    self._review_prompt(task, review_context),
                    temperature,
                )
            except Exception as exc:
                await self._log_failure(model, exc)
                continue
            if not text:
                await self._log_failure(model, None)
                continue
            reviewer_success = True
            final_text = text
            label = await self._label(model)
            discussion.append({"label": label, "content": text})
            review_context.append(f"评审 {label}：{text}")

        if not reviewer_success:
            raise RoundtableGenerationError("生成失败，请配置或检查模型")
        return RoundtableOutput(final_text=final_text, discussion=discussion)

    async def _call_model(self, model: RoundtableModelConfig, prompt: str, temperature: float) -> str:
        system = model.persona or default_roundtable_persona(model.role, model.content_type)
        return await self.llm.generate(
            model.provider_id,
            prompt,
            system_prompt=system,
            timeout_seconds=model.timeout_seconds,
            temperature=temperature,
        )

    @staticmethod
    def _proposal_prompt(task: str, previous: list[str], round_number: int) -> str:
        context = "\n\n".join(previous) if previous else "暂无前序提案"
        return (
            f"{task}\n\n当前是第{round_number}轮提案。\n"
            f"前面所有模型的内容：\n{context}\n\n"
            "请提出你的版本。不要讨论插件实现，不要声称自己是最终评审。"
        )

    @staticmethod
    def _review_prompt(task: str, previous: list[str]) -> str:
        context = "\n\n".join(previous) if previous else "没有成功提案，请直接依据任务完成评审输出"
        return (
            f"{task}\n\n全部提案及先前评审：\n{context}\n\n"
            "综合、纠错并改写。你的输出将被直接使用，必须严格遵循任务规定的输出格式，不要附加解释。"
        )

    async def _label(self, model: RoundtableModelConfig) -> str:
        description = await self.llm.describe_provider(model.provider_id)
        return description.label

    async def _log_failure(self, model: RoundtableModelConfig, exc: Exception | None) -> None:
        label = await self._label(model)
        if exc is None:
            self.logger.warning(f"{label} 生成失败")
        else:
            self.logger.warning(f"{label} 生成失败: {exc}")
