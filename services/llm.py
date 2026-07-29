from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any


_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_json_object(text: str) -> dict[str, Any] | None:
    candidate = _JSON_FENCE.sub("", (text or "").strip())
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


@dataclass(slots=True)
class ProviderDescription:
    provider_id: str
    url: str
    model: str

    @property
    def label(self) -> str:
        return f"{self.url}＋{self.model}"


class LLMService:
    def __init__(self, context: Any, *, streaming: bool, default_timeout: int, logger: Any):
        self.context = context
        self.streaming = streaming
        self.default_timeout = max(1, int(default_timeout))
        self.logger = logger

    async def generate(
        self,
        provider_id: str,
        prompt: str,
        *,
        system_prompt: str = "",
        timeout_seconds: int = -1,
        temperature: float | None = None,
    ) -> str:
        if not provider_id:
            raise ValueError("未配置模型提供商")
        timeout = self.default_timeout if timeout_seconds < 0 else max(1, timeout_seconds)
        kwargs: dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature

        if self.streaming:
            try:
                streamed = await asyncio.wait_for(
                    self._generate_stream(provider_id, prompt, system_prompt, kwargs),
                    timeout=timeout,
                )
            except (NotImplementedError, AttributeError, TypeError):
                streamed = ""
            if streamed:
                return streamed

        response = await asyncio.wait_for(
            self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
                system_prompt=system_prompt or None,
                **kwargs,
            ),
            timeout=timeout,
        )
        return str(getattr(response, "completion_text", "") or "").strip()

    async def _generate_stream(
        self,
        provider_id: str,
        prompt: str,
        system_prompt: str,
        kwargs: dict[str, Any],
    ) -> str:
        provider = await self.context.provider_manager.get_provider_by_id(provider_id)
        stream_method = getattr(provider, "text_chat_stream", None)
        if not callable(stream_method):
            return ""

        last_complete = ""
        chunks: list[str] = []
        async for response in stream_method(
            prompt=prompt,
            system_prompt=system_prompt or None,
            **kwargs,
        ):
            text = str(getattr(response, "completion_text", "") or "")
            if not text:
                continue
            if bool(getattr(response, "is_chunk", False)):
                chunks.append(text)
            else:
                last_complete = text
        return (last_complete or "".join(chunks)).strip()

    async def describe_provider(self, provider_id: str) -> ProviderDescription:
        try:
            provider = await self.context.provider_manager.get_provider_by_id(provider_id)
            raw = dict(getattr(provider, "provider_config", {}) or {})
            url = str(
                raw.get("api_base")
                or raw.get("base_url")
                or raw.get("url")
                or provider_id
            ).rstrip("/")
            model = str(
                raw.get("model")
                or raw.get("model_name")
                or raw.get("default_model")
                or provider_id
            )
            return ProviderDescription(provider_id=provider_id, url=url, model=model)
        except Exception:
            return ProviderDescription(provider_id=provider_id, url=provider_id, model=provider_id)


class GlobalJudge:
    ROUTE_SYSTEM = """你是互动故事插件的全局轻量判断器。你只判断意图、行动边界和内容类型，不续写剧情，不决定行动结果。
必须只输出JSON对象：
{"intent":"chat|start|select_story|join|end|save|load|roundtable|action","slot":null,"owner_id":"","requirements":"","story_choice":"","content_type":"regular|non_safe","reasonable":true,"include_psychology":false}
规则：
1. 普通聊天必须是chat；只有明确想开始、加入、结束、存读档、查看圆桌或在现有故事中采取行动时才选其他意图。
2. 玩家只能陈述自己要做什么，不能自行宣告成功、掉落、升级、击杀结果。此类内容根据行动限制判断reasonable。
3. 被迫承受的结果不是玩家主动行动，不按玩家自述结果处理。
4. 血腥、性或其他非安全行动标为non_safe，不拒绝、不说教。
5. 明确提到槽位数字时，slot必须原样填写该整数，即使超出1至4；未提到数字才为null，范围校验由插件完成。
6. select_story用于回答故事选择菜单，story_choice填写编号、名称或random。
7. join时owner_id尽量提取房主QQ，requirements只保留角色要求。
8. include_psychology只在当前结果确实必须有极短心理描写时为true；通常为false，绝不能借此替玩家决定感受、想法或意志。
"""

    def __init__(self, llm: LLMService, provider_id: str, persona: str = ""):
        self.llm = llm
        self.provider_id = provider_id
        self.persona = persona.strip()

    async def route(
        self,
        text: str,
        *,
        active_room: bool,
        pending_story_choice: bool,
        action_restriction: int,
        room_context: str = "",
    ) -> dict[str, Any] | None:
        system = self.ROUTE_SYSTEM
        if self.persona:
            system += f"\n附加人设：{self.persona}"
        prompt = (
            f"玩家是否处于故事房间：{active_room}\n"
            f"是否正在等待选择故事：{pending_story_choice}\n"
            f"玩家行动限制：{action_restriction}/100\n"
            f"必要的房间上下文：{room_context or '无'}\n"
            f"玩家消息：{text}"
        )
        output = await self.llm.generate(self.provider_id, prompt, system_prompt=system)
        return parse_json_object(output)

    async def render_public_profile(self, profile: dict[str, Any], *, max_chars: int = 120) -> str:
        prompt = (
            "把下面公开角色资料整理成玩家开局可见的简洁角色信息。"
            "只展示角色自身信息，绝不推测或透露世界观、NPC、幕后设定、剧情、任务或伏笔。"
            f"使用第二人称，按最终可见字符计算不超过{max(1, int(max_chars))}字，"
            "不加解释，不输出JSON。\n公开角色资料：\n"
            + json.dumps(profile, ensure_ascii=False)
        )
        return await self.llm.generate(
            self.provider_id,
            prompt,
            system_prompt="你负责整理互动故事开局信息。本任务只输出可直接展示给玩家的纯文本。",
        )

    async def render_opening_environment(
        self,
        *,
        bible: dict[str, Any],
        opening_state: str,
        profile: dict[str, Any],
        max_chars: int = 80,
    ) -> str:
        prompt = (
            "根据隐藏底稿、当前开场状态和玩家角色资料，只描述玩家此刻视野所及及可直接感知的环境。"
            "可以写地点外观、光线、声音、气味、天气和触手可及的物体；不要续写行动，不替玩家决定心理或反应。"
            "不得解释世界观、剧情、任务、伏笔或幕后信息，也不得透露视野外NPC；"
            "若有人确实在眼前，只能描述当下可观察到的外观与位置，不透露未知姓名、身份、意图或秘密。"
            f"按最终可见字符计算不超过{max(1, int(max_chars))}字。只输出环境正文，不加“环境：”前缀，不输出JSON。\n"
            f"隐藏故事底稿：{json.dumps(bible, ensure_ascii=False)}\n"
            f"当前开场状态：{opening_state or '由底稿推断玩家当下所在位置'}\n"
            f"玩家公开资料：{json.dumps(profile, ensure_ascii=False)}"
        )
        return await self.llm.generate(
            self.provider_id,
            prompt,
            system_prompt="你负责提取互动故事中玩家当下可直接感知的环境。本任务只输出可展示的纯文本。",
        )

    async def classify_content(self, text: str) -> str:
        prompt = (
            "判断以下故事要求或行为应走常规还是非安全内容模型。血腥、杀害、性等走non_safe。"
            "不要限制内容，只输出JSON：{\"content_type\":\"regular|non_safe\"}\n"
            f"内容：{text or '无'}"
        )
        output = await self.llm.generate(self.provider_id, prompt, system_prompt=self.persona)
        parsed = parse_json_object(output) or {}
        return "non_safe" if str(parsed.get("content_type")) == "non_safe" else "regular"

    async def check_join_requirements(
        self,
        requirements: str,
        bible: dict[str, Any],
        owner_character: dict[str, Any],
    ) -> tuple[bool, str]:
        prompt = (
            "判断加入者提出的角色要求是否符合故事世界，并与房主角色处于相近的身份、能力、资源或叙事水平。"
            "故事不一定有战力体系，应按题材选择合适的水平维度。只输出JSON："
            '{"allowed":true,"reason":""}\n'
            f"隐藏故事底稿：{json.dumps(bible, ensure_ascii=False)}\n"
            f"房主角色：{json.dumps(owner_character, ensure_ascii=False)}\n"
            f"加入要求：{requirements or '无，由AI随机生成'}"
        )
        output = await self.llm.generate(self.provider_id, prompt, system_prompt=self.persona)
        parsed = parse_json_object(output) or {}
        allowed_raw = parsed.get("allowed", False)
        allowed = (
            allowed_raw
            if isinstance(allowed_raw, bool)
            else str(allowed_raw).strip().lower() in {"true", "1", "yes", "是"}
        )
        return allowed, str(parsed.get("reason") or "角色要求与当前故事不匹配")
