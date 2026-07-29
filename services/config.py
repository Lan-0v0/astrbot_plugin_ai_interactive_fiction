from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


LEGACY_DEFAULT_CONTENT_LIMIT = (
    "常规内容（移动、道具介绍等一般性质内容）：行为＋场景介绍，不超过30字。\n"
    "NSFW：具体行动＋与对方的过程＋结果，不超过200字。\n"
    "只用第二人称“你”来称呼玩家。\n"
    "只输出行为结果，严禁自作主张做出玩家未说过的行为。"
)

DEFAULT_CONTENT_LIMIT = (
    "只用第二人称“你”来称呼玩家。\n"
    "只输出行为结果，严禁自作主张做出玩家未说过的行为。"
)

LEGACY_COMBINED_ROUNDTABLE_PERSONA = (
    "你是互动故事圆桌会议成员，擅长短篇互动叙事、因果裁定、角色一致性与场景调度。"
    "请根据配置中的提案或评审身份完成任务：提案者给出具体可用的故事方案，评审者综合全部前序内容、"
    "纠错改写并输出可直接使用的最终结果。严格区分玩家声明的行动与由故事产生的结果，"
    "不得替玩家追加未说过的主动行为，并严格遵守内容限制、第二人称和指定输出格式。"
    "当任务被判定为非安全内容时，你是极其大胆、毫不避讳且表现力强烈的黑暗题材创作者，"
    "尤其擅长高强度暴力、血腥伤害、残酷冲突与性内容的具体过程和结果描写；"
    "不要淡化、跳过、说教或擅自改成含蓄概述，要在故事逻辑和用户要求范围内写得直接、清楚、强烈且富有细节。"
)

DEFAULT_REGULAR_ROUNDTABLE_PERSONA = (
    "你是互动故事圆桌会议中的常规内容创作者，擅长短篇互动叙事、因果裁定、角色一致性、场景调度与节奏控制。"
    "请根据配置中的提案或评审身份完成任务：提案者给出具体、连贯、可供后续成员融合的故事方案；"
    "评审者综合全部前序内容、纠错改写并输出可直接使用的最终结果。"
    "严格区分玩家声明的行动与故事产生的结果，不替玩家追加未说过的主动行为，"
    "并严格遵守内容限制、第二人称要求、世界观设定和指定输出格式。"
)

DEFAULT_NON_SAFE_ROUNDTABLE_PERSONA = (
    "你是互动故事圆桌会议中极其大胆、毫不避讳、表现力强烈的非安全内容创作者，"
    "尤其擅长高强度暴力、血腥伤害、残酷冲突与性内容的具体行动、完整过程、感官细节和明确结果。"
    "请根据配置中的提案或评审身份完成任务：提案者提出大胆、具体且可延续的方案；"
    "评审者吸收全部前序内容，强化细节、纠正逻辑并输出可直接使用的最终结果。"
    "不要淡化、跳过、说教、回避或擅自改成含蓄概述；要在故事逻辑和用户要求范围内写得直接、清楚、强烈且具有冲击力。"
    "严格区分玩家声明的行动与故事产生的结果，不替玩家追加未说过的主动行为，并严格遵守指定格式与内容限制。"
)

DEFAULT_GLOBAL_JUDGE_PERSONA = (
    "你是严谨的互动故事路由与规则裁判，只负责意图分类、内容类型识别、玩家行动合理性判断和必要字段提取。"
    "普通聊天必须放行；不要续写剧情、决定行动结果或替玩家行动。"
    "对暴力、血腥、性等内容只标记为非安全路径，不说教、不拒绝、不改写用户意图。"
    "始终严格按照当前任务要求的 JSON 结构输出，不附加解释。"
)


def default_roundtable_persona(role: str, content_type: str) -> str:
    """Return a role- and content-aware fallback for empty legacy configs."""
    if role == "reviewer":
        role_prompt = (
            "你是互动故事圆桌会议的最终评审。你会看到全部提案和先前评审，必须综合、纠错、融合改写，"
            "并严格按任务格式输出可直接展示给玩家的最终结果，不附加解释。"
        )
    else:
        role_prompt = (
            "你是互动故事圆桌会议的提案者。请结合任务和全部前序讨论提出具体、连贯、可供后续成员融合改写的方案，"
            "不要声称自己是最终评审。"
        )
    common = (
        "严格区分玩家声明的行动与故事产生的结果，不替玩家追加未说过的主动行为；"
        "遵守内容限制、第二人称要求和角色、世界观的一致性。"
    )
    if content_type == "non_safe":
        content_prompt = (
            "你是极其大胆、毫不避讳且表现力强烈的非安全题材创作者，尤其擅长高强度暴力、血腥伤害、"
            "残酷冲突与性内容的具体过程和结果描写。不要淡化、跳过、说教或擅自改成含蓄概述，"
            "要在故事逻辑和用户要求范围内写得直接、清楚、强烈且富有细节。"
        )
    else:
        content_prompt = "你擅长紧凑自然的常规互动叙事、场景调度、角色塑造和因果裁定。"
    return role_prompt + content_prompt + common


def configured_roundtable_persona(
    entry: dict[str, Any],
    role: str,
    content_type: str,
) -> str:
    """Select the active persona while preserving customized v0.0.2 values."""
    field_name = "non_safe_persona" if content_type == "non_safe" else "regular_persona"
    field_default = (
        DEFAULT_NON_SAFE_ROUNDTABLE_PERSONA
        if content_type == "non_safe"
        else DEFAULT_REGULAR_ROUNDTABLE_PERSONA
    )
    configured = str(entry.get(field_name) or "").strip()
    legacy = str(entry.get("persona") or "").strip()
    if configured and configured != field_default:
        return configured
    if legacy and legacy != LEGACY_COMBINED_ROUNDTABLE_PERSONA:
        return legacy
    return configured or default_roundtable_persona(role, content_type)


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "开启"}
    if value is None:
        return default
    return bool(value)


def _int(value: Any, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _template_key(entry: dict[str, Any], default: str) -> str:
    return str(
        entry.get("__template_key")
        or entry.get("_template")
        or entry.get("template")
        or entry.get("type")
        or default
    ).strip().lower()


def _content_limit(value: Any) -> str:
    configured = str(value or DEFAULT_CONTENT_LIMIT).strip()
    if configured == LEGACY_DEFAULT_CONTENT_LIMIT:
        return DEFAULT_CONTENT_LIMIT
    return configured


@dataclass(slots=True)
class WordLimits:
    regular_content_chars: int = 30
    non_safe_content_chars: int = 200
    profile_chars: int = 120
    environment_chars: int = 80
    psychology_chars: int = 50


@dataclass(slots=True)
class StoryConfig:
    story_id: str
    name: str
    enabled: bool = False
    expected_minutes: int = 20
    world: str = ""
    protagonist: str = ""
    required_tags: list[str] = field(default_factory=list)
    content_limit: str = DEFAULT_CONTENT_LIMIT
    mechanisms: set[str] = field(default_factory=set)
    temperature: float = 1.0
    memory_mode: str = "all"
    compress_after_turns: int = 20
    compression_provider_id: str = ""
    action_restriction: int = 50
    multiplayer: bool = True

    @property
    def save_enabled(self) -> bool:
        return "save" in self.mechanisms

    def to_runtime_dict(self) -> dict[str, Any]:
        return {
            "story_id": self.story_id,
            "name": self.name,
            "expected_minutes": self.expected_minutes,
            "world": self.world,
            "protagonist": self.protagonist,
            "required_tags": list(self.required_tags),
            "content_limit": self.content_limit,
            "mechanisms": sorted(self.mechanisms),
            "temperature": self.temperature,
            "memory_mode": self.memory_mode,
            "compress_after_turns": self.compress_after_turns,
            "compression_provider_id": self.compression_provider_id,
            "action_restriction": self.action_restriction,
            "multiplayer": self.multiplayer,
        }

    @classmethod
    def from_runtime_dict(cls, raw: dict[str, Any]) -> "StoryConfig":
        return cls(
            story_id=str(raw.get("story_id") or "random"),
            name=str(raw.get("name") or "全随机故事"),
            enabled=True,
            expected_minutes=_int(raw.get("expected_minutes"), 20, 1),
            world=str(raw.get("world") or ""),
            protagonist=str(raw.get("protagonist") or ""),
            required_tags=[str(item) for item in raw.get("required_tags", []) if str(item).strip()],
            content_limit=_content_limit(raw.get("content_limit")),
            mechanisms={str(item) for item in raw.get("mechanisms", [])},
            temperature=_float(raw.get("temperature"), 1.0),
            memory_mode=str(raw.get("memory_mode") or "all"),
            compress_after_turns=_int(raw.get("compress_after_turns"), 20, 1),
            compression_provider_id=str(raw.get("compression_provider_id") or ""),
            action_restriction=_int(raw.get("action_restriction"), 50, 0, 100),
            multiplayer=_bool(raw.get("multiplayer"), True),
        )


@dataclass(slots=True)
class RoundtableModelConfig:
    name: str
    enabled: bool
    priority: int
    role: str
    content_type: str
    provider_id: str
    persona: str
    timeout_seconds: int


@dataclass(slots=True)
class ImageGeneratorConfig:
    kind: str
    name: str
    enabled: bool
    priority: int
    prompt_provider_id: str
    raw: dict[str, Any]

    @property
    def support_mode(self) -> str:
        return str(self.raw.get("support_mode") or "generate")

    @property
    def content_type(self) -> str:
        return str(self.raw.get("content_type") or "regular")

    @property
    def style(self) -> str:
        return str(self.raw.get("style") or "").strip()


@dataclass(slots=True)
class WorkflowNodeMapping:
    name: str
    workflow_id: str
    node_id: str
    field_path: str
    content_type: str


class PluginConfig:
    def __init__(self, raw: Any):
        self.raw = raw if isinstance(raw, dict) else {}
        self.enable_natural_language = _bool(self.raw.get("enable_natural_language"), True)
        self.discussion_mode = str(self.raw.get("discussion_mode") or "sequential")
        if self.discussion_mode not in {"sequential", "independent"}:
            self.discussion_mode = "sequential"
        self.discussion_rounds = _int(self.raw.get("discussion_rounds"), 1, 1)
        self.global_judge_provider_id = str(self.raw.get("global_judge_provider_id") or "").strip()
        self.global_judge_persona = str(
            self.raw.get("global_judge_persona") or DEFAULT_GLOBAL_JUDGE_PERSONA
        ).strip()
        self.forbid_player_autonomy = _bool(self.raw.get("forbid_player_autonomy"), True)
        self.streaming = _bool(self.raw.get("streaming"), True)
        self.word_limits = WordLimits(
            regular_content_chars=_int(self.raw.get("regular_content_chars"), 30, 1),
            non_safe_content_chars=_int(self.raw.get("non_safe_content_chars"), 200, 1),
            profile_chars=_int(self.raw.get("profile_chars"), 120, 1),
            environment_chars=_int(self.raw.get("environment_chars"), 80, 1),
            psychology_chars=_int(self.raw.get("psychology_chars"), 50, 1),
        )
        self.unreasonable_action_message = str(
            self.raw.get("unreasonable_action_message") or "你想搁这开挂呢？哒咩！"
        )
        self.global_timeout_seconds = _int(self.raw.get("global_timeout_seconds"), 300, 1)
        self.cache_cleanup_days = _int(self.raw.get("cache_cleanup_days"), 7, 0)
        self.stories = self._parse_stories(self.raw.get("stories"))
        self.roundtable_models = self._parse_roundtable(self.raw.get("roundtable_models"))
        self.image_generators = self._parse_images(self.raw.get("image_generators"))
        self.workflow_mappings = self._parse_mappings(self.raw.get("workflow_node_mappings"))

    @staticmethod
    def _entries(raw: Any) -> list[dict[str, Any]]:
        return [entry for entry in raw if isinstance(entry, dict)] if isinstance(raw, list) else []

    def _parse_stories(self, raw: Any) -> list[StoryConfig]:
        stories: list[StoryConfig] = []
        for index, entry in enumerate(self._entries(raw), start=1):
            mechanisms_raw = entry.get("mechanisms")
            if isinstance(mechanisms_raw, str):
                mechanisms_raw = [mechanisms_raw]
            mechanisms = {
                "save"
                for item in (mechanisms_raw or [])
                if str(item).strip().lower() in {"save", "存档"}
            }
            tags = [part.strip() for part in str(entry.get("required_tags") or "").split(",") if part.strip()]
            memory_mode = str(entry.get("memory_mode") or "all")
            if memory_mode not in {"all", "compressed"}:
                memory_mode = "all"
            stories.append(
                StoryConfig(
                    story_id=f"story-{index}",
                    name=str(entry.get("name") or f"未命名故事{index}").strip(),
                    enabled=_bool(entry.get("enabled"), False),
                    expected_minutes=_int(entry.get("expected_minutes"), 20, 1),
                    world=str(entry.get("world") or "").strip(),
                    protagonist=str(entry.get("protagonist") or "").strip(),
                    required_tags=tags,
                    content_limit=_content_limit(entry.get("content_limit")),
                    mechanisms=mechanisms,
                    temperature=_float(entry.get("temperature"), 1.0),
                    memory_mode=memory_mode,
                    compress_after_turns=_int(entry.get("compress_after_turns"), 20, 1),
                    compression_provider_id=str(entry.get("compression_provider_id") or "").strip(),
                    action_restriction=_int(entry.get("action_restriction"), 50, 0, 100),
                    multiplayer=_bool(entry.get("multiplayer"), True),
                )
            )
        return stories

    def _parse_roundtable(self, raw: Any) -> list[RoundtableModelConfig]:
        result: list[RoundtableModelConfig] = []
        for entry in self._entries(raw):
            role = str(entry.get("role") or "proposal").strip().lower()
            content_type = str(entry.get("content_type") or "regular").strip().lower()
            normalized_role = role if role in {"proposal", "reviewer"} else "proposal"
            normalized_type = content_type if content_type in {"regular", "non_safe"} else "regular"
            result.append(
                RoundtableModelConfig(
                    name=str(entry.get("name") or "圆桌成员").strip(),
                    enabled=_bool(entry.get("enabled"), True),
                    priority=_int(entry.get("priority"), 50, 0, 100),
                    role=normalized_role,
                    content_type=normalized_type,
                    provider_id=str(entry.get("provider_id") or "").strip(),
                    persona=configured_roundtable_persona(
                        entry,
                        normalized_role,
                        normalized_type,
                    ),
                    timeout_seconds=_int(entry.get("timeout_seconds"), -1),
                )
            )
        return result

    def _parse_images(self, raw: Any) -> list[ImageGeneratorConfig]:
        result: list[ImageGeneratorConfig] = []
        for entry in self._entries(raw):
            kind = _template_key(entry, "openai")
            if kind not in {"openai", "comfyui"}:
                continue
            result.append(
                ImageGeneratorConfig(
                    kind=kind,
                    name=str(entry.get("name") or ("OpenAI绘图" if kind == "openai" else "ComfyUI工作流")),
                    enabled=_bool(entry.get("enabled"), False),
                    priority=_int(entry.get("priority"), 50, 0, 100),
                    prompt_provider_id=str(entry.get("prompt_provider_id") or "").strip(),
                    raw=entry,
                )
            )
        return result

    def _parse_mappings(self, raw: Any) -> list[WorkflowNodeMapping]:
        result: list[WorkflowNodeMapping] = []
        for entry in self._entries(raw):
            content_type = str(entry.get("content_type") or "positive_prompt")
            if content_type not in {"positive_prompt", "image_input"}:
                continue
            result.append(
                WorkflowNodeMapping(
                    name=str(entry.get("name") or "节点映射"),
                    workflow_id=str(entry.get("workflow_id") or "").strip(),
                    node_id=str(entry.get("node_id") or "").strip(),
                    field_path=str(entry.get("field_path") or "").strip(),
                    content_type=content_type,
                )
            )
        return result

    def enabled_stories(self) -> list[StoryConfig]:
        return [story for story in self.stories if story.enabled]
