from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DEFAULT_CONTENT_LIMIT = (
    "常规内容（移动、道具介绍等一般性质内容）：行为＋场景介绍，不超过30字。\n"
    "NSFW：具体行动＋与对方的过程＋结果，不超过200字。\n"
    "只用第二人称“你”来称呼玩家。\n"
    "只输出行为结果，严禁自作主张做出玩家未说过的行为。"
)


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
            content_limit=str(raw.get("content_limit") or DEFAULT_CONTENT_LIMIT),
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
        self.global_judge_persona = str(self.raw.get("global_judge_persona") or "").strip()
        self.forbid_player_autonomy = _bool(self.raw.get("forbid_player_autonomy"), True)
        self.streaming = _bool(self.raw.get("streaming"), True)
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
                    content_limit=str(entry.get("content_limit") or DEFAULT_CONTENT_LIMIT).strip(),
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
            result.append(
                RoundtableModelConfig(
                    name=str(entry.get("name") or "圆桌成员").strip(),
                    enabled=_bool(entry.get("enabled"), False),
                    priority=_int(entry.get("priority"), 50, 0, 100),
                    role=role if role in {"proposal", "reviewer"} else "proposal",
                    content_type=content_type if content_type in {"regular", "non_safe"} else "regular",
                    provider_id=str(entry.get("provider_id") or "").strip(),
                    persona=str(entry.get("persona") or "").strip(),
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
