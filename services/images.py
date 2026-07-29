from __future__ import annotations

import asyncio
import base64
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

from .config import ImageGeneratorConfig, WorkflowNodeMapping
from .llm import LLMService
from .prompts import image_prompt_task, scene_image_prompt_task
from .workflows import ComfyUIRunner, ImageGenerationError


@dataclass(slots=True)
class GeneratedImage:
    path: Path
    prompt: str
    generator: ImageGeneratorConfig


def apply_art_style(prompt: str, style: str) -> str:
    generated = str(prompt or "").strip().lstrip("，, ")
    configured = str(style or "").strip().rstrip("，,。；; ")
    if not configured:
        return generated
    prefix = configured if configured.endswith("画风") else f"{configured}画风"
    if generated == prefix or any(
        generated.startswith(prefix + separator)
        for separator in ("，", ",", "。", " ")
    ):
        return generated
    return f"{prefix}，{generated}" if generated else prefix


class OpenAIImageRunner:
    async def generate(
        self,
        generator: ImageGeneratorConfig,
        *,
        prompt: str,
        output_dir: Path,
        timeout_seconds: int,
        input_image: Path | None = None,
    ) -> Path:
        raw_keys = generator.raw.get("api_keys") or []
        if isinstance(raw_keys, str):
            raw_keys = [line.strip() for line in raw_keys.replace(",", "\n").splitlines()]
        keys = [str(key).strip() for key in raw_keys if str(key).strip()]
        if not keys:
            raise ImageGenerationError(f"{generator.name}未配置API Key")
        last_error: Exception | None = None
        for key in keys:
            try:
                return await self._with_key(
                    generator,
                    key=key,
                    prompt=prompt,
                    output_dir=output_dir,
                    timeout_seconds=timeout_seconds,
                    input_image=input_image,
                )
            except Exception as exc:
                last_error = exc
        raise ImageGenerationError(str(last_error or "OpenAI兼容绘图失败"))

    async def _with_key(
        self,
        generator: ImageGeneratorConfig,
        *,
        key: str,
        prompt: str,
        output_dir: Path,
        timeout_seconds: int,
        input_image: Path | None,
    ) -> Path:
        base_url = str(generator.raw.get("url") or "https://api.openai.com/v1").rstrip("/")
        for suffix in ("/images/generations", "/images/edits"):
            if base_url.endswith(suffix):
                base_url = base_url[: -len(suffix)]
        model = str(generator.raw.get("model_name") or "gpt-image-1")
        endpoint = f"{base_url}/images/edits" if input_image else f"{base_url}/images/generations"
        timeout = aiohttp.ClientTimeout(total=max(1, timeout_seconds))
        headers = {"Authorization": f"Bearer {key}"}
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                data = await self._request(session, endpoint, headers, model, prompt, input_image, quality=True)
            except ImageGenerationError:
                data = await self._request(session, endpoint, headers, model, prompt, input_image, quality=False)
            content = await self._extract(session, data)
        return await asyncio.to_thread(self._save_bytes, content, output_dir)

    async def _request(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        headers: dict[str, str],
        model: str,
        prompt: str,
        input_image: Path | None,
        *,
        quality: bool,
    ) -> dict[str, Any]:
        if input_image is None:
            payload: dict[str, Any] = {"model": model, "prompt": prompt, "n": 1}
            if quality:
                payload["quality"] = "high"
            async with session.post(endpoint, json=payload, headers=headers) as response:
                body = await response.json(content_type=None)
        else:
            form = aiohttp.FormData()
            form.add_field("model", model)
            form.add_field("prompt", prompt)
            form.add_field("n", "1")
            if quality:
                form.add_field("quality", "high")
            form.add_field("image", input_image.read_bytes(), filename=input_image.name, content_type="application/octet-stream")
            async with session.post(endpoint, data=form, headers=headers) as response:
                body = await response.json(content_type=None)
        if response.status < 200 or response.status >= 300:
            message = f"HTTP {response.status}"
            if isinstance(body, dict):
                error = body.get("error")
                if isinstance(error, dict):
                    message = str(error.get("message") or message)
                elif error:
                    message = str(error)
            raise ImageGenerationError(message)
        if not isinstance(body, dict):
            raise ImageGenerationError("图片API响应格式异常")
        return body

    async def _extract(self, session: aiohttp.ClientSession, data: dict[str, Any]) -> bytes:
        items = data.get("data")
        if not isinstance(items, list) or not items or not isinstance(items[0], dict):
            raise ImageGenerationError("图片API响应缺少data")
        item = items[0]
        if item.get("b64_json"):
            try:
                return base64.b64decode(str(item["b64_json"]))
            except ValueError as exc:
                raise ImageGenerationError("图片API返回无效base64") from exc
        if item.get("url"):
            async with session.get(str(item["url"])) as response:
                if response.status != 200:
                    raise ImageGenerationError(f"下载生成图片失败: HTTP {response.status}")
                return await response.read()
        raise ImageGenerationError("图片API响应中没有图片")

    @staticmethod
    def _save_bytes(content: bytes, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"openai_{uuid.uuid4().hex}.png"
        path.write_bytes(content)
        return path


class ImageService:
    def __init__(
        self,
        llm: LLMService,
        generators: list[ImageGeneratorConfig],
        mappings: list[WorkflowNodeMapping],
        *,
        default_timeout: int,
        logger: Any,
    ) -> None:
        self.llm = llm
        self.generators = generators
        self.mappings = mappings
        self.default_timeout = default_timeout
        self.logger = logger
        self.openai = OpenAIImageRunner()
        self.comfyui = ComfyUIRunner()

    def _candidates(self, *, mode: str, non_safe: bool) -> list[ImageGeneratorConfig]:
        candidates: list[ImageGeneratorConfig] = []
        for generator in self.generators:
            if not generator.enabled:
                continue
            if generator.kind == "comfyui":
                if generator.support_mode == mode:
                    candidates.append(generator)
                continue
            if mode == "edit" and non_safe and generator.content_type != "free":
                continue
            if mode == "generate" and generator.content_type not in {"regular", "free"}:
                continue
            candidates.append(generator)
        return sorted(
            candidates,
            key=lambda item: (item.priority, item.kind == "openai" and item.content_type == "regular"),
            reverse=True,
        )

    async def generate_character_image(
        self,
        *,
        bible: dict[str, Any],
        character: dict[str, Any],
        output_dir: Path,
        event_context: str = "首次登场",
        input_image: Path | None = None,
        original_prompt: str = "",
        non_safe: bool = False,
    ) -> GeneratedImage:
        mode = "edit" if input_image else "generate"
        last_error: Exception | None = None
        for generator in self._candidates(mode=mode, non_safe=non_safe):
            try:
                if not generator.prompt_provider_id:
                    raise ImageGenerationError(f"{generator.name}未配置提示词生成模型")
                timeout = int(generator.raw.get("timeout_seconds") or -1)
                timeout = self.default_timeout if timeout < 0 else max(1, timeout)
                prompt = await self.llm.generate(
                    generator.prompt_provider_id,
                    image_prompt_task(
                        bible=bible,
                        character=character,
                        mode=mode,
                        event_context=event_context,
                        original_prompt=original_prompt,
                        style=generator.style,
                    ),
                    system_prompt="你是专业的角色图像提示词工程师。保持人物身份和视觉一致性。",
                    timeout_seconds=timeout,
                )
                prompt = apply_art_style(prompt, generator.style)
                if generator.kind == "openai":
                    path = await self.openai.generate(
                        generator,
                        prompt=prompt,
                        output_dir=output_dir,
                        timeout_seconds=timeout,
                        input_image=input_image,
                    )
                else:
                    path = await self.comfyui.generate(
                        generator,
                        self.mappings,
                        prompt=prompt,
                        output_dir=output_dir,
                        timeout_seconds=self.default_timeout,
                        input_image=input_image,
                    )
                return GeneratedImage(path=path, prompt=prompt, generator=generator)
            except Exception as exc:
                last_error = exc
                self.logger.warning(f"绘图条目 {generator.name} 生成失败: {exc}")
        raise ImageGenerationError(str(last_error or "没有可用的人物绘图条目"))

    async def generate_scene_image(
        self,
        *,
        bible: dict[str, Any],
        event_context: str,
        output_dir: Path,
    ) -> GeneratedImage:
        last_error: Exception | None = None
        for generator in self._candidates(mode="generate", non_safe=False):
            try:
                if not generator.prompt_provider_id:
                    raise ImageGenerationError(f"{generator.name}未配置提示词生成模型")
                timeout = int(generator.raw.get("timeout_seconds") or -1)
                timeout = self.default_timeout if timeout < 0 else max(1, timeout)
                prompt = await self.llm.generate(
                    generator.prompt_provider_id,
                    scene_image_prompt_task(
                        bible=bible,
                        event_context=event_context,
                        style=generator.style,
                    ),
                    system_prompt="你是专业的场景图像提示词工程师。只描绘当前可见环境。",
                    timeout_seconds=timeout,
                )
                prompt = apply_art_style(prompt, generator.style)
                if generator.kind == "openai":
                    path = await self.openai.generate(
                        generator,
                        prompt=prompt,
                        output_dir=output_dir,
                        timeout_seconds=timeout,
                    )
                else:
                    path = await self.comfyui.generate(
                        generator,
                        self.mappings,
                        prompt=prompt,
                        output_dir=output_dir,
                        timeout_seconds=self.default_timeout,
                    )
                return GeneratedImage(path=path, prompt=prompt, generator=generator)
            except Exception as exc:
                last_error = exc
                self.logger.warning(f"绘图条目 {generator.name} 场景生成失败: {exc}")
        raise ImageGenerationError(str(last_error or "没有可用的人物绘图条目"))
