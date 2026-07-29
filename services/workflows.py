from __future__ import annotations

import asyncio
import copy
import json
import time
import uuid
from pathlib import Path
from typing import Any

import aiohttp

from .config import ImageGeneratorConfig, WorkflowNodeMapping


class ImageGenerationError(RuntimeError):
    pass


MAX_IMAGE_BYTES = 32 * 1024 * 1024


async def read_image_response(response: aiohttp.ClientResponse) -> bytes:
    declared = response.content_length
    if declared is not None and declared > MAX_IMAGE_BYTES:
        raise ImageGenerationError("生成图片超过32MB限制")
    content = bytearray()
    async for chunk in response.content.iter_chunked(64 * 1024):
        content.extend(chunk)
        if len(content) > MAX_IMAGE_BYTES:
            raise ImageGenerationError("生成图片超过32MB限制")
    if not content:
        raise ImageGenerationError("生成图片内容为空")
    return bytes(content)


def set_dot_path(container: Any, path: str, value: Any) -> None:
    segments = [segment for segment in path.strip().split(".") if segment]
    if not segments:
        raise KeyError("字段路径不能为空")
    current = container
    for segment in segments[:-1]:
        if isinstance(current, list):
            current = current[int(segment)]
        elif isinstance(current, dict):
            current = current[segment]
        else:
            raise TypeError(f"无法访问字段 {segment}")
    last = segments[-1]
    if isinstance(current, list):
        current[int(last)] = value
    elif isinstance(current, dict):
        if last not in current:
            raise KeyError(f"字段不存在: {last}")
        current[last] = value
    else:
        raise TypeError(f"无法设置字段 {last}")


def merge_workflow(
    generator: ImageGeneratorConfig,
    mappings: list[WorkflowNodeMapping],
    *,
    prompt: str,
    uploaded_image: str = "",
) -> dict[str, Any]:
    try:
        payload = json.loads(str(generator.raw.get("workflow_content") or "{}"))
    except json.JSONDecodeError as exc:
        raise ImageGenerationError(f"工作流「{generator.name}」JSON解析失败: {exc}") from exc
    if not isinstance(payload, dict):
        raise ImageGenerationError(f"工作流「{generator.name}」内容必须是JSON对象")
    payload = copy.deepcopy(payload)
    prompt_mapped = False
    image_mapped = False
    for mapping in mappings:
        if mapping.workflow_id != generator.name:
            continue
        node = payload.get(mapping.node_id)
        if not isinstance(node, dict):
            raise ImageGenerationError(f"工作流「{generator.name}」未找到节点 {mapping.node_id}")
        if mapping.content_type == "positive_prompt":
            value = prompt
            prompt_mapped = True
        elif mapping.content_type == "image_input":
            if not uploaded_image:
                continue
            value = uploaded_image
            image_mapped = True
        else:
            continue
        try:
            set_dot_path(node, mapping.field_path, value)
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise ImageGenerationError(
                f"工作流「{generator.name}」节点 {mapping.node_id} 字段路径 {mapping.field_path} 无效: {exc}"
            ) from exc
    if not prompt_mapped:
        raise ImageGenerationError(f"工作流「{generator.name}」缺少正向提示词节点映射")
    if uploaded_image and not image_mapped:
        raise ImageGenerationError(f"工作流「{generator.name}」缺少图像输入节点映射")
    return payload


class ComfyUIRunner:
    async def generate(
        self,
        generator: ImageGeneratorConfig,
        mappings: list[WorkflowNodeMapping],
        *,
        prompt: str,
        output_dir: Path,
        timeout_seconds: int,
        input_image: Path | None = None,
    ) -> Path:
        base_url = str(generator.raw.get("service_url") or "http://127.0.0.1:8188").rstrip("/")
        client_timeout = aiohttp.ClientTimeout(total=max(1, timeout_seconds))
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            uploaded = ""
            if input_image is not None:
                uploaded = await self._upload(session, base_url, input_image)
            payload = merge_workflow(generator, mappings, prompt=prompt, uploaded_image=uploaded)
            prompt_id = await self._submit(session, base_url, payload)
            history = await self._poll(session, base_url, prompt_id, timeout_seconds)
            reference = self._first_image(history)
            if not reference:
                raise ImageGenerationError(f"工作流「{generator.name}」未返回图片")
            content = await self._download(session, base_url, reference)
        return await asyncio.to_thread(self._save_bytes, content, output_dir, "comfyui")

    async def _upload(self, session: aiohttp.ClientSession, base_url: str, path: Path) -> str:
        data = aiohttp.FormData()
        data.add_field("image", path.read_bytes(), filename=path.name, content_type="application/octet-stream")
        async with session.post(f"{base_url}/upload/image", data=data) as response:
            body = await response.json(content_type=None)
            if response.status != 200:
                raise ImageGenerationError(f"ComfyUI上传图片失败: HTTP {response.status}")
        name = str(body.get("name") or body.get("filename") or "") if isinstance(body, dict) else ""
        if not name:
            raise ImageGenerationError("ComfyUI上传响应缺少文件名")
        return name

    async def _submit(self, session: aiohttp.ClientSession, base_url: str, payload: dict[str, Any]) -> str:
        async with session.post(
            f"{base_url}/prompt",
            json={"prompt": payload, "client_id": uuid.uuid4().hex},
        ) as response:
            body = await response.json(content_type=None)
            if response.status != 200:
                raise ImageGenerationError(f"ComfyUI提交任务失败: HTTP {response.status}")
        prompt_id = str(body.get("prompt_id") or "") if isinstance(body, dict) else ""
        if not prompt_id:
            raise ImageGenerationError("ComfyUI响应缺少prompt_id")
        return prompt_id

    async def _poll(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        prompt_id: str,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + max(1, timeout_seconds)
        while time.monotonic() < deadline:
            async with session.get(f"{base_url}/history/{prompt_id}") as response:
                body = await response.json(content_type=None)
                if response.status != 200:
                    raise ImageGenerationError(f"ComfyUI查询任务失败: HTTP {response.status}")
            if isinstance(body, dict) and isinstance(body.get(prompt_id), dict):
                entry = body[prompt_id]
                if entry.get("outputs"):
                    return entry
                status = entry.get("status")
                if isinstance(status, dict) and str(status.get("status_str") or "").lower() in {"error", "failed"}:
                    raise ImageGenerationError("ComfyUI工作流执行失败")
            await asyncio.sleep(min(1.0, max(0.1, deadline - time.monotonic())))
        raise ImageGenerationError("ComfyUI任务超时")

    @staticmethod
    def _first_image(history: dict[str, Any]) -> dict[str, str] | None:
        outputs = history.get("outputs")
        if not isinstance(outputs, dict):
            return None
        for output in outputs.values():
            if not isinstance(output, dict) or not isinstance(output.get("images"), list):
                continue
            for image in output["images"]:
                if isinstance(image, dict) and image.get("filename"):
                    return image
        return None

    async def _download(self, session: aiohttp.ClientSession, base_url: str, image: dict[str, str]) -> bytes:
        params = {
            "filename": image.get("filename", ""),
            "subfolder": image.get("subfolder", ""),
            "type": image.get("type", "output"),
        }
        async with session.get(f"{base_url}/view", params=params) as response:
            if response.status != 200:
                raise ImageGenerationError(f"ComfyUI下载图片失败: HTTP {response.status}")
            return await read_image_response(response)

    @staticmethod
    def _save_bytes(content: bytes, output_dir: Path, prefix: str) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{prefix}_{uuid.uuid4().hex}.png"
        path.write_bytes(content)
        return path
