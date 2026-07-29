from __future__ import annotations

from pathlib import Path
from typing import Any

from astrbot.api.message_components import Image, Node, Nodes, Plain
from astrbot.core.message.message_event_result import MessageChain


async def send_roundtable_forward(event: Any, context: Any, discussion: list[dict[str, str]]) -> bool:
    if not discussion:
        return False
    sender_id = str(event.get_sender_id() or "")
    sender_name = str(event.get_sender_name() or sender_id or "玩家")
    nodes = [
        Node(
            uin=sender_id,
            name=sender_name,
            content=[Plain(f"{item.get('label', '')}：{item.get('content', '')}")],
        )
        for item in discussion
    ]
    if await _onebot_forward(event, context, nodes):
        return True
    try:
        await event.send(MessageChain([Nodes(nodes=nodes)]))
        return True
    except Exception:
        return False


async def send_generated_image(event: Any, context: Any, path: Path, generator: Any) -> bool:
    component = Image.fromFileSystem(str(path))
    if generator.kind == "openai":
        raw = generator.raw
        if str(raw.get("fake_forward_mode") or "requester") == "custom_qq":
            sender_id = "".join(ch for ch in str(raw.get("custom_qq") or "") if ch.isdigit())
            sender_name = sender_id
        else:
            sender_id = str(event.get_sender_id() or "")
            sender_name = str(event.get_sender_name() or sender_id or "玩家")
        if sender_id:
            nodes = [Node(uin=sender_id, name=sender_name, content=[component])]
            if await _onebot_forward(event, context, nodes):
                return True
    try:
        await event.send(MessageChain([component]))
        return True
    except Exception:
        return False


async def _onebot_forward(event: Any, context: Any, nodes: list[Any]) -> bool:
    get_platform = getattr(context, "get_platform_inst", None)
    if not callable(get_platform):
        return False
    platform = get_platform(event.get_platform_id())
    if platform is None:
        return False
    try:
        client = platform.get_client()
    except Exception:
        return False
    if client is None or not hasattr(client, "call_action"):
        return False
    try:
        payload: list[dict[str, Any]] = []
        for node in nodes:
            content: list[dict[str, Any]] = []
            for component in list(getattr(node, "content", []) or []):
                if isinstance(component, Plain):
                    content.append({"type": "text", "data": {"text": component.text}})
                elif isinstance(component, Image):
                    encoded = await component.convert_to_base64()
                    content.append({"type": "image", "data": {"file": f"base64://{encoded.split(',', 1)[-1]}"}})
                else:
                    return False
            payload.append(
                {
                    "type": "node",
                    "data": {
                        "uin": str(getattr(node, "uin", "") or ""),
                        "name": str(getattr(node, "name", "") or "玩家"),
                        "content": content,
                    },
                }
            )
    except Exception:
        return False
    session_id = str(event.get_group_id() or event.get_sender_id() or "")
    if not session_id.isdigit():
        return False
    try:
        if event.get_group_id():
            await client.call_action("send_group_forward_msg", group_id=int(session_id), messages=payload)
        else:
            await client.call_action("send_private_forward_msg", user_id=int(session_id), messages=payload)
        return True
    except Exception:
        return False
