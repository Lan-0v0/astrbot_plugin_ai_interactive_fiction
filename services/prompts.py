from __future__ import annotations

import json
from typing import Any

from .config import StoryConfig


def story_creation_task(
    story: StoryConfig | None,
    *,
    player_requirements: str,
    owner_name: str,
) -> str:
    if story is None:
        constraints = (
            "这是全随机模式。随机选择适合约10至40分钟一局的题材、预期时长、世界观、主角和是否启用存档；"
            "若玩家提出要求则必须融入。"
        )
    else:
        constraints = json.dumps(
            {
                "故事名称": story.name,
                "预期分钟": story.expected_minutes,
                "自定义世界观": story.world or "由AI补充",
                "主角描述": story.protagonist or "由AI补充",
                "必要标签": story.required_tags,
                "功能机制": sorted(story.mechanisms),
            },
            ensure_ascii=False,
        )
    return f"""为一局短篇互动故事生成隐藏底稿和房主角色。开局绝不能向玩家泄露世界观、NPC、剧情、目标或伏笔。
约束：{constraints}
玩家额外要求：{player_requirements or '无'}
玩家显示名：{owner_name or '玩家'}

只输出一个JSON对象，不加代码块：
{{
  "story_bible": {{
    "title": "故事内部标题",
    "world": "完整隐藏世界观",
    "tone": ["基调"],
    "plot": "隐藏剧情脉络与约20分钟节奏规划",
    "npcs": [{{"id":"稳定ID","name":"姓名","profile":"身份性格外观","secrets":"隐藏信息"}}],
    "rules": "世界规则",
    "ending_conditions": "自然结局条件"
  }},
  "public_player_profile": {{"name":"角色名","age":"年龄","appearance":"外貌服饰","identity":"仅角色自己知道且不泄露世界的身份信息","abilities":"自身能力","inventory":"自身初始持有物"}},
  "private_player_profile": {{"background":"不公开背景","story_links":"与隐藏剧情的联系"}},
  "opening_state": "供后续故事模型使用的隐藏开场状态，不直接展示",
  "runtime": {{"expected_minutes":20,"save_enabled":true}}
}}
必要标签必须真实进入底稿。public_player_profile只能包含角色自身可知信息。"""


def join_character_task(
    *,
    bible: dict[str, Any],
    owner_character: dict[str, Any],
    requirements: str,
    player_name: str,
) -> str:
    return f"""为加入现有多人故事的新玩家创建独立角色。角色必须符合隐藏世界观，并与房主角色处于相近的身份、能力、资源或叙事水平；没有战力体系时按题材选择合适维度。
隐藏故事底稿：{json.dumps(bible, ensure_ascii=False)}
房主角色：{json.dumps(owner_character, ensure_ascii=False)}
新玩家显示名：{player_name or '玩家'}
新玩家要求：{requirements or '无，由AI全随机生成'}

只输出JSON，不加代码块：
{{"public_player_profile":{{"name":"角色名","age":"年龄","appearance":"外貌服饰","identity":"角色自身可知身份","abilities":"自身能力","inventory":"自身持有物"}},"private_player_profile":{{"background":"隐藏背景","story_links":"隐藏联系"}}}}
公开资料不能泄露世界观、NPC、剧情或伏笔。"""


def action_task(
    *,
    story: StoryConfig,
    bible: dict[str, Any],
    world_state: str,
    memory_context: str,
    characters: dict[str, Any],
    actor_id: str,
    action: str,
    content_type: str,
    forbid_player_autonomy: bool,
    current_choices: list[str],
    include_psychology: bool,
) -> str:
    autonomy = (
        "严禁替玩家补充任何未明确说出的主动行动。只裁定当前行动产生的结果。"
        if forbid_player_autonomy
        else "不要无必要地替玩家扩展主动行动。"
    )
    return f"""{story.content_limit}

你要裁定一条互动故事行动。{autonomy}
玩家只负责行动选择，行动是否成功及一切结果由你根据世界规则、当前状态和前文决定。被迫承受的事情属于结果，不属于玩家主动行动。
本次内容路径：{content_type}
预期整局时长：{story.expected_minutes}分钟，请动态控制节奏并允许自然结束。
隐藏故事底稿：{json.dumps(bible, ensure_ascii=False)}
当前世界状态：{world_state or '开局'}
故事记忆：{memory_context or '无'}
所有玩家角色：{json.dumps(characters, ensure_ascii=False)}
上一轮给出的候选行动：{json.dumps(current_choices, ensure_ascii=False) if current_choices else '无'}
行动者ID：{actor_id}
玩家原始行动（不得改写成已成功的事实）：{action}
本轮是否确有必要附带心理描写：{include_psychology}。即使为true也不得替玩家决定感受、意志或后续行动，且不超过50字。

只输出JSON对象，不加代码块：
{{
  "narrative":"直接发给玩家的叙事正文；内容限制中的字数仅计算此字段",
  "psychology":"通常留空；仅在被要求时给出不超过50字且不控制玩家自由意志的心理描写",
  "choices":["接下来可选择的行动1","行动2","行动3"],
  "state_summary":"行动后供下一轮使用的客观世界状态",
  "death":false,
  "story_ended":false,
  "major_node":false,
  "new_characters":[{{"id":"稳定ID","name":"姓名","age":"年龄","appearance":"具体外观与服饰","profile":"身份性格；不得泄露秘密"}}],
  "changed_characters":[{{"id":"已有稳定ID","age_changed":false,"clothes_changed":false,"appearance":"变化后的外观服饰"}}],
  "cg_trigger":"none|violation|killing",
  "cg_character_id":"相关角色稳定ID"
}}
以候选行动选项为主，choices通常给出2至4项并固定显示；玩家仍可不选这些选项而自由描述合理行动。死亡或故事自然结束时choices留空。
侵犯或杀害行动本身不得强制结束故事；产生结果后，若目标和情境仍允许，应继续给出与同一角色互动的其他候选行动及结束当前互动的选择。
玩家死亡时只设置death=true，不要设置story_ended=true，后续由玩家自行选择读档或结束故事。
new_characters仅列本轮第一次正式登场的角色。年龄或服饰有较大变化时才写changed_characters。死亡时narrative中不要写英文死亡提示，插件会单独追加。"""


def compression_prompt(previous_summary: str, history: list[dict[str, Any]]) -> str:
    return (
        "把互动故事记忆压缩成可供后续续写的客观摘要。保留人物关系、物品、伤势、地点、承诺、已知线索、"
        "未解决事件和世界状态；不要添加新事实。只输出摘要正文。\n"
        f"既有摘要：{previous_summary or '无'}\n"
        f"待压缩记录：{json.dumps(history, ensure_ascii=False)}"
    )


def image_prompt_task(
    *,
    bible: dict[str, Any],
    character: dict[str, Any],
    mode: str,
    event_context: str,
    original_prompt: str = "",
) -> str:
    if mode == "edit":
        instruction = (
            "生成一段高质量改图提示词。必须最大限度保持原角色的脸、发型、体型和画风一致，只改变剧情要求的动作、服饰或状态。"
            f"原始立绘提示词：{original_prompt}"
        )
    else:
        instruction = "生成一段最高质量的单人角色立绘提示词，清楚描述脸部、发型、体型、服饰、年龄感、姿态和画风，以便后续改图保持一致。"
    return (
        f"{instruction}\n只输出可直接提交给图片模型的提示词，不加解释。\n"
        f"隐藏故事风格资料：{json.dumps(bible, ensure_ascii=False)}\n"
        f"角色资料：{json.dumps(character, ensure_ascii=False)}\n"
        f"当前事件：{event_context or '首次登场'}"
    )
