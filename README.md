# astrbot_plugin_ai_interactive_fiction

[![AstrBot](https://img.shields.io/badge/AstrBot-%3E%3D4.10.4%2C%3C5-4c7dff)](https://github.com/AstrBotDevs/AstrBot)
[![Release](https://img.shields.io/github/v/release/Lan-0v0/astrbot_plugin_ai_interactive_fiction?display_name=tag)](https://github.com/Lan-0v0/astrbot_plugin_ai_interactive_fiction/releases)
[![License](https://img.shields.io/github/license/Lan-0v0/astrbot_plugin_ai_interactive_fiction)](LICENSE)

面向 AstrBot 的多模型 AI 互动故事插件。故事正文由圆桌提案模型讨论、评审模型依次评审改写，最后一名成功评审给出玩家实际看到的结果。支持单人故事、OneBot 群聊多人同游、自然语言操作、存读档、长记忆压缩和人物绘图。

当前版本：`v0.0.2`

## 功能特性

- 普通聊天与游玩消息共存：仅在全局判断 LLM 确认消息属于插件操作或玩家行动时消费事件。
- AstrBot 原生注册：故事、存读档与圆桌会议会显示在指令管理中，并提供自然语言互动故事函数工具。
- 圆桌会议：支持轮流讨论或单独提案、优先级排序、多轮讨论和常规/非安全模型路由。
- 多人同游：房主创建共享世界，每位玩家拥有独立角色，首条有效行动进入房间行动锁。
- 存档回溯：每人四个手动槽和一个自动槽；多人读档会同步回溯共享世界。
- 记忆管理：支持全部记忆、长期自动压缩，以及超过上下文时的临时压缩。
- 人物绘图：支持 OpenAI 兼容图像接口与本地 ComfyUI API 工作流。
- OneBot 合并转发：可查看最近一次行动完整的圆桌讨论记录。

## 环境要求

- AstrBot `>=4.10.4,<5`
- Python `>=3.10`
- OneBot v11 / `aiocqhttp` 平台适配器
- 至少一个已在 AstrBot 中配置的 LLM 提供商

人物绘图是可选能力。使用 ComfyUI 时，需要可访问的 ComfyUI 服务和 API 格式工作流 JSON。

## 安装

### AstrBot WebUI

在插件管理页选择从 GitHub 仓库安装，并填写：

```text
https://github.com/Lan-0v0/astrbot_plugin_ai_interactive_fiction
```

### 手动安装

将仓库目录放入 AstrBot 的 `data/plugins/astrbot_plugin_ai_interactive_fiction`，安装依赖后重启 AstrBot：

```powershell
python -m pip install -r requirements.txt
```

## 最小配置

首次使用至少需要：

1. 选择“全局判断 LLM”。
2. 在“圆桌会议”中添加至少一名常规提案模型并选择提供商；新条目默认启用。
3. 添加至少一名常规评审模型并选择提供商；可直接使用预设人设或自行修改。
4. 保存配置并重载插件。

若缺少常规提案或评审配置，插件会明确提示缺少模型配置。完整字段说明见 [配置指南](docs/CONFIGURATION.md)。

## 指令

| 指令 | 作用 |
| --- | --- |
| `/故事` | 显示完整帮助 |
| `/故事 开始 [要求]` | 开始故事；可携带主角或故事要求 |
| `/故事 加入@房主 [角色要求]` | 加入房主的多人故事 |
| `/故事 结束` | 房主结束故事，参与者退出当前房间 |
| `/存档 1` | 保存到 1 至 4 号个人存档槽 |
| `/读档 1` | 从 1 至 4 号个人存档槽读档 |
| `/圆桌会议` | 查看最近一次行动的圆桌讨论 |

启用“自然语言”后，开始、加入、结束、存读档、查看圆桌以及实际行动都可以直接用正常话语表达。插件判断为普通聊天时，会将消息继续交给 AstrBot 的正常聊天链路。

## 圆桌规则

提案和评审模型均按优先级从低到高运行，同优先级随机。高优先级模型因此能看到更多前序内容；最后一名成功的评审模型负责最终输出。

非安全内容优先提交给对应类型的模型；对应类型缺失时回退到常规模型。失败模型会被跳过并记录 `URL + 模型名称 生成失败`；全部提案失败时不会继续下一轮。

`/圆桌会议` 只展示模型讨论，不展示开局底稿。讨论记录仅保留最近一次行动，不会写入长期故事记忆。

## 多人与存档

每名玩家同一时间只能处于一个故事房间。多人房间共享世界状态，但每位玩家拥有独立角色和存档槽。任意参与者读档都会影响所有人的共享进度。

读档回到其他玩家加入前时，这些角色会从房间移除。其下一次尝试行动会收到“角色因世界回溯消失，请重新加入”，重新加入后再生成角色。

运行数据位于 AstrBot 的 `data/plugin_data/astrbot_plugin_ai_interactive_fiction`。故事结束后对应数据立即清理；未结束故事的缓存按最后活跃时间和面板中的自动清理天数处理。

## 人物绘图

- OpenAI 兼容接口：支持多个 API Key、优先级、常规/自由类型与 `quality=high` 失败降级重试。
- ComfyUI：通过工作流名称、节点 ID 和字段路径注入正向提示词或改图输入。
- 首次登场生成新立绘；后续相关 CG 尽量使用缓存原图和提示词进行改图，以保持人物一致性。

人物绘图失败不会中断故事，只发送“CG生成失败，请配置或检查模型”。

## 开发与测试

```powershell
python -m pip install -r requirements.txt
python -B -m unittest -v test_core test_astrbot_integration
python scripts/validate_release.py
```

项目结构和核心数据流见 [架构说明](docs/ARCHITECTURE.md)，贡献规范见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 发布

推送符合语义化版本的标签（例如 `v0.0.2`）后，GitHub Actions 会运行测试、检查版本一致性、构建小于 16 MB 的插件压缩包并创建 GitHub Release。详细步骤见 [发布指南](docs/RELEASING.md)。

## 许可证

本项目使用 [MIT License](LICENSE)。
