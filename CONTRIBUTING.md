# 贡献指南

感谢你改进 AI 互动故事插件。提交 Issue 或 Pull Request 前，请先搜索是否已有相同问题。

## 开发环境

1. 使用 Python 3.10 或更高版本。
2. 安装运行依赖：`python -m pip install -r requirements.txt`。
3. 在 AstrBot `>=4.10.4,<5` 中加载插件进行集成验证。

## 提交前检查

```powershell
python -B -m unittest -v test_core
python -m compileall -q main.py services test_core.py scripts
python scripts/validate_release.py
```

涉及配置字段时，请同步修改 `_conf_schema.json`、`services/config.py`、README 或 `docs/CONFIGURATION.md`，并补充对应测试。

涉及版本发布时，请同步修改：

- `metadata.yaml` 的 `version`
- `main.py` 中 `@register` 的版本
- `CHANGELOG.md`
- `docs/releases/vX.Y.Z.md`

## Pull Request

- 一个 PR 聚焦一个问题，避免无关重构。
- 描述行为变化、验证方式和兼容性影响。
- 不提交 API Key、QQ 号、真实聊天记录、存档或生成图片。
- 保持普通聊天透传、房间状态和多人回档等公共行为的向后兼容。
