## 变更说明

说明修改了什么，以及解决的具体问题。

## 行为影响

说明对普通聊天透传、圆桌生成、多人状态、存读档、记忆或绘图链路的影响。

## 验证

- [ ] `python -B -m unittest -v test_core`
- [ ] `python -m compileall -q main.py services test_core.py scripts`
- [ ] `python scripts/validate_release.py`
- [ ] 涉及配置时已同步 schema、配置解析和文档
- [ ] 未提交 API Key、QQ 号、聊天记录、存档、日志或生成图片

## 兼容性

列出 AstrBot 版本、OneBot 实现和需要的迁移步骤；没有兼容性变化时填写“无”。
