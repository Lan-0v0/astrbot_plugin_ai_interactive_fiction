# 发布指南

本仓库使用语义化版本，Git 标签使用 `vX.Y.Z` 格式。插件内部版本不带 `v` 前缀。

## 发布前

1. 修改 `metadata.yaml` 中的 `version`。
2. 修改 `main.py` 中 `@register` 的版本参数。
3. 在 `CHANGELOG.md` 顶部增加对应版本。
4. 新增 `docs/releases/vX.Y.Z.md`。
5. 运行验证：

```powershell
python -B -m unittest -v test_core test_astrbot_integration
python -m compileall -q main.py services test_core.py test_astrbot_integration.py scripts
python scripts/validate_release.py
```

## 创建发布

提交所有变更后创建并推送标签：

```powershell
git tag -a v0.0.3 -m "v0.0.3"
git push origin main
git push origin v0.0.3
```

标签会触发 `.github/workflows/release.yml`：

1. 校验标签、`metadata.yaml` 和 `@register` 版本一致。
2. 执行单元测试和 Python 编译检查。
3. 只打包插件运行所需文件，排除测试、脚本、仓库配置和缓存。
4. 检查 ZIP 不超过 AstrBot 插件市场要求的 16 MB。
5. 创建 GitHub Release 并附加 `astrbot_plugin_ai_interactive_fiction-vX.Y.Z.zip`。

## 发布 AstrBot 插件市场

GitHub Release 成功后，前往 [AstrBot 插件发布页面](https://cloud.astrbot.app/publish) 填写仓库信息。`metadata.yaml` 是插件市场读取名称、描述、版本、平台和仓库地址的来源。

## v0.0.1

首个版本的发布说明见 [v0.0.1](releases/v0.0.1.md)。

## v0.0.2

原生指令、函数工具、配置预设与回归测试版本的发布说明见 [v0.0.2](releases/v0.0.2.md)。

## v0.0.3

配置副标题、分类人设与新版帮助菜单的发布说明见 [v0.0.3](releases/v0.0.3.md)。
