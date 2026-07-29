"""Validate metadata and version consistency before publishing a release."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")
MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
EXPECTED_REPOSITORY = "https://github.com/Lan-0v0/astrbot_plugin_ai_interactive_fiction"
REQUIRED_FILES = (
    "main.py",
    "__init__.py",
    "_conf_schema.json",
    "metadata.yaml",
    "requirements.txt",
    "README.md",
    "CHANGELOG.md",
    "LICENSE",
)
RUNTIME_PATHS = REQUIRED_FILES + ("services", "logo.png")


def metadata_scalar(text: str, key: str) -> str:
    match = re.search(rf"(?m)^{re.escape(key)}:\s*(.+?)\s*$", text)
    if not match:
        raise ValueError(f"metadata.yaml 缺少 {key}")
    return match.group(1).strip().strip('"\'')


def registered_values(source: str) -> tuple[str, str]:
    tree = ast.parse(source, filename="main.py")
    plugin_name = ""
    register_version = ""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "PLUGIN_NAME" for target in node.targets):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    plugin_name = node.value.value
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "register":
            continue
        if len(node.args) >= 4 and isinstance(node.args[3], ast.Constant):
            register_version = str(node.args[3].value)
    if not plugin_name or not register_version:
        raise ValueError("main.py 缺少可识别的 PLUGIN_NAME 或 @register 版本")
    return plugin_name, register_version


def runtime_size() -> int:
    total = 0
    for relative in RUNTIME_PATHS:
        path = ROOT / relative
        if not path.exists():
            continue
        files = path.rglob("*") if path.is_dir() else (path,)
        for item in files:
            if not item.is_file() or "__pycache__" in item.parts or item.suffix == ".pyc":
                continue
            total += item.stat().st_size
    return total


def validate(tag: str | None) -> None:
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).is_file()]
    if missing:
        raise ValueError(f"缺少发布文件: {', '.join(missing)}")

    metadata_text = (ROOT / "metadata.yaml").read_text(encoding="utf-8")
    main_source = (ROOT / "main.py").read_text(encoding="utf-8")
    metadata_name = metadata_scalar(metadata_text, "name")
    metadata_version = metadata_scalar(metadata_text, "version")
    repository = metadata_scalar(metadata_text, "repo")
    plugin_name, register_version = registered_values(main_source)

    if metadata_name != plugin_name:
        raise ValueError(f"插件名称不一致: metadata={metadata_name}, main={plugin_name}")
    if not SEMVER.fullmatch(metadata_version):
        raise ValueError(f"版本不是语义化版本: {metadata_version}")
    if metadata_version != register_version:
        raise ValueError(f"版本不一致: metadata={metadata_version}, @register={register_version}")
    if repository != EXPECTED_REPOSITORY:
        raise ValueError(f"仓库地址与插件名称不匹配: {repository}")
    if not re.search(r"(?ms)^support_platforms:\s*\n(?:\s+-\s+.*\n)*?\s+-\s+aiocqhttp\s*$", metadata_text):
        raise ValueError("metadata.yaml 必须声明 aiocqhttp 支持")
    if tag and tag != f"v{metadata_version}":
        raise ValueError(f"标签与插件版本不一致: tag={tag}, version={metadata_version}")

    release_notes = ROOT / "docs" / "releases" / f"v{metadata_version}.md"
    if not release_notes.is_file():
        raise ValueError(f"缺少版本说明: {release_notes.relative_to(ROOT)}")

    with (ROOT / "_conf_schema.json").open(encoding="utf-8") as handle:
        schema = json.load(handle)
    if not isinstance(schema, dict) or not schema:
        raise ValueError("_conf_schema.json 必须是非空 JSON 对象")

    size = runtime_size()
    if size > MAX_ARCHIVE_BYTES:
        raise ValueError(f"运行文件超过 AstrBot 16 MB 限制: {size} bytes")
    print(f"release metadata valid: {plugin_name} v{metadata_version}, runtime files {size} bytes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", help="Expected Git tag, for example v0.0.1")
    args = parser.parse_args()
    validate(args.tag)


if __name__ == "__main__":
    main()
