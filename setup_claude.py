#!/usr/bin/env python3
"""Claude Code 一键配置脚本"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def log_info(msg):
    print(f"🔹 {msg}")


def log_success(msg):
    print(f"✅ {msg}")


def log_error(msg):
    print(f"❌ {msg}", file=sys.stderr)


def check_nodejs():
    """检查 Node.js 是否安装，版本 >= 18"""
    if shutil.which("node") is None:
        log_error("Node.js 未安装，请先安装 Node.js >= 18")
        log_info("安装命令: curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash")
        sys.exit(1)

    result = subprocess.run(["node", "-v"], capture_output=True, text=True)
    version = result.stdout.strip().lstrip("v")
    major = int(version.split(".")[0])

    if major < 18:
        log_error(f"Node.js 版本过低: v{version}，需要 >= 18")
        sys.exit(1)

    log_success(f"Node.js v{version}")


def install_claude_code():
    """安装 Claude Code"""
    if shutil.which("claude") is not None:
        result = subprocess.run(["claude", "--version"], capture_output=True, text=True)
        log_success(f"Claude Code 已安装: {result.stdout.strip()}")
        return

    log_info("正在安装 Claude Code...")
    result = subprocess.run(
        ["npm", "install", "-g", "@anthropic-ai/claude-code"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log_error(f"安装失败: {result.stderr}")
        sys.exit(1)

    log_success("Claude Code 安装完成")


def install_plugins(config, dry_run=False):
    """离线安装插件到 ~/.claude/plugins/"""
    plugins_dir = config.get("plugins_dir", "deploy/plugins")
    plugins_src = Path(plugins_dir)
    if not plugins_src.is_absolute():
        plugins_src = Path(__file__).parent / plugins_src

    if not plugins_src.exists():
        log_error(f"插件目录不存在: {plugins_src}")
        return False

    home = Path.home()
    plugins_dst = home / ".claude" / "plugins"

    # 1. 复制 cache 目录（只追加，不覆盖）
    cache_src = plugins_src / "cache"
    if cache_src.exists():
        for marketplace_dir in cache_src.iterdir():
            if not marketplace_dir.is_dir():
                continue
            dst = plugins_dst / "cache" / marketplace_dir.name
            if dst.exists():
                # 只复制不存在的插件子目录
                for plugin_dir in marketplace_dir.iterdir():
                    plugin_dst = dst / plugin_dir.name
                    if not plugin_dst.exists():
                        if dry_run:
                            log_info(f"[dry-run] 将复制: {plugin_dir} -> {plugin_dst}")
                        else:
                            shutil.copytree(plugin_dir, plugin_dst)
                            log_success(f"已安装插件: {plugin_dir.name}")
                    else:
                        log_info(f"插件已存在，跳过: {plugin_dir.name}")
            else:
                if dry_run:
                    log_info(f"[dry-run] 将复制: {marketplace_dir} -> {dst}")
                else:
                    shutil.copytree(marketplace_dir, dst)
                    log_success(f"已安装 marketplace: {marketplace_dir.name}")

    # 2. 复制 marketplaces 目录
    mkt_src = plugins_src / "marketplaces"
    if mkt_src.exists():
        for mkt_dir in mkt_src.iterdir():
            if not mkt_dir.is_dir():
                continue
            dst = plugins_dst / "marketplaces" / mkt_dir.name
            if not dst.exists():
                if dry_run:
                    log_info(f"[dry-run] 将复制: {mkt_dir} -> {dst}")
                else:
                    shutil.copytree(mkt_dir, dst)
                    log_success(f"已安装 marketplace 源: {mkt_dir.name}")
            else:
                log_info(f"marketplace 已存在，跳过: {mkt_dir.name}")

    # 3. 处理 installed_plugins.json（只保留实际部署的插件，替换路径）
    installed_src = plugins_src / "installed_plugins.json"
    if installed_src.exists():
        with open(installed_src) as f:
            installed = json.load(f)

        # 只保留 cache 中实际存在的插件
        cache_src = plugins_src / "cache"
        deployed_plugin_ids = set()
        if cache_src.exists():
            for marketplace_dir in cache_src.iterdir():
                if not marketplace_dir.is_dir():
                    continue
                for plugin_dir in marketplace_dir.iterdir():
                    if plugin_dir.is_dir():
                        # 格式: plugin_name@marketplace_name
                        deployed_plugin_ids.add(f"{plugin_dir.name}@{marketplace_dir.name}")

        # 过滤并替换路径
        filtered_plugins = {}
        for plugin_id, entries in installed.get("plugins", {}).items():
            if plugin_id not in deployed_plugin_ids:
                continue
            for entry in entries:
                old_path = entry.get("installPath", "")
                if "/cache/" in old_path:
                    rel_path = old_path.split("/cache/")[1]
                    entry["installPath"] = str(plugins_dst / "cache" / rel_path)
            filtered_plugins[plugin_id] = entries

        installed["plugins"] = filtered_plugins

        installed_dst = plugins_dst / "installed_plugins.json"
        if dry_run:
            log_info(f"[dry-run] 将写入: {installed_dst}（{len(filtered_plugins)} 个插件）")
        else:
            plugins_dst.mkdir(parents=True, exist_ok=True)
            with open(installed_dst, "w") as f:
                json.dump(installed, f, indent=2, ensure_ascii=False)
            log_success(f"已写入插件注册表: {installed_dst}（{len(filtered_plugins)} 个插件）")

    # 4. 处理 known_marketplaces.json（只保留已部署的 marketplace）
    known_src = plugins_src / "known_marketplaces.json"
    if known_src.exists():
        with open(known_src) as f:
            known = json.load(f)

        # 只保留已部署的 marketplace
        deployed_mkt = set()
        if cache_src.exists():
            for marketplace_dir in cache_src.iterdir():
                if marketplace_dir.is_dir():
                    deployed_mkt.add(marketplace_dir.name)
        mkt_src = plugins_src / "marketplaces"
        if mkt_src.exists():
            for mkt_dir in mkt_src.iterdir():
                if mkt_dir.is_dir():
                    deployed_mkt.add(mkt_dir.name)

        filtered_known = {k: v for k, v in known.items() if k in deployed_mkt}

        known_dst = plugins_dst / "known_marketplaces.json"
        if dry_run:
            log_info(f"[dry-run] 将写入: {known_dst}（{len(filtered_known)} 个 marketplace）")
        else:
            plugins_dst.mkdir(parents=True, exist_ok=True)
            with open(known_dst, "w") as f:
                json.dump(filtered_known, f, indent=2, ensure_ascii=False)
            log_success(f"已写入 marketplace 注册表: {known_dst}（{len(filtered_known)} 个 marketplace）")

    return True


def write_settings(config, dry_run=False):
    """写入 ~/.claude/settings.json"""
    settings_dir = Path.home() / ".claude"
    settings_dir.mkdir(exist_ok=True)
    settings_file = settings_dir / "settings.json"

    # 读取已有配置
    if settings_file.exists():
        with open(settings_file) as f:
            settings = json.load(f)
    else:
        settings = {}

    # 更新 env 配置
    settings["env"] = {
        "ANTHROPIC_AUTH_TOKEN": config["api_key"],
        "ANTHROPIC_BASE_URL": config["api_base_url"],
        "ANTHROPIC_MODEL": config["model"],
        "API_TIMEOUT_MS": str(config["api_timeout_ms"]),
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }

    # 合并插件配置（不覆盖已有）
    if config.get("plugins_enabled", True):
        enabled = settings.get("enabledPlugins", {})
        for plugin_id in [
            "superpowers@claude-plugins-official",
            "code-simplifier@claude-plugins-official",
            "ralph-loop@claude-plugins-official",
            "code-review@claude-plugins-official",
            "frontend-design@claude-plugins-official",
        ]:
            enabled.setdefault(plugin_id, True)
        settings["enabledPlugins"] = enabled

        marketplaces = settings.get("extraKnownMarketplaces", {})
        marketplaces.setdefault("superpowers-marketplace", {
            "source": {"source": "github", "repo": "obra/superpowers-marketplace"}
        })
        marketplaces.setdefault("claude-plugins-official", {
            "source": {"source": "github", "repo": "anthropics/claude-plugins-official"}
        })
        settings["extraKnownMarketplaces"] = marketplaces

    if dry_run:
        log_info(f"[dry-run] 将写入: {settings_file}")
        log_info(f"[dry-run] settings 内容: {json.dumps(settings, indent=2, ensure_ascii=False)}")
    else:
        with open(settings_file, "w") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        log_success(f"配置已写入 {settings_file}")


def write_onboarding():
    """设置 hasCompletedOnboarding"""
    config_file = Path.home() / ".claude.json"

    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
    else:
        config = {}

    config["hasCompletedOnboarding"] = True

    with open(config_file, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    log_success("Onboarding 标记已设置")


def main():
    parser = argparse.ArgumentParser(description="Claude Code 一键配置")
    parser.add_argument("config", help="配置文件路径")
    parser.add_argument("--dry-run", action="store_true", help="仅打印操作，不实际执行")
    parser.add_argument("--skip-plugins", action="store_true", help="跳过插件安装")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        log_error(f"配置文件不存在: {config_path}")
        sys.exit(1)

    # 读取配置
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 验证必填项
    if not config.get("api_key") or config["api_key"] == "your-api-key-here":
        log_error("请在配置文件中填写 api_key")
        sys.exit(1)

    # 设置默认值
    config.setdefault("api_base_url", "https://open.bigmodel.cn/api/anthropic")
    config.setdefault("model", "claude-sonnet-4-6")
    config.setdefault("api_timeout_ms", 3000000)
    config.setdefault("plugins_enabled", True)

    dry_run = args.dry_run
    if dry_run:
        print("🔍 [dry-run 模式] 仅打印操作，不实际执行\n")

    print("🚀 开始配置 Claude Code\n")

    check_nodejs()
    install_claude_code()

    # 安装插件
    if config.get("plugins_enabled", True) and not args.skip_plugins:
        install_plugins(config, dry_run=dry_run)

    write_settings(config, dry_run=dry_run)
    if not dry_run:
        write_onboarding()

    print()
    log_success("🎉 配置完成！运行 `claude` 即可使用")


if __name__ == "__main__":
    main()
