#!/usr/bin/env python
"""
Deterministic documentation quality benchmark for LLM-Factory.

Goal metric: llm_doc_quality_score (0-100), higher_is_better.
Target: max(baseline + 10, 80).

This is a DETERMINISTIC rubric scorer (Option B). The non-standard provider
environment (glm-5.2[1m] via proxy) makes `claude -p` unreliable in headless
mode (>90s timeouts observed), so we avoid LLM-as-judge for reproducibility.
Same repo state → same score, every run.

Scored dimensions (scope from goal.md):
  1. README.md section coverage (quickstart, install, examples, FAQ, troubleshooting)
  2. CLAUDE.md section coverage + size
  3. Per-script module docstring presence and length
     (run_classification.py, run_mlm.py, args_parser.py)
  4. configs/*.yaml comment density + "usage" header presence
  5. Executable examples (YAML parseable + script --help works)

Outputs JSON to stdout as the LAST line:
  {"primary": <0-100>, "sub_scores": {...}, "details": {...}}

Exit 0 on success, non-zero on error.
"""

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Scope (from goal.md) ---
README_PATHS = [REPO_ROOT / "README.md", REPO_ROOT / "CLAUDE.md"]
SCRIPT_PATHS = [
    REPO_ROOT / "run_classification.py",
    REPO_ROOT / "run_mlm.py",
    REPO_ROOT / "args_parser.py",
]
CONFIGS_DIR = REPO_ROOT / "configs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Dimension 1: README/CLAUDE section coverage
# ---------------------------------------------------------------------------

# Key sections that a "beginner-friendly" training toolkit should have.
# Each entry: (regex pattern to find a heading, weight)
DOC_SECTION_PATTERNS = [
    (r"(?im)^\s*#{1,4}\s*(快速上手|quick\s*start|快速开始|30\s*秒)", 2.0),   # quickstart
    (r"(?im)^\s*#{1,4}\s*(安装|install|依赖|requirements|环境)", 1.5),       # install
    (r"(?im)^\s*#{1,4}\s*(示例|example|用法|usage|使用)", 1.5),              # examples
    (r"(?im)^\s*#{1,4}\s*(常见问题|faq|q\s*&\s*a)", 1.5),                    # FAQ
    (r"(?im)^\s*#{1,4}\s*(排查|troubleshoot|调试|错误|问题定位)", 1.5),       # troubleshooting
    (r"(?im)^\s*#{1,4}\s*(配置|config|yaml)", 1.0),                          # config docs
    (r"(?im)^\s*#{1,4}\s*(功能|feature|概览|overview)", 1.0),                # feature overview
    (r"(?im)^\s*#{1,4}\s*(数据|data|dataset|数据集)", 1.0),                  # data docs
    (r"(?im)^\s*#{1,4}\s*(部署|deploy|推理|inference)", 0.5),                # deploy/inference
    (r"(?im)```", 1.0),                                                     # any code block
]


def score_doc_sections(text: str) -> tuple[float, dict]:
    """Return (0-100 score, details)."""
    if not text.strip():
        return 0.0, {"found": [], "missing": [p for p, _ in DOC_SECTION_PATTERNS]}

    total_weight = sum(w for _, w in DOC_SECTION_PATTERNS)
    found_weight = 0.0
    found = []
    missing = []
    for pat, weight in DOC_SECTION_PATTERNS:
        if re.search(pat, text):
            found_weight += weight
            found.append(pat[:40])
        else:
            missing.append(pat[:40])

    # Length bonus: reward longer docs (capped) — beginners need depth.
    line_count = len(text.splitlines())
    length_bonus = min(line_count / 200.0, 1.0) * 10.0  # up to +10

    raw = (found_weight / total_weight) * 90.0 + length_bonus
    return clamp(raw), {
        "found_count": len(found),
        "missing_count": len(missing),
        "line_count": line_count,
        "length_bonus": round(length_bonus, 2),
    }


# ---------------------------------------------------------------------------
# Dimension 2: Per-script module docstring
# ---------------------------------------------------------------------------

def score_script_docstring(path: Path) -> tuple[float, dict]:
    """Score based on top-of-module docstring presence + substance."""
    text = read_text(path)
    if not text:
        return 0.0, {"exists": False}

    try:
        tree = ast.parse(text)
        ds = ast.get_docstring(tree) if tree.body else None
    except SyntaxError:
        ds = None

    if not ds:
        # Fall back: detect a triple-quoted string at file top (after shebang/license).
        # This catches the common case of a one-line summary below the PEP 723 block.
        m = re.search(r'(?:^|\n)\s*(?:"""|\x27\x27\x27)(.+?)(?:"""|\x27\x27\x27)', text, re.DOTALL)
        if m:
            ds = m.group(1).strip()
            has_module_doc = False  # not a true PEP 224 module docstring
        else:
            return 10.0, {"exists": False, "reason": "no module docstring"}
    else:
        has_module_doc = True

    # Measure substance
    ds_lines = [l for l in ds.splitlines() if l.strip()]
    ds_chars = len(ds)

    # Reward: presence, multi-line, mentions of key concepts
    score = 30.0  # base for having any docstring
    score += min(len(ds_lines) * 5.0, 30.0)          # up to +30 for length
    score += min(ds_chars / 10.0, 25.0)              # up to +25 for detail
    # Bonus for true ast module docstring (better tooling integration)
    if has_module_doc:
        score += 10.0
    # Bonus for mentioning workflow / args / example
    for kw in ("用法", "示例", "example", "workflow", "工作流", "参数", "argument"):
        if kw.lower() in ds.lower():
            score += 1.0
    # Bonus for ASCII diagrams (workflow picture)
    if re.search(r"[\|/\\\]\[><\-]=>", ds):
        score += 5.0

    return clamp(score), {
        "exists": True,
        "is_module_docstring": has_module_doc,
        "line_count": len(ds_lines),
        "char_count": ds_chars,
    }


# ---------------------------------------------------------------------------
# Dimension 3: configs/*.yaml quality
# ---------------------------------------------------------------------------

def score_configs_dir(configs_dir: Path) -> tuple[float, dict]:
    yamls = sorted(configs_dir.glob("*.yaml"))
    if not yamls:
        return 0.0, {"yaml_count": 0}

    per_file = []
    parse_ok = 0
    try:
        import yaml  # type: ignore
        has_yaml = True
    except ImportError:
        has_yaml = False

    for yf in yamls:
        text = read_text(yf)
        lines = text.splitlines()
        non_blank = [l for l in lines if l.strip()]
        if not non_blank:
            per_file.append({"file": yf.name, "score": 0.0})
            continue

        comment_lines = [l for l in lines if l.strip().startswith("#")]
        comment_density = len(comment_lines) / max(len(non_blank), 1)

        # Has usage header (a # comment in the first 3 lines mentioning 用法/usage)
        head = "\n".join(lines[:3])
        has_usage_header = bool(re.search(r"(用法|usage|示例)", head, re.IGNORECASE))

        # Has "必改字段" / "可选项" / "用途" style structured comments
        has_structured = bool(re.search(r"(必改|可选|用途|必须修改)", text))

        # Parseable
        parseable = False
        if has_yaml:
            try:
                yaml.safe_load(text)
                parseable = True
                parse_ok += 1
            except Exception:
                parseable = False

        # Score: density (60) + usage header (15) + structured (15) + parseable (10)
        s = min(comment_density * 150.0, 60.0)
        s += 15.0 if has_usage_header else 0.0
        s += 15.0 if has_structured else 0.0
        s += 10.0 if parseable else 0.0

        per_file.append({
            "file": yf.name,
            "score": round(clamp(s), 2),
            "comment_density": round(comment_density, 3),
            "has_usage_header": has_usage_header,
            "has_structured_comments": has_structured,
            "parseable": parseable,
        })

    avg = sum(p["score"] for p in per_file) / len(per_file)
    return clamp(avg), {
        "yaml_count": len(yamls),
        "parseable_count": parse_ok,
        "per_file": per_file,
    }


# ---------------------------------------------------------------------------
# Dimension 4: Executable examples (scripts --help, YAML parse)
# ---------------------------------------------------------------------------

def score_executability() -> tuple[float, dict]:
    """Check that scripts can produce --help and configs can be loaded."""
    details = {"scripts_help": {}, "configs_parseable": 0, "configs_total": 0}
    score = 0.0
    max_score = 100.0

    # Scripts --help (weight 50%)
    script_weight = 50.0
    scripts_ok = 0
    for script in [REPO_ROOT / "run_classification.py", REPO_ROOT / "run_mlm.py"]:
        if not script.exists():
            details["scripts_help"][script.name] = False
            continue
        try:
            r = subprocess.run(
                [sys.executable, str(script), "--help"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            ok = r.returncode == 0
            details["scripts_help"][script.name] = ok
            if ok:
                scripts_ok += 1
        except Exception as e:
            details["scripts_help"][script.name] = f"error: {type(e).__name__}"
    # Normalize: 2 scripts expected
    score += (scripts_ok / 2.0) * script_weight

    # Config parseability (weight 50%)
    config_weight = 50.0
    try:
        import yaml  # type: ignore
        configs = list(CONFIGS_DIR.glob("*.yaml")) if CONFIGS_DIR.exists() else []
        details["configs_total"] = len(configs)
        ok_count = 0
        for cfg in configs:
            try:
                yaml.safe_load(cfg.read_text(encoding="utf-8"))
                ok_count += 1
            except Exception:
                pass
        details["configs_parseable"] = ok_count
        if configs:
            score += (ok_count / len(configs)) * config_weight
        else:
            score += 0.0
    except ImportError:
        # Without pyyaml we can't verify; give neutral partial credit.
        score += config_weight * 0.5

    return clamp(score), details


# ---------------------------------------------------------------------------
# Main scoring
# ---------------------------------------------------------------------------

def main() -> int:
    results = {}

    # Dimension 1: README + CLAUDE.md coverage (weight 35%)
    readme_scores = {}
    readme_details = {}
    for rp in README_PATHS:
        text = read_text(rp)
        s, d = score_doc_sections(text)
        readme_scores[rp.name] = s
        readme_details[rp.name] = d
    readme_avg = sum(readme_scores.values()) / max(len(readme_scores), 1)
    results["readme_coverage"] = {
        "score": round(readme_avg, 2),
        "per_file": {k: round(v, 2) for k, v in readme_scores.items()},
        "details": readme_details,
    }

    # Dimension 2: Script docstrings (weight 25%)
    script_scores = {}
    script_details = {}
    for sp in SCRIPT_PATHS:
        s, d = score_script_docstring(sp)
        script_scores[sp.name] = s
        script_details[sp.name] = d
    script_avg = sum(script_scores.values()) / max(len(script_scores), 1)
    results["script_docstrings"] = {
        "score": round(script_avg, 2),
        "per_file": {k: round(v, 2) for k, v in script_scores.items()},
        "details": script_details,
    }

    # Dimension 3: Configs (weight 25%)
    cfg_score, cfg_details = score_configs_dir(CONFIGS_DIR)
    results["configs_quality"] = {
        "score": round(cfg_score, 2),
        "details": cfg_details,
    }

    # Dimension 4: Executability (weight 15%)
    exec_score, exec_details = score_executability()
    results["executability"] = {
        "score": round(exec_score, 2),
        "details": exec_details,
    }

    # Weighted primary score
    primary = (
        readme_avg * 0.35
        + script_avg * 0.25
        + cfg_score * 0.25
        + exec_score * 0.15
    )
    primary = round(clamp(primary), 2)

    sub_scores = {
        "readme_coverage": round(readme_avg, 2),
        "script_docstrings": round(script_avg, 2),
        "configs_quality": round(cfg_score, 2),
        "executability": round(exec_score, 2),
    }

    output = {
        "primary": primary,
        "sub_scores": sub_scores,
        "details": results,
    }
    print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
