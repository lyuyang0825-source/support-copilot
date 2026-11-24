#!/usr/bin/env python3
"""离线评测：决策是否正确、引用是否命中；可选跑全部会话质检。

默认跑 200 条开发集；`--suite core` 跑最初的 10 条边界用例。
配置了模型后可用 `--llm` 做同一套用例的 A/B。

用法：
    python3 scripts/run_evals.py
    python3 scripts/run_evals.py --suite core --verbose
    python3 scripts/run_evals.py --llm
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from support_copilot.agent import run_once  # noqa: E402
from support_copilot.llm import LLMConfig  # noqa: E402
from support_copilot.qa import score_dialog  # noqa: E402

KB = ROOT / "data" / "kb"
SUITES = {
    "core": ROOT / "data" / "evals" / "cases.json",
    "full": ROOT / "data" / "evals" / "cases_200.json",
}
TICKETS = ROOT / "data" / "tickets"
EVAL_DATE = "2026-11-24"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=sorted(SUITES), default="full")
    parser.add_argument("--tickets", action="store_true", help="同时输出全部会话质检分数")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--verbose", action="store_true", help="打印每一条用例")
    parser.add_argument("--llm", action="store_true", help="启用模型润色，跑 A/B")
    args = parser.parse_args()

    llm_config = LLMConfig.from_env()
    if args.llm and not llm_config:
        print("启用 --llm 需要设置 LLM_BASE_URL（托管服务再加 LLM_API_KEY）", file=sys.stderr)
        return 2

    cases = json.loads(SUITES[args.suite].read_text(encoding="utf-8"))
    passed = 0
    rows = []
    llm_used = 0
    by_category = defaultdict(lambda: [0, 0])

    for case in cases:
        ans = run_once(
            KB,
            case["query"],
            case.get("context") or {},
            on=EVAL_DATE,
            use_llm=True if args.llm else False,
        )
        cited = [c.id for c in ans.citations]
        ok_action = ans.action == case["expect_action"]
        ok_kb = case.get("expect_kb") is None or case["expect_kb"] in cited
        ok = ok_action and ok_kb
        passed += int(ok)
        category = case.get("category", "未分类")
        by_category[category][0] += int(ok)
        by_category[category][1] += 1
        if any(n.startswith("llm=") for n in ans.notes):
            llm_used += 1
        rows.append(
            (
                ok,
                case["query"],
                category,
                case["expect_action"],
                ans.action,
                case.get("expect_kb"),
                ",".join(cited),
            )
        )

    if args.verbose:
        print(f"{'结果':<4} {'问题':<30} {'期望':<18} {'实际':<18} {'期望知识':<16} 实际引用")
        for ok, q, _cat, exp, act, kb, cited in rows:
            print(f"{'PASS' if ok else 'FAIL':<4} {q[:28]:<30} {exp:<18} {act:<18} {str(kb):<16} {cited}")
    else:
        failures = [r for r in rows if not r[0]]
        for _ok, q, cat, exp, act, kb, cited in failures:
            print(f"FAIL [{cat}] {q}：期望 {exp}/{kb}，实际 {act}/{cited or '无引用'}")
        if not failures:
            print("全部用例通过。")

    rate = passed / max(len(cases), 1)
    print(f"\n通过 {passed}/{len(cases)}  ({rate:.1%})")
    print("\n分类：")
    for cat, (ok, total) in sorted(by_category.items()):
        print(f"  {cat:<10} {ok:>3}/{total:<3} ({ok / total:.0%})")
    if args.llm:
        print(f"\n模型润色实际生效：{llm_used}/{len(cases)} 条（其余回退规则答案）")

    code = 0 if rate >= args.threshold else 1

    if args.tickets:
        print("\n会话质检：")
        for p in sorted(TICKETS.glob("*.json")):
            data = json.loads(p.read_text(encoding="utf-8"))
            result = score_dialog(data["turns"])
            print(
                f"  {p.name:<26} {result.score:>3} 分  "
                f"{'; '.join(result.issues) if result.issues else '无问题'}"
            )
    return code


if __name__ == "__main__":
    sys.exit(main())
