#!/usr/bin/env python3
"""离线评测：决策是否正确、引用是否命中；可选跑全部会话质检。

用法：
    python3 scripts/run_evals.py
    python3 scripts/run_evals.py --tickets
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from support_copilot.agent import run_once  # noqa: E402
from support_copilot.qa import score_dialog  # noqa: E402

KB = ROOT / "data" / "kb"
CASES = ROOT / "data" / "evals" / "cases.json"
TICKETS = ROOT / "data" / "tickets"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickets", action="store_true", help="同时输出全部会话质检分数")
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    cases = json.loads(CASES.read_text(encoding="utf-8"))
    passed = 0
    rows = []
    for case in cases:
        ans = run_once(KB, case["query"], case.get("context") or {})
        cited = [c.id for c in ans.citations]
        ok_action = ans.action == case["expect_action"]
        ok_kb = case.get("expect_kb") is None or case["expect_kb"] in cited
        ok = ok_action and ok_kb
        passed += int(ok)
        rows.append((ok, case["query"], case["expect_action"], ans.action, case.get("expect_kb"), ",".join(cited)))

    print(f"{'结果':<4} {'问题':<28} {'期望':<18} {'实际':<18} {'期望知识':<16} 实际引用")
    for ok, q, exp, act, kb, cited in rows:
        print(f"{'PASS' if ok else 'FAIL':<4} {q[:26]:<28} {exp:<18} {act:<18} {str(kb):<16} {cited}")

    rate = passed / max(len(cases), 1)
    print(f"\n通过 {passed}/{len(cases)}  ({rate:.0%})")
    code = 0 if rate >= args.threshold else 1

    if args.tickets:
        print("\n会话质检：")
        for p in sorted(TICKETS.glob("*.json")):
            data = json.loads(p.read_text(encoding="utf-8"))
            result = score_dialog(data["turns"])
            print(f"  {p.name:<26} {result.score:>3} 分  {'; '.join(result.issues) if result.issues else '无问题'}")
    return code


if __name__ == "__main__":
    sys.exit(main())
