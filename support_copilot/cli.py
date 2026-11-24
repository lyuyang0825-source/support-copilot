from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import SupportAgent, run_once
from .kb import KnowledgeBase
from .qa import score_dialog

DEMO_QUERIES = [
    ("退款多久到账？", {}),
    ("我要退款，订单号 A20261124001", {"order_id": "A20261124001"}),
    ("价保几天？帮我补差价，订单号 A20261124002", {"order_id": "A20261124002", "product": "降噪耳机"}),
    ("账号被冻结了怎么解封", {}),
    ("发票抬头能改吗", {}),
    ("你们支持火星配送吗", {}),
]


def _print_answer(ans, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(ans.to_dict(), ensure_ascii=False, indent=2))
        return
    print(f"\n[{ans.action}] intent={ans.intent} confidence={ans.confidence}")
    print(ans.reply)
    if ans.citations:
        print("\n引用：" + "、".join(f"{c.id}({c.score})" for c in ans.citations))
    if ans.ticket:
        print(f"工单：{ans.ticket.ticket_id} → {ans.ticket.queue}；缺失信息：{ans.ticket.missing_slots}")
        print(f"交接摘要：{ans.ticket.handoff_summary}")
    if ans.notes:
        print("备注：" + "；".join(ans.notes))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="可溯源、会认怂、会交接的客服副驾驶")
    parser.add_argument("--query", "-q", help="用户问题")
    parser.add_argument("--context", default="{}", help='JSON 上下文，如 {"order_id":"A1"}')
    parser.add_argument("--kb", default="data/kb")
    parser.add_argument("--date", help="按指定日期加载生效知识（YYYY-MM-DD）")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--lint", action="store_true", help="检查知识库冲突与质量问题")
    parser.add_argument("--eval", help="对会话 JSON 跑质检")
    parser.add_argument("--demo", action="store_true", help="跑内置演示问题集")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    kb = KnowledgeBase.load(args.kb)

    if args.lint:
        issues = kb.lint()
        print("\n".join(issues) if issues else "知识库检查通过：无冲突、无缺失字段。")
        return 1 if any(i.startswith("冲突") for i in issues) else 0

    if args.eval:
        data = json.loads(Path(args.eval).read_text(encoding="utf-8"))
        result = score_dialog(data["turns"])
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0

    if args.demo:
        out_dir = Path("demo_out")
        out_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for q, ctx in DEMO_QUERIES:
            ans = run_once(args.kb, q, ctx, args.date)
            _print_answer(ans)
            results.append(ans.to_dict())
        (out_dir / "answers.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        lines = ["# 客服副驾驶演示结果", ""]
        for r in results:
            lines.append(f"## {r['query']}")
            lines.append("")
            lines.append(f"- 动作：`{r['action']}`｜置信度 {r['confidence']}｜意图 `{r['intent']}`")
            lines.append(f"- 引用：{'、'.join(c['id'] for c in r['citations']) or '—'}")
            if r["ticket"]:
                lines.append(f"- 工单：{r['ticket']['ticket_id']} → {r['ticket']['queue']}")
            lines.append("")
            lines.append(r["reply"].replace("\n", "\n\n"))
            lines.append("")
        (out_dir / "demo.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"\n[支持] 结果已写入 {out_dir/'answers.json'} 与 {out_dir/'demo.md'}")
        return 0

    if not args.query:
        parser.error("请提供 --query，或使用 --demo / --lint / --eval")

    context = json.loads(args.context)
    ans = run_once(args.kb, args.query, context, args.date)
    _print_answer(ans, args.json)
    return 2 if ans.action == "refuse_and_escalate" else 0


if __name__ == "__main__":
    sys.exit(main())
