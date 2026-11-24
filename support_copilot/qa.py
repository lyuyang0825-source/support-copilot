"""会话质检：对机器人已发出的对话打分，用于上线前抽检与日常巡检。

评分不是“像不像人”，而是四个可运营的指标：
引用覆盖、越权承诺、该转未转、重复询问。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .policy import detect_intent, scan_promises


@dataclass
class QAResult:
    score: int
    metrics: dict
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"score": self.score, "metrics": self.metrics, "issues": self.issues}


ESCALATION_HINTS = ("转人工", "工单", "人工确认", "人工处理")


def score_dialog(turns: list[dict]) -> QAResult:
    bot_turns = [t for t in turns if t.get("role") in ("bot", "assistant")]
    user_turns = [t for t in turns if t.get("role") == "user"]
    issues: list[str] = []
    score = 100

    cited = sum(1 for t in bot_turns if t.get("citation_ids"))
    coverage = cited / max(len(bot_turns), 1)
    if coverage < 1.0:
        penalty = int(30 * (1 - coverage))
        score -= penalty
        issues.append(f"引用覆盖率 {coverage:.0%}：{len(bot_turns) - cited} 条机器人回复没有引用来源")

    promises = []
    for t in bot_turns:
        promises.extend(scan_promises(t.get("text", "")))
    if promises:
        score -= min(30, 10 * len(promises))
        issues.append(f"越权承诺：{sorted(set(promises))}")

    risky = [t for t in user_turns if detect_intent(t.get("text", "")).human_required]
    escalated = any(
        t.get("escalated")
        or any(h in t.get("text", "") for h in ESCALATION_HINTS)
        for t in bot_turns
    )
    if risky and not escalated:
        score -= 25
        issues.append(f"{len(risky)} 轮涉及退款/赔偿/账号等高风险诉求，但全程未转人工")
    elif escalated and not risky:
        score -= 5
        issues.append("简单咨询也触发了转人工，自动化率偏低")

    repeated = 0
    for t in bot_turns:
        text = t.get("text", "")
        if "订单号" in text and any("订单号" in u.get("text", "") for u in user_turns):
            repeated += 1
    if repeated:
        score -= 10
        issues.append("用户已提供订单号，机器人仍重复询问")

    resolved = any(("已创建工单" in t.get("text", "") or "已为您" in t.get("text", "")) for t in bot_turns)
    if not resolved:
        score -= 5
        issues.append("结尾缺少明确的下一步（工单号或办理路径）")

    metrics = {
        "bot_turns": len(bot_turns),
        "user_turns": len(user_turns),
        "citation_coverage": round(coverage, 2),
        "promise_hits": len(promises),
        "risky_user_turns": len(risky),
        "escalated": escalated,
        "resolved_signal": resolved,
    }
    return QAResult(score=max(0, score), metrics=metrics, issues=issues)
