"""客服副驾驶：检索 → 策略 → 回复 / 工单。

回答永远带引用；不确定就转人工；转人工时把已收集信息打包，避免用户复述。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path

from .bm25 import BM25
from .kb import KBEntry, KnowledgeBase
from .llm import LLMConfig, polish_body
from .policy import (
    QUEUE_MAP,
    confidence_from_score,
    decide,
    detect_intent,
    mask_pii,
    scan_promises,
    untraceable_numbers,
)

REQUIRED_SLOTS = {
    "refund": ["order_id"],
    "compensation": ["order_id"],
    "account": ["account_id"],
    "invoice": ["order_id"],
    "shipping": ["order_id"],
    "warranty": ["order_id", "product"],
    "price": ["order_id", "product"],
    "complaint": ["order_id"],
    "legal": ["order_id"],
    "contract": ["contract_id"],
}

GREETINGS = ("你好", "您好", "在吗", "在么", "hi", "hello", "有人吗")
OUT_OF_SCOPE_TERMS = ("火星", "月球", "外太空", "国际空间站")

# 意图与知识条目对齐：避免“退款后券会回来吗”被退款条目抢走
INTENT_KB_PREFIX = {
    "refund": "kb-refund",
    "price": "kb-price",
    "account": "kb-account",
    "invoice": "kb-invoice",
    "shipping": "kb-ship",
    "warranty": "kb-warranty",
    "coupon": "kb-coupon",
    "hours": "kb-hours",
}


@dataclass
class Citation:
    id: str
    question: str
    answer: str
    owner: str
    effective_from: str
    effective_to: str
    primary: bool = True
    score: float = 0.0


@dataclass
class Ticket:
    ticket_id: str
    created_at: str
    queue: str
    reason: str
    intent: str
    confidence: float
    customer_query: str
    collected: dict
    missing_slots: list[str]
    checked_kb: list[str]
    handoff_summary: str
    do_not_reask: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Answer:
    query: str
    reply: str
    action: str
    confidence: float
    intent: str
    citations: list[Citation] = field(default_factory=list)
    ticket: Ticket | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "reply": self.reply,
            "action": self.action,
            "confidence": self.confidence,
            "intent": self.intent,
            "citations": [asdict(c) for c in self.citations],
            "ticket": self.ticket.to_dict() if self.ticket else None,
            "notes": self.notes,
        }


class SupportAgent:
    def __init__(
        self,
        kb: KnowledgeBase,
        today: date | None = None,
        llm: LLMConfig | None = None,
        use_llm: bool | None = None,
    ):
        self.kb = kb
        self.today = today or date.today()
        self.entries: list[KBEntry] = kb.active(self.today)
        self.bm25 = BM25([e.text for e in self.entries]) if self.entries else None
        self.llm = llm if llm is not None else LLMConfig.from_env()
        # 默认：配置了就启用；显式传 False 可强制关掉，便于评测做 A/B。
        self.use_llm = bool(self.llm) if use_llm is None else bool(use_llm and self.llm)

    def _cite(self, entry: KBEntry, score: float, primary: bool) -> Citation:
        return Citation(
            id=entry.id,
            question=entry.question,
            answer=entry.answer,
            owner=entry.owner,
            effective_from=entry.effective_from.isoformat(),
            effective_to=entry.effective_to.isoformat(),
            primary=primary,
            score=round(score, 2),
        )

    def _make_ticket(self, intent, confidence, query, context, checked, reason) -> Ticket:
        slots = REQUIRED_SLOTS.get(intent.name, [] if intent.name == "hours" else ["order_id"])
        collected = {k: v for k, v in context.items() if k in slots and v}
        missing = [s for s in slots if s not in collected]
        q = mask_pii(query)[:120]
        known = "、".join(f"{k}={v}" for k, v in collected.items()) or "无"
        summary = (
            f"客户咨询：{q}。已核对知识：{'、'.join(checked) or '无命中'}。"
            f"需人工处理：{reason or '确认答复口径'}。已知信息：{known}。"
        )
        return Ticket(
            ticket_id="T" + uuid.uuid4().hex[:10].upper(),
            created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            queue=QUEUE_MAP.get(intent.name, "人工客服"),
            reason=reason or "机器人置信度不足",
            intent=intent.name,
            confidence=confidence,
            customer_query=q,
            collected=collected,
            missing_slots=missing,
            checked_kb=checked,
            handoff_summary=summary,
            do_not_reask=list(collected.keys())
            + (["问题描述"] if not collected else []),
        )

    def _base_reply(self, entry: KBEntry, body: str) -> str:
        return (
            f"关于「{entry.question}」：{body}\n\n"
            f"依据：{entry.id}（{entry.owner}，{entry.effective_from.isoformat()} 至 "
            f"{entry.effective_to.isoformat()} 有效）"
        )

    def _polish(self, entry: KBEntry, query: str, action: str) -> tuple[str, list[str]]:
        """模型只改措辞；任何校验失败都回退到知识原文。"""
        body = entry.answer
        if not (self.use_llm and self.llm):
            return body, []
        outcome = polish_body(
            self.llm,
            query=mask_pii(query),
            question=entry.question,
            source_answer=entry.answer,
            action=action,
        )
        if outcome.text:
            return outcome.text, [outcome.note]
        return body, [f"llm 未采用：{outcome.note}"]

    def respond(self, query: str, context: dict | None = None) -> Answer:
        context = context or {}
        q = query.strip()
        intent = detect_intent(q)

        if len(q) <= 6 and any(g in q.lower() for g in GREETINGS):
            return Answer(
                query=q,
                reply=(
                    "您好，我在。请直接描述问题（订单号、商品和遇到的情况），"
                    "我会先按规则给出可核对的答复；涉及退款金额或账号安全会转人工确认。"
                ),
                action="clarify",
                confidence=0.0,
                intent=intent.name,
                notes=["问候语，不消耗知识检索"],
            )

        if any(term in q for term in OUT_OF_SCOPE_TERMS):
            ticket = self._make_ticket(intent, 0.0, q, context, [], "超出服务范围")
            return Answer(
                q,
                "这个问题超出了当前服务范围，知识库里也没有对应规则，已转人工确认，不会凭空承诺。",
                "refuse_and_escalate",
                0.0,
                intent.name,
                [],
                ticket,
                ["范围外问题：不在知识库覆盖内"],
            )

        if not self.entries or self.bm25 is None:
            ticket = self._make_ticket(intent, 0.0, q, context, [], "当前没有生效知识")
            return Answer(q, "当前知识库没有生效条目，已转人工。", "refuse_and_escalate", 0.0, intent.name, [], ticket)

        raw_scores = self.bm25.search(q)
        scores = list(raw_scores)
        prefix = INTENT_KB_PREFIX.get(intent.name)
        if prefix:
            scores = [
                (i, s + (2.0 if self.entries[i].id.startswith(prefix) else 0.0))
                for i, s in scores
            ]
            scores.sort(key=lambda x: x[1], reverse=True)
        idx, _ranked_score = scores[0]
        # 意图只负责“把正确条目排上来”，置信度仍用原始检索分，避免把零证据抬成可答。
        top_score = dict(raw_scores)[idx]
        entry = self.entries[idx]
        conf = confidence_from_score(top_score)
        action = decide(top_score, intent)

        related = [
            self._cite(self.entries[i], s, primary=False)
            for i, s in scores[1:4]
            if s >= max(2.0, top_score * 0.45)
        ]
        citations = [self._cite(entry, top_score, primary=True)] + related
        checked = [c.id for c in citations]

        if action == "refuse_and_escalate":
            reason = "知识库没有足够可靠的答案"
            ticket = self._make_ticket(intent, conf, q, context, checked, reason)
            reply = (
                "这个问题我在知识库里没有找到足够可靠的答案，为避免给错信息，先转人工。\n\n"
                f"已创建工单 {ticket.ticket_id}（{ticket.queue}），"
                "请补充订单号或具体页面截图；人工不会重复询问你已提供的信息。"
            )
            return Answer(q, reply, action, conf, intent.name, [], ticket, ["低置信度：未输出未经核实的结论"])

        if action == "answer_with_human":
            reason = intent.reason or "置信度不足以直接下结论"
            ticket = self._make_ticket(intent, conf, q, context, checked, reason)
            body, llm_notes = self._polish(entry, q, action)
            reply = self._base_reply(entry, body)
            reply += (
                f"\n\n说明：{reason}，以上为规则解释，最终结果需人工确认。\n"
                f"已创建工单 {ticket.ticket_id}（{ticket.queue}），"
                "人工会沿用上面的信息，不需要你重复描述。"
            )
            return Answer(q, reply, action, conf, intent.name, citations, ticket, llm_notes)

        body, llm_notes = self._polish(entry, q, action)
        reply = self._base_reply(entry, body)
        if related:
            reply += "\n相关规则：" + "、".join(f"{c.id}" for c in related) + "（可一并核对）"
        notes = list(llm_notes)
        promises = scan_promises(reply)
        if promises:
            notes.append(f"回复含承诺性表述 {promises}，建议法务复核")
        bad_numbers = untraceable_numbers(reply, [c.answer + c.question for c in citations])
        if bad_numbers:
            notes.append(f"存在未能溯源的数字 {bad_numbers}，已降级为人工确认")
            ticket = self._make_ticket(intent, conf, q, context, checked, "回复含未溯源数字")
            reply += f"\n\n已创建工单 {ticket.ticket_id}，请人工核对上述数字后再答复。"
            return Answer(q, reply, "answer_with_human", conf, intent.name, citations, ticket, notes)
        return Answer(q, reply, action, conf, intent.name, citations, None, notes)


def run_once(
    kb_path: str | Path,
    query: str,
    context: dict | None = None,
    on: str | None = None,
    use_llm: bool | None = None,
    llm: LLMConfig | None = None,
) -> Answer:
    kb = KnowledgeBase.load(kb_path)
    day = date.fromisoformat(on) if on else date.today()
    agent = SupportAgent(kb, today=day, llm=llm, use_llm=use_llm)
    return agent.respond(query, context)


if __name__ == "__main__":
    print(json.dumps(run_once("data/kb", "退款多久到账").to_dict(), ensure_ascii=False, indent=2))
