"""回答策略：置信度闸门、强制转人工、承诺词与数字溯源。

核心原则：
- 知识库没有可靠答案时，宁可说“不知道”并转人工，也不猜；
- 涉及退款金额、赔偿、价格、合同、账号安全等“会改变用户权益”的结论，
  机器人只解释规则、收集信息、建工单，最终确认交给人工（对齐客服新国标的协同要求）；
- 回复里的每个数字，都必须能在引用的知识原文里找到。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PII_PATTERNS = [
    (re.compile(r"\b1[3-9]\d{9}\b"), "[手机号已脱敏]"),
    (re.compile(r"\b\d{17}[\dXx]\b"), "[身份证已脱敏]"),
    (re.compile(r"\b\d{16,19}\b"), "[银行卡已脱敏]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.\w+\b"), "[邮箱已脱敏]"),
]

PROMISE_WORDS = ("保证", "一定", "肯定", "承诺", "立即退款", "马上到账", "马上退款", "包退", "100%", "绝对")

INTENT_RULES = [
    ("legal", ("起诉", "律师", "法院", "12315", "消协", "工商", "报警")),
    ("complaint", ("投诉", "差评", "举报", "曝光")),
    ("hours", ("客服时间", "上班时间", "几点", "人工客服", "留言", "回电", "晚上", "下班", "点后")),
    ("account", ("账号", "账户", "封号", "解封", "冻结", "解冻", "登录", "身份核验", "身份信息", "手机号", "申诉")),
    ("compensation", ("赔偿", "赔付", "补偿")),
    ("price", ("价保", "补差", "补差价", "退差价", "差价", "价格保护", "降价")),
    ("coupon", ("优惠券", "满减", "叠加", "折扣", "券")),
    ("refund", ("退款", "退货", "退钱", "退给", "无理由", "7天", "退")),
    ("contract", ("合同", "协议", "条款", "签约")),
    ("invoice", ("发票", "开票", "抬头", "开票信息", "邮箱")),
    ("shipping", ("发货", "物流", "到货", "快递", "出库", "送达", "配送", "收到")),
    ("warranty", ("保修", "维修", "质保", "换新", "摔损", "摔坏", "进液", "进水")),
    ("hours", ("客服时间", "上班时间", "几点", "人工客服")),
]

HUMAN_INTENTS = {"legal", "complaint", "account", "compensation", "contract", "privacy"}
TRANSACTIONAL_MARKERS = (
    "我要", "帮我", "给我", "申请退款", "申请退货", "申请价保", "申请补差",
    "现在", "马上", "立即", "立刻", "今天", "必须", "退给", "退钱", "给我退",
)

QUEUE_MAP = {
    "refund": "售后组",
    "compensation": "客诉组",
    "price": "售后组",
    "account": "账户安全组",
    "complaint": "客诉组",
    "legal": "法务对接",
    "contract": "合同审核",
    "invoice": "财务开票组",
    "shipping": "物流组",
    "warranty": "售后组",
    "coupon": "运营支持",
    "hours": "人工客服",
    "other": "人工客服",
}

THRESHOLDS = {"answer": 0.70, "caveat": 0.52}
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:个工作日|工作日|天|日|小时|分钟|元|%|个月|年|次|倍|张)")


def mask_pii(text: str) -> str:
    for pattern, repl in PII_PATTERNS:
        text = pattern.sub(repl, text)
    return text


@dataclass
class Intent:
    name: str
    human_required: bool
    reason: str
    transactional: bool


def detect_intent(query: str) -> Intent:
    matched = "other"
    for name, keys in INTENT_RULES:
        if any(k in query for k in keys):
            matched = name
            break
    transactional = any(k in query for k in TRANSACTIONAL_MARKERS)

    human = matched in HUMAN_INTENTS
    reason = ""
    if matched == "refund" and transactional:
        human = True
        reason = "退款金额与到账方式需要人工确认"
    elif matched == "price" and transactional:
        human = True
        reason = "价保补差需要人工核价"
    elif matched in HUMAN_INTENTS:
        reason = {
            "legal": "涉及法律与监管口径",
            "complaint": "涉及客诉与平台介入",
            "account": "涉及账号安全与身份核验",
            "compensation": "涉及赔付金额",
            "contract": "涉及合同条款变更",
            "privacy": "涉及个人信息处理",
        }.get(matched, "需要人工确认")
    return Intent(name=matched, human_required=human, reason=reason, transactional=transactional)


def scan_promises(text: str) -> list[str]:
    return [w for w in PROMISE_WORDS if w in text]


def untraceable_numbers(text: str, sources: list[str]) -> list[str]:
    """回复中出现、但引用原文里没有的数字。"""
    src = " ".join(sources)
    src_norm = src.replace(" ", "").replace("–", "-").replace("—", "-")
    out = []
    for m in NUMBER_RE.finditer(text):
        token = m.group(0).replace(" ", "")
        if token not in src_norm and token.replace("-", "–") not in src:
            out.append(m.group(0))
    return out


def confidence_from_score(score: float) -> float:
    return round(min(0.95, score / (score + 3.0)), 2)


def decide(score: float, intent: Intent) -> str:
    """返回 answer / answer_with_human / refuse_and_escalate。"""
    conf = confidence_from_score(score)
    if score < 2.0 or conf < THRESHOLDS["caveat"]:
        return "refuse_and_escalate"
    if intent.name == "other" and score < 8.0:
        return "refuse_and_escalate"
    if intent.human_required:
        return "answer_with_human"
    if conf < THRESHOLDS["answer"]:
        return "answer_with_human"
    return "answer"
