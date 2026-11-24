"""轻量 BM25 检索（纯标准库），面向中文客服问句。

中文用“单字 + 二元组”切分，兼顾召回与短语匹配；再叠加一层电商同义词扩展。
真实生产可替换为向量检索，但“引用可溯源”的约束不变。
"""

from __future__ import annotations

import math
import re
from collections import Counter

LATIN_RE = re.compile(r"[a-z0-9]+")
CJK_RE = re.compile(r"[\u4e00-\u9fa5]+")

SYNONYMS = {
    "价保": ["价格保护", "降价", "补差", "退差价"],
    "价格保护": ["价保", "补差", "退差价"],
    "补差": ["价保", "补差价", "退差价", "差价"],
    "补差价": ["价保", "补差", "退差价"],
    "退差价": ["价保", "补差", "差价"],
    "差价": ["价保", "补差", "退差价"],
    "现金": ["折现", "优惠券"],
    "退货": ["退款", "无理由", "七天无理由"],
    "退款": ["退货", "退回", "原路退回", "退钱"],
    "退钱": ["退款", "退货", "退回"],
    "退": ["退款", "退货", "退回"],
    "签收": ["签收日", "自签收", "收货"],
    "签收当天": ["自签收", "签收日", "7天"],
    "当天": ["签收日", "自签收", "起算"],
    "发货": ["出库", "寄出", "物流"],
    "出库": ["发货", "付款", "48小时"],
    "物流": ["发货", "配送", "到货", "快递"],
    "快递": ["物流", "配送", "到货"],
    "送达": ["到货", "配送", "签收"],
    "配送": ["送达", "到货", "地区"],
    "收到": ["到货", "送达", "签收"],
    "到货": ["送达", "收货", "签收"],
    "普通地区": ["地区", "送达", "5天"],
    "偏远": ["偏远地区", "配送", "5–7天"],
    "保修": ["质保", "维修", "换新"],
    "维修": ["保修", "免费维修", "售后"],
    "摔损": ["人为", "保修", "维修"],
    "摔坏": ["摔损", "人为", "保修"],
    "进液": ["进水", "人为", "保修"],
    "免费修": ["免费维修", "保修"],
    "发票": ["开票", "电子发票", "抬头"],
    "开票信息": ["开票", "抬头", "发票"],
    "邮箱": ["下单邮箱", "开票", "推送"],
    "账号": ["账户", "登录", "封号"],
    "封号": ["冻结", "解封", "申诉", "账号"],
    "解冻": ["解封", "冻结", "申诉"],
    "身份信息": ["身份核验", "审核", "解冻"],
    "身份核验": ["身份信息", "审核", "解冻"],
    "手机号": ["账号", "身份信息", "登录"],
    "运费": ["邮费", "包邮"],
    "优惠券": ["券", "满减", "折扣"],
    "满减": ["优惠券", "折扣", "叠加"],
    "折扣": ["优惠券", "满减", "叠加"],
    "叠加": ["优惠券", "满减", "一起用"],
    "几张": ["最多", "张", "券"],
    "一单": ["同一订单", "订单", "券"],
    "人工": ["真人", "客服", "转人工"],
    "几天": ["多久", "时效", "时间"],
    "什么时候": ["多久", "时间", "时效"],
    "上班": ["工作时间", "服务时间", "几点", "在线"],
    "营业": ["服务时间", "工作时间", "几点"],
    "工作时间": ["服务时间", "几点", "上班"],
    "几点": ["时间", "服务时间"],
    "在线": ["人工", "服务时间"],
}


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens: list[str] = LATIN_RE.findall(text)
    for run in CJK_RE.findall(text):
        if len(run) == 1:
            tokens.append(run)
            continue
        # 中文以二元组为主，避免单字带来的伪匹配；整词用于精确短语加分
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
        if len(run) <= 8:
            tokens.append(run)
    return tokens


def expand(query: str) -> str:
    extra: list[str] = []
    for key, syns in SYNONYMS.items():
        if key in query:
            extra.extend(syns)
    return query + (" " + " ".join(extra) if extra else "")


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [tokenize(d) for d in docs]
        self.k1 = k1
        self.b = b
        self.n = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(self.n, 1)
        self.tf = [Counter(d) for d in self.docs]
        df: Counter = Counter()
        for d in self.docs:
            for term in set(d):
                df[term] += 1
        self.idf = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def score(self, query: str, index: int) -> float:
        q = tokenize(expand(query))
        doc_len = len(self.docs[index])
        s = 0.0
        for term in q:
            if term not in self.tf[index]:
                continue
            freq = self.tf[index][term]
            s += self.idf.get(term, 0.0) * freq * (self.k1 + 1) / (
                freq + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1))
            )
        return s

    def search(self, query: str) -> list[tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(self.n)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
