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
    "价保": ["价格保护", "降价", "补差"],
    "退货": ["退款", "无理由", "七天无理由"],
    "退款": ["退货", "退回", "原路退回"],
    "发货": ["出库", "寄出", "物流"],
    "保修": ["质保", "维修", "换新"],
    "发票": ["开票", "电子发票", "抬头"],
    "账号": ["账户", "登录", "封号"],
    "运费": ["邮费", "包邮"],
    "优惠券": ["券", "满减", "折扣"],
    "人工": ["真人", "客服", "转人工"],
    "几天": ["多久", "时效", "时间"],
    "到货": ["送达", "收货", "签收"],
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
