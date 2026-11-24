"""知识库：解析、校验、有效期过滤、冲突检测。

每条知识是“可引用的最小事实单元”，必须带 id、适用口径和生效区间。
没有 id 的知识不允许进入回答——这是防幻觉的第一道闸。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

REQUIRED_FIELDS = ("id", "question", "answer", "scope", "effective_from", "effective_to", "owner")

FRONT_RE = re.compile(r"^---[ \t]*\n(.*?)\n---[ \t]*(?:\n(.+))?$", re.S)


def _parse_meta(text: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"元信息行缺少冒号: {line!r}")
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta


def _parse_date(value: str) -> date:
    return date.fromisoformat(value.strip())


@dataclass
class KBEntry:
    id: str
    question: str
    answer: str
    scope: str
    effective_from: date
    effective_to: date
    owner: str
    risk: str = "normal"
    tags: list[str] = field(default_factory=list)
    source_path: str = ""

    @property
    def text(self) -> str:
        return f"{self.question}\n{self.answer}\n{self.scope}"

    def is_active(self, on: date) -> bool:
        return self.effective_from <= on <= self.effective_to


@dataclass
class Conflict:
    question: str
    ids: list[str]
    reason: str


def parse_entry(path: Path | str) -> KBEntry:
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    m = FRONT_RE.match(raw.strip())
    if not m:
        raise ValueError(f"{p.name}: 缺少 --- 元信息块")
    meta = _parse_meta(m.group(1))
    missing = [k for k in REQUIRED_FIELDS if k not in meta]
    if missing:
        raise ValueError(f"{p.name}: 缺少字段 {missing}")
    body = (m.group(2) or "").strip()
    answer = body if body else meta.get("answer", "").strip()
    if not answer:
        raise ValueError(f"{p.name}: 缺少答案（正文与 answer 字段都为空）")
    tags = [t.strip() for t in meta.get("tags", "").split(",") if t.strip()]
    return KBEntry(
        id=meta["id"],
        question=meta["question"],
        answer=answer,
        scope=meta["scope"],
        effective_from=_parse_date(meta["effective_from"]),
        effective_to=_parse_date(meta["effective_to"]),
        owner=meta["owner"],
        risk=meta.get("risk", "normal"),
        tags=tags,
        source_path=str(p),
    )


def _norm_question(q: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fa5]", "", q).lower()


class KnowledgeBase:
    def __init__(self, entries: list[KBEntry]):
        self.entries = entries
        self._by_id = {e.id: e for e in entries}
        if len(self._by_id) != len(entries):
            dup = sorted({e.id for e in entries if [x.id for x in entries].count(e.id) > 1})
            raise ValueError(f"知识 id 重复: {dup}")

    @classmethod
    def load(cls, root: Path | str) -> "KnowledgeBase":
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"知识库目录不存在: {root}")
        entries = [parse_entry(p) for p in sorted(root.glob("*.md"))]
        if not entries:
            raise ValueError(f"知识库为空: {root}")
        return cls(entries)

    def get(self, entry_id: str) -> KBEntry:
        return self._by_id[entry_id]

    def active(self, on: date | None = None) -> list[KBEntry]:
        on = on or date.today()
        return [e for e in self.entries if e.is_active(on)]

    def conflicts(self) -> list[Conflict]:
        """同一问题、生效区间重叠、答案不一致 → 冲突。"""
        conflicts: list[Conflict] = []
        groups: dict[str, list[KBEntry]] = {}
        for entry in self.entries:
            groups.setdefault(_norm_question(entry.question), []).append(entry)
        for _, items in groups.items():
            if len(items) < 2:
                continue
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    a, b = items[i], items[j]
                    overlap = a.effective_from <= b.effective_to and b.effective_from <= a.effective_to
                    if overlap and a.answer.strip() != b.answer.strip():
                        conflicts.append(
                            Conflict(
                                question=a.question,
                                ids=[a.id, b.id],
                                reason="生效区间重叠但答案不一致，回答时可能口径冲突",
                            )
                        )
        return conflicts

    def lint(self, on: date | None = None) -> list[str]:
        on = on or date.today()
        issues: list[str] = []
        for e in self.entries:
            if not e.is_active(on):
                continue
            if len(e.answer) < 10:
                issues.append(f"{e.id}: 答案过短，缺少可执行细节")
            if "http" in e.answer:
                issues.append(f"{e.id}: 答案含外链，客服场景建议改为站内路径")
            boundary = ("例外", "除外", "不参与", "不在", "不含", "仅", "最多", "不超过", "无法")
            if e.risk == "high" and not any(k in e.answer for k in boundary):
                issues.append(f"{e.id}: 高风险知识建议写明例外条款")
        for c in self.conflicts():
            issues.append(f"冲突 {c.ids}: {c.reason}")
        return issues
