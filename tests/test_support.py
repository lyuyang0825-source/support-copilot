import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from support_copilot.agent import SupportAgent, run_once  # noqa: E402
from support_copilot.kb import KnowledgeBase, parse_entry  # noqa: E402
from support_copilot.policy import detect_intent, mask_pii, untraceable_numbers  # noqa: E402
from support_copilot.qa import score_dialog  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "data" / "kb"


class TestKB(unittest.TestCase):
    def test_load_entries(self):
        kb = KnowledgeBase.load(KB_DIR)
        self.assertEqual(len(kb.entries), 8)
        entry = kb.get("kb-refund-001")
        self.assertIn("7 天", entry.answer)

    def test_lint_clean(self):
        self.assertEqual(KnowledgeBase.load(KB_DIR).lint(on=date(2026, 11, 24)), [])

    def test_conflict_detection(self):
        def entry(eid: str, answer: str) -> str:
            return (
                "---\n"
                f"id: {eid}\n"
                "question: 保修多久\n"
                f"answer: {answer}\n"
                "scope: 电子产品\n"
                "effective_from: 2025-01-01\n"
                "effective_to: 2026-12-31\n"
                "owner: 售后组\n"
                "---\n"
            )

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.md").write_text(entry("kb-a", "保修 12 个月"), encoding="utf-8")
            (Path(tmp) / "b.md").write_text(entry("kb-b", "保修 24 个月"), encoding="utf-8")
            kb = KnowledgeBase.load(tmp)
            self.assertEqual(len(kb.conflicts()), 1)

    def test_expired_knowledge_not_used(self):
        kb = KnowledgeBase.load(KB_DIR)
        agent = SupportAgent(kb, today=date(2027, 6, 1))
        ans = agent.respond("退款多久到账")
        self.assertEqual(ans.action, "refuse_and_escalate")


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBase.load(KB_DIR)
        self.agent = SupportAgent(self.kb, today=date(2026, 11, 24))

    def test_refund_info_answer_with_citation(self):
        ans = self.agent.respond("退款多久到账？")
        self.assertEqual(ans.action, "answer")
        self.assertEqual(ans.citations[0].id, "kb-refund-001")
        self.assertIn("依据：kb-refund-001", ans.reply)

    def test_transactional_refund_requires_human(self):
        ans = self.agent.respond(
            "我要退款，订单号 A20261124001", {"order_id": "A20261124001"}
        )
        self.assertEqual(ans.action, "answer_with_human")
        self.assertIsNotNone(ans.ticket)
        self.assertEqual(ans.ticket.collected["order_id"], "A20261124001")
        self.assertEqual(ans.ticket.missing_slots, [])
        self.assertAlmostEqual(ans.confidence, 0.8, delta=0.2)

    def test_price_answer(self):
        ans = self.agent.respond("价保几天？")
        self.assertEqual(ans.citations[0].id, "kb-price-001")

    def test_shipping_answer(self):
        ans = self.agent.respond("一般几天能到货")
        self.assertEqual(ans.citations[0].id, "kb-ship-001")

    def test_invoice_answer(self):
        ans = self.agent.respond("发票抬头怎么改")
        self.assertEqual(ans.citations[0].id, "kb-invoice-001")

    def test_unknown_refused_without_citation(self):
        ans = self.agent.respond("你们支持火星配送吗")
        self.assertEqual(ans.action, "refuse_and_escalate")
        self.assertEqual(ans.citations, [])
        self.assertIsNotNone(ans.ticket)

    def test_greeting_clarify(self):
        ans = self.agent.respond("你好")
        self.assertEqual(ans.action, "clarify")

    def test_pii_masked_in_ticket(self):
        ans = self.agent.respond("我要退款，手机号 13800138000", {})
        self.assertIn("[手机号已脱敏]", ans.ticket.customer_query)
        self.assertNotIn("13800138000", ans.ticket.customer_query)

    def test_account_requires_human(self):
        ans = self.agent.respond("账号被冻结了怎么解封")
        self.assertEqual(ans.action, "answer_with_human")
        self.assertEqual(ans.ticket.queue, "账户安全组")

    def test_service_hours_synonyms(self):
        for q in ("你们几点上班", "客服工作时间", "营业到几点"):
            ans = self.agent.respond(q)
            self.assertEqual(ans.action, "answer", q)
            self.assertEqual(ans.citations[0].id, "kb-hours-001", q)


class TestPolicy(unittest.TestCase):
    def test_detect_intents(self):
        self.assertEqual(detect_intent("我要退款").name, "refund")
        self.assertTrue(detect_intent("我要退款").human_required)
        self.assertFalse(detect_intent("退款规则是什么").human_required)
        self.assertEqual(detect_intent("我要起诉你们").name, "legal")

    def test_untraceable_numbers(self):
        self.assertEqual(untraceable_numbers("退款 3 个工作日到账", ["退款 5 个工作日到账"]), ["3 个工作日"])
        self.assertEqual(untraceable_numbers("退款 3–5 个工作日", ["退款 3–5 个工作日原路退回"]), [])

    def test_mask_pii(self):
        self.assertEqual(mask_pii("电话 13800138000"), "电话 [手机号已脱敏]")


class TestQA(unittest.TestCase):
    def test_good_dialog(self):
        data = json.loads((ROOT / "data" / "tickets" / "ticket_01_good.json").read_text(encoding="utf-8"))
        result = score_dialog(data["turns"])
        self.assertGreaterEqual(result.score, 90, result.issues)

    def test_bad_dialog(self):
        data = json.loads((ROOT / "data" / "tickets" / "ticket_02_bad.json").read_text(encoding="utf-8"))
        result = score_dialog(data["turns"])
        self.assertLess(result.score, 70)
        self.assertTrue(any("承诺" in i for i in result.issues))
        self.assertTrue(any("转人工" in i for i in result.issues))

    def test_safety_complaint_needs_human(self):
        data = json.loads((ROOT / "data" / "tickets" / "ticket_03_safety.json").read_text(encoding="utf-8"))
        result = score_dialog(data["turns"])
        self.assertLess(result.score, 50)
        self.assertTrue(any("转人工" in i for i in result.issues), result.issues)


if __name__ == "__main__":
    unittest.main()
