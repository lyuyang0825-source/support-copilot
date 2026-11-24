"""support-copilot: 可溯源、会认怂、会交接的客服副驾驶。"""

__version__ = "0.1.0"

from .agent import Answer, SupportAgent, run_once
from .kb import KBEntry, KnowledgeBase, parse_entry
from .qa import score_dialog

__all__ = ["Answer", "SupportAgent", "run_once", "KBEntry", "KnowledgeBase", "parse_entry", "score_dialog"]
