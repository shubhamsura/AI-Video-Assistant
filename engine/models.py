from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class ActionItem:
    task: str
    owner: str = "Unassigned"
    deadline: str = "Not specified"
    priority: str = "Medium"


@dataclass
class KeyDecision:
    decision: str
    category: str = "General"


@dataclass
class OpenQuestion:
    question: str
    target_person: str = "Team"


@dataclass
class MeetingAnalysisResult:
    """Unified container for meeting processing outputs."""

    title: str
    transcript: str
    summary: str
    action_items: str
    key_decisions: str
    open_questions: str
    rag_chain: Optional[object] = None
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    word_count: int = 0
    chunk_count: int = 0

    def __post_init__(self):
        if self.transcript:
            self.word_count = len(self.transcript.split())
