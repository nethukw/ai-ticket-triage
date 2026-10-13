from dataclasses import dataclass, asdict

CATEGORIES = ("billing", "technical", "account", "feature_request", "general")
PRIORITIES = ("low", "medium", "high", "urgent")
SENTIMENTS = ("negative", "neutral", "positive")


@dataclass
class Ticket:
    id: str
    customer: str
    subject: str
    body: str


@dataclass
class TriageResult:
    ticket_id: str
    customer: str
    subject: str
    category: str
    priority: str
    sentiment: str
    summary: str
    suggested_reply: str
    needs_human: bool
    source: str  # "openai", "mock" or "fallback"

    def to_dict(self) -> dict:
        return asdict(self)
