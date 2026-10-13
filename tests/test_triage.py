import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from triage.llm import LLMError, classify_mock, classify_with_openai, validate_result
from triage.models import Ticket
from triage.notify import build_message
from triage.pipeline import load_tickets, run, write_csv, write_markdown


def ticket(subject, body="", customer="Test"):
    return Ticket(id="1", customer=customer, subject=subject, body=body)


class FakeClient:
    """Mimics client.chat.completions.create() so tests never hit the network."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        message = SimpleNamespace(content=reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class ValidateTests(unittest.TestCase):
    def test_invalid_values_fall_back_to_safe_defaults(self):
        out = validate_result({"category": "banana", "priority": "???", "sentiment": 5})
        self.assertEqual(out["category"], "general")
        self.assertEqual(out["priority"], "medium")
        self.assertEqual(out["sentiment"], "neutral")

    def test_values_are_normalised(self):
        out = validate_result({"category": " Billing ", "priority": "URGENT"})
        self.assertEqual(out["category"], "billing")
        self.assertEqual(out["priority"], "urgent")
        self.assertTrue(out["needs_human"])

    def test_non_dict_raises(self):
        with self.assertRaises(LLMError):
            validate_result(["not", "a", "dict"])


class MockClassifierTests(unittest.TestCase):
    def test_billing_urgent(self):
        r = classify_mock(ticket("Charged twice", "refund me"))
        self.assertEqual(r["category"], "billing")
        self.assertEqual(r["priority"], "urgent")

    def test_feature_request_is_low_priority(self):
        r = classify_mock(ticket("Idea", "It would be great to add dark mode"))
        self.assertEqual(r["category"], "feature_request")
        self.assertEqual(r["priority"], "low")

    def test_unknown_is_general(self):
        self.assertEqual(classify_mock(ticket("Hello", "Opening hours?"))["category"], "general")


class OpenAITests(unittest.TestCase):
    GOOD = json.dumps({"category": "technical", "priority": "high", "sentiment": "negative",
                       "summary": "App crashes", "suggested_reply": "Sorry!", "needs_human": True})

    def test_success(self):
        client = FakeClient([self.GOOD])
        r = classify_with_openai(ticket("Crash"), client, "m", sleep=lambda s: None)
        self.assertEqual(r["category"], "technical")
        self.assertEqual(client.calls, 1)

    def test_retries_then_succeeds(self):
        client = FakeClient([RuntimeError("rate limit"), "not json", self.GOOD])
        r = classify_with_openai(ticket("Crash"), client, "m", sleep=lambda s: None)
        self.assertEqual(r["priority"], "high")
        self.assertEqual(client.calls, 3)

    def test_gives_up_after_max_retries(self):
        client = FakeClient([RuntimeError("down")] * 3)
        with self.assertRaises(LLMError):
            classify_with_openai(ticket("Crash"), client, "m", sleep=lambda s: None)


class PipelineTests(unittest.TestCase):
    def test_one_failure_does_not_stop_batch(self):
        def flaky(t):
            if t.subject == "bad":
                raise RuntimeError("boom")
            return classify_mock(t)

        tickets = [ticket("bad"), ticket("Hello", "Opening hours?")]
        results = run(tickets, flaky, "mock")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].source, "fallback")
        self.assertTrue(results[0].needs_human)
        self.assertEqual(results[1].source, "mock")

    def test_load_tickets_reports_missing_field(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.json"
            p.write_text(json.dumps([{"customer": "A", "subject": "B"}]))
            with self.assertRaises(ValueError):
                load_tickets(p)

    def test_reports_are_written(self):
        results = run([ticket("Charged twice", "refund")], classify_mock, "mock")
        with tempfile.TemporaryDirectory() as d:
            write_csv(results, Path(d) / "r.csv")
            write_markdown(results, Path(d) / "r.md")
            self.assertIn("ticket_id", (Path(d) / "r.csv").read_text())
            self.assertIn("urgent", (Path(d) / "r.md").read_text())

    def test_notify_message_only_for_urgent(self):
        low = run([ticket("Hello", "Opening hours?")], classify_mock, "mock")
        self.assertEqual(build_message(low), "")
        urgent = run([ticket("Charged twice")], classify_mock, "mock")
        self.assertIn("1 ticket", build_message(urgent))


if __name__ == "__main__":
    unittest.main()
