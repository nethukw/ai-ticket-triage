"""Load tickets, run the classifier, and write CSV / Markdown reports."""
import csv
import json
from collections import Counter
from pathlib import Path

from .models import Ticket, TriageResult


def load_tickets(path) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    tickets = []
    for i, item in enumerate(data, start=1):
        try:
            tickets.append(Ticket(
                id=str(item.get("id", i)),
                customer=item["customer"],
                subject=item["subject"],
                body=item["body"],
            ))
        except KeyError as missing:
            raise ValueError(f"Ticket #{i} is missing field {missing}") from None
    return tickets


def run(tickets, classifier, source: str) -> list:
    """Triage every ticket. One failure never stops the whole batch."""
    results = []
    for t in tickets:
        try:
            r = classifier(t)
            results.append(TriageResult(
                ticket_id=t.id, customer=t.customer, subject=t.subject,
                source=source, **r,
            ))
        except Exception as exc:
            results.append(TriageResult(
                ticket_id=t.id, customer=t.customer, subject=t.subject,
                category="general", priority="medium", sentiment="neutral",
                summary=f"Automatic triage failed: {exc}"[:200],
                suggested_reply="", needs_human=True, source="fallback",
            ))
    return results


def write_csv(results, path) -> None:
    fields = list(results[0].to_dict().keys()) if results else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())


def write_markdown(results, path) -> None:
    by_priority = Counter(r.priority for r in results)
    by_category = Counter(r.category for r in results)
    order = ["urgent", "high", "medium", "low"]
    ranked = sorted(results, key=lambda r: order.index(r.priority))

    lines = ["# Ticket Triage Report", "", f"Total tickets: **{len(results)}**", ""]
    lines += ["## By priority", ""]
    lines += [f"- {p}: {by_priority.get(p, 0)}" for p in order]
    lines += ["", "## By category", ""]
    lines += [f"- {c}: {n}" for c, n in sorted(by_category.items())]
    lines += ["", "## Tickets (highest priority first)", "",
              "| ID | Priority | Category | Sentiment | Summary | Needs human |",
              "|----|----------|----------|-----------|---------|-------------|"]
    for r in ranked:
        summary = r.summary.replace("|", "/")
        lines.append(
            f"| {r.ticket_id} | {r.priority} | {r.category} | {r.sentiment} "
            f"| {summary} | {'yes' if r.needs_human else 'no'} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
