"""Command-line entry point: python -m triage.cli data/sample_tickets.json"""
import argparse
import os
import sys
from pathlib import Path

from .llm import LLMError, make_classifier
from .notify import build_message, send_webhook
from .pipeline import load_tickets, run, write_csv, write_markdown


def load_dotenv(path=".env") -> None:
    """Tiny .env reader (KEY=VALUE per line) so no extra dependency is needed."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AI-powered support ticket triage")
    p.add_argument("input", help="Path to a JSON file of tickets")
    p.add_argument("--output-dir", default="output", help="Where to write reports")
    p.add_argument("--mock", action="store_true",
                   help="Use the offline keyword classifier (no API key needed)")
    p.add_argument("--model", default=None,
                   help="OpenAI model (default: env OPENAI_MODEL or gpt-4o-mini)")
    p.add_argument("--notify", action="store_true",
                   help="Send urgent tickets to WEBHOOK_URL")
    return p


def main(argv=None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    model = args.model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    try:
        tickets = load_tickets(args.input)
        classifier, source = make_classifier(args.mock, model)
    except (OSError, ValueError, LLMError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Triaging {len(tickets)} tickets using '{source}' backend...")
    results = run(tickets, classifier, source)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(results, out / "triage_results.csv")
    write_markdown(results, out / "triage_report.md")
    print(f"Wrote {out / 'triage_results.csv'} and {out / 'triage_report.md'}")

    for r in results:
        flag = " [needs human]" if r.needs_human else ""
        print(f"  #{r.ticket_id:<4} {r.priority:<7} {r.category:<16} {r.summary[:60]}{flag}")

    if args.notify:
        url = os.environ.get("WEBHOOK_URL")
        message = build_message(results)
        if not url:
            print("WEBHOOK_URL not set, skipping notification.", file=sys.stderr)
        elif not message:
            print("No urgent tickets, nothing to send.")
        else:
            ok = send_webhook(url, message)
            print("Webhook alert sent." if ok else "Webhook alert failed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
