# AI Ticket Triage

A Python tool that reads customer support tickets, uses an LLM (OpenAI API) to classify them,
set a priority, summarise them and draft a reply, then writes CSV and Markdown reports and
can alert a Slack/Discord webhook about urgent tickets.

Built as a small automation prototype: **LLM API + data pipeline + third-party integration**.

## Features

- **LLM triage**: category, priority, sentiment, one-line summary, suggested reply, and a
  "needs human" flag, returned as structured JSON.
- **Robust by design**: output is validated and normalised, API calls retry with exponential
  backoff, and one failing ticket never stops the whole batch (it is flagged for a human).
- **Prompt-injection aware**: the system prompt tells the model to treat ticket text as data.
- **Reports**: `triage_results.csv` and a `triage_report.md` summary sorted by priority.
- **Webhook alerts** for urgent tickets (Slack/Discord-style incoming webhooks).
- **Offline `--mock` mode** (keyword classifier) so it runs, and is tested, without an API key.
- **Unit tests** that use a fake OpenAI client, so no network or API cost is needed.

## Project structure

```
triage/
  models.py     dataclasses and allowed values
  llm.py        OpenAI backend, output validation, offline mock classifier
  pipeline.py   load tickets, run triage, write CSV/Markdown
  notify.py     webhook alerts for urgent tickets
  cli.py        command-line interface
tests/          unittest suite
data/           sample_tickets.json
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then add your OPENAI_API_KEY
```
## Using other OpenAI-compatible providers

The tool uses the OpenAI Python SDK, so it also works with any OpenAI-compatible API (for example Groq).
Set these in `.env`:

    OPENAI_API_KEY=your-provider-key
    OPENAI_BASE_URL=https://api.groq.com/openai/v1
    OPENAI_MODEL=<a model ID available on your provider>

Model names change over time. List the ones your key can use with the provider's `/models` endpoint.

## Usage

```bash
# Try it without an API key
python -m triage.cli data/sample_tickets.json --mock

# Use the OpenAI API
python -m triage.cli data/sample_tickets.json

# Choose a model and alert a webhook about urgent tickets
python -m triage.cli data/sample_tickets.json --model gpt-4o-mini --notify
```

Reports are written to `output/`. Ticket files are JSON lists of
`{"id", "customer", "subject", "body"}`.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Design decisions

- **Validate everything the model returns.** LLM output is untrusted, so
  `validate_result` maps unexpected values to safe defaults instead of crashing.
- **Isolate failures.** The pipeline catches per-ticket errors and marks them
  `needs_human`, so a batch of 1,000 tickets is not lost to one bad response.
- **Dependency injection for testing.** The OpenAI client is passed into the classifier,
  which makes it easy to test retries and errors with a fake client.

## Ideas for future work

- Read tickets from email/IMAP, Google Sheets or a helpdesk API instead of JSON.
- Add a FastAPI endpoint so other systems can call the triage service.
- Evaluate accuracy against a labelled dataset and compare models.
- Run the pipeline from n8n or on a schedule.
