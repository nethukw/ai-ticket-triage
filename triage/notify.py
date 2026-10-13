"""Send an alert about urgent tickets to a Slack/Discord-style webhook."""
import json
import urllib.error
import urllib.request


def build_message(results) -> str:
    urgent = [r for r in results if r.priority == "urgent"]
    if not urgent:
        return ""
    lines = [f"URGENT: {len(urgent)} ticket(s) need attention:"]
    lines += [f"- #{r.ticket_id} {r.customer}: {r.summary}" for r in urgent]
    return "\n".join(lines)


def send_webhook(url: str, message: str, timeout: int = 10) -> bool:
    """POST the message. Returns True on success, False on any failure."""
    if not message:
        return False
    payload = json.dumps({"text": message, "content": message}).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False
