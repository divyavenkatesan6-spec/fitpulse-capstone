"""Part C - Support ticket triage: guardrails -> redact personal data -> classify -> validate -> route.

Run from the project root:   python src/triage.py

No real AI service is used. `fake_llm` below is a small stand-in that behaves like a real
model would: usually right, sometimes busy (error), sometimes badly formatted. This lets us
practise the safety code (retries, validation) that real AI systems need.
"""
import json
import random
import re
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "outputs"

ALLOWED = ["billing", "freeze", "equipment", "harassment"]
MAX_WORDS = 150            # longer tickets are not sent to the model
TOKEN_BUDGET = 20000       # stop calling the model after this many (estimated) tokens

# ---------------------------------------------------------------- fake AI model
class TransientLLMError(Exception):
    """A temporary failure ("server busy"). Trying again usually works."""


_calls = {"n": 0}

KEYWORDS = [   # checked in this order: safety first
    ("harassment", {"harassment", "inappropriate", "unsafe", "abusive", "threatened", "following", "staring"}),
    ("freeze", {"freeze", "pause", "paused"}),
    ("billing", {"charged", "payment", "invoice", "bill", "refund", "deducted"}),
    ("equipment", {"broken", "treadmill", "machine", "machines", "shower", "water", "fix"}),
]


def estimate_tokens(text):
    return int(len(text.split()) * 1.3) + 1


def fake_llm(system_prompt, user_text):
    """Pretend AI. Returns (reply_text, tokens_used). Mistakes are scripted so we can test safety code."""
    _calls["n"] += 1
    n = _calls["n"]
    if n % 8 == 0:
        raise TransientLLMError("503 server busy")
    words = set(re.findall(r"[a-z]+", user_text.lower()))
    category = next((c for c, kw in KEYWORDS if kw & words), "equipment")
    if n % 11 == 0:                                       # chatty answer, not JSON
        reply = f"This looks like a {category} problem."
    elif n % 15 == 0:                                     # JSON but a category we do not allow
        reply = json.dumps({"category": "maintenance", "reason": "unsure"})
    else:
        reply = json.dumps({"category": category, "reason": "keyword match"})
    return reply, estimate_tokens(system_prompt) + estimate_tokens(user_text) + estimate_tokens(reply)


# ---------------------------------------------------------------- step 1: privacy
PATTERNS = [   # order matters: email first, because an email contains digits and letters
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("member_id", re.compile(r"\bM\d{4}\b"), "[MEMBER_ID]"),
]


def redact(text):
    """Replace personal details with tags and count what was removed."""
    counts = {}
    for name, pattern, tag in PATTERNS:
        text, n = pattern.subn(tag, text)
        counts[name] = n
    return text, counts


# ---------------------------------------------------------------- step 2: retry
def call_with_retry(system_prompt, user_text, ticket_id, max_retries=3, base_delay=0.05):
    """Try the model again (waiting a little longer each time) if it says 'busy'."""
    for attempt in range(max_retries + 1):
        try:
            return fake_llm(system_prompt, user_text)
        except TransientLLMError:
            if attempt == max_retries:
                raise
            time.sleep(base_delay * 2 ** attempt)          # 0.05s, 0.1s, 0.2s ...


# ---------------------------------------------------------------- step 3: validate the answer
def parse_category(reply):
    """Return (category, how). Never trust the model blindly."""
    try:
        category = json.loads(reply).get("category")
        if category in ALLOWED:
            return category, "json"
        return None, "invalid_category"
    except (json.JSONDecodeError, AttributeError):
        found = {w for w in re.findall(r"[a-z]+", reply.lower()) if w in ALLOWED}
        if len(found) == 1:                                # exactly one allowed word -> accept
            return found.pop(), "keyword_fallback"
        return None, "unreadable"


# ---------------------------------------------------------------- step 4: routing
ROUTES = {"harassment": ("regional_manager", "HIGH"), "freeze": ("membership_desk", "NORMAL"),
          "billing": ("billing_team", "NORMAL"), "equipment": ("facilities_team", "NORMAL")}

SYSTEM_PROMPT = ('CLASSIFY the gym support ticket. Allowed categories: billing, freeze, equipment, harassment. '
                 'Reply with JSON only: {"category": "...", "reason": "..."}')


def run_pipeline():
    tickets = pd.read_csv(RAW / "tickets.csv", keep_default_na=False)
    members = pd.read_csv(CLEAN / "members.csv")
    plan_of = members.set_index("member_id").plan.to_dict()
    random.seed(42)
    _calls["n"] = 0
    rows, tokens_used, removed = [], 0, {name: 0 for name, _, _ in PATTERNS}

    for _, t in tickets.iterrows():
        text = str(t.text)
        category, status, tokens = "", "", 0
        if not text.strip():
            status = "skipped_empty"                       # guardrail 1
        elif len(text.split()) > MAX_WORDS:
            status = "flagged_too_long"                    # guardrail 2
        elif tokens_used >= TOKEN_BUDGET:
            status = "budget_exhausted"                    # guardrail 3
        else:
            safe_text, counts = redact(text)               # privacy BEFORE the model sees anything
            for k, v in counts.items():
                removed[k] += v
            try:
                reply, tokens = call_with_retry(SYSTEM_PROMPT, safe_text, t.ticket_id)
                tokens_used += tokens
                category, how = parse_category(reply)
                status = f"classified_{how}" if category else f"flagged_{how}"
            except TransientLLMError:
                status = "flagged_model_unavailable"

        if category:
            route, priority = ROUTES[category]
        else:
            route, priority = "human_review", "NORMAL"     # never guess: a person decides
        if plan_of.get(t.member_id) == "Premium" and priority == "NORMAL":
            priority = "HIGH"                              # Premium members get faster handling
        rows.append({"ticket_id": t.ticket_id, "member_id": t.member_id, "plan": plan_of.get(t.member_id, ""),
                     "true_category": t.true_category, "predicted": category or "", "status": status,
                     "priority": priority, "route": route, "tokens": tokens})

    result = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    result.to_csv(OUT / "triage_results.csv", index=False)
    judged = result[result.predicted != ""]
    accuracy = round(float((judged.predicted == judged.true_category).mean()), 3)
    summary = {"tickets": len(result), "tokens_used": tokens_used, "personal_data_removed": removed,
               "accuracy_on_classified": accuracy, "status_counts": result.status.value_counts().to_dict()}
    return result, summary


if __name__ == "__main__":
    result, summary = run_pipeline()
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("\nROUTES\n", result.groupby(["route", "priority"]).size().rename("tickets").reset_index().to_string(index=False))
    print("\nSAMPLE\n", result.head(8)[["ticket_id", "status", "predicted", "route", "priority"]].to_string(index=False))
    sample, _ = redact("Call 9876543210 or mail asha@example.com about member M0123.")
    print("\nREDACTION DEMO:", sample)
