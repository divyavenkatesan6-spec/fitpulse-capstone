"""Part D (2/2) - Membership freeze agent: an AI "agent" that uses TOOLS in a safe loop.

Run from the project root:   python run_agent_demo.py

An agent repeats:  THINK (choose a tool) -> ACT (run it) -> OBSERVE (read result) ... until done.
Safety rules (from conduct_policy.txt): maximum 5 steps, then hand over to a human with a summary.
Extra limits: token budget and time limit. Progress is saved after every step (checkpoint).
"""
import json
import re
import sqlite3
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

from src.policy_rag import PolicyRAG

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "fitpulse.db"
OUT = ROOT / "outputs"
STATE = ROOT / "state"
LEDGER = OUT / "freeze_ledger.jsonl"
MAX_STEPS = 5                      # conduct_policy.txt: hand over to a human after 5 steps
rag = PolicyRAG()


# ====================================================================== TOOLS
def get_member(member_id):
    """Look the member up in the database and count freezes already used."""
    with sqlite3.connect(DB) as con:
        row = con.execute("select member_id, plan, city from members where member_id=?", (member_id,)).fetchone()
    if not row:
        return {"error": "member not found"}
    used = 0
    if LEDGER.exists():
        used = sum(1 for line in LEDGER.read_text().splitlines() if json.loads(line)["member_id"] == member_id)
    return {"member_id": row[0], "plan": row[1], "city": row[2], "freezes_used": used}


def search_policy(query):
    """Read the policy documents (uses the RAG search from the previous file)."""
    return {"results": [{"source": h["source"], "text": h["text"]} for h in rag.retrieve(query, 2)]}


def check_freeze(plan, freezes_used, days, today):
    """Apply the membership policy. Billing happens on the 1st of every month."""
    today = date.fromisoformat(today)
    next_billing = date(today.year + (today.month == 12), today.month % 12 + 1, 1)
    days_before_billing = (next_billing - today).days
    if days < 15:
        return {"eligible": False, "reason": "freeze must be at least 15 days", "fee_inr": 0}
    if days > 60:
        return {"eligible": False, "reason": "freeze cannot exceed 60 days in a year", "fee_inr": 0}
    if days_before_billing < 3:
        return {"eligible": False, "reason": "request must be made at least 3 days before billing date", "fee_inr": 0}
    free = plan == "Premium" and freezes_used == 0          # Premium: one free freeze per year
    return {"eligible": True, "reason": "meets policy", "fee_inr": 0 if free else 200}


def approve_freeze(member_id, days, fee_inr):
    """Write the decision into a ledger file (a teaching stand-in for a real billing system)."""
    OUT.mkdir(exist_ok=True)
    entry = {"time": datetime.now(timezone.utc).isoformat(), "member_id": member_id, "days": days, "fee_inr": fee_inr}
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"status": "approved", "days": days, "fee_inr": fee_inr}


TOOLS = {"get_member": get_member, "search_policy": search_policy,
         "check_freeze": check_freeze, "approve_freeze": approve_freeze}


# ====================================================================== THE "BRAIN"
def agent_brain(messages):
    """Stand-in for an AI model: looks at the conversation and decides the next move.
    Returns ("tool", name, arguments) or ("final", text)."""
    request = next(m["content"] for m in messages if m["role"] == "user")
    results = {m["name"]: json.loads(m["content"]) for m in messages if m["role"] == "tool"}
    member = re.search(r"\bM\d{4}\b", request)
    days = re.search(r"(\d+)\s*days", request)
    today = re.search(r"\d{4}-\d{2}-\d{2}", request)

    if "[LOOP]" in request:                                    # demo of a confused agent
        return ("tool", "search_policy", {"query": "freeze policy"})
    if not (member and days and today):
        return ("final", "NEEDS_INFO: please give member id, number of days and today's date.")
    if "get_member" not in results:
        return ("tool", "get_member", {"member_id": member.group(0)})
    info = results["get_member"]
    if "error" in info:
        return ("final", "NOT_FOUND: no such member.")
    if "search_policy" not in results:
        return ("tool", "search_policy", {"query": "freeze membership rules fee days"})
    if "check_freeze" not in results:
        return ("tool", "check_freeze", {"plan": info["plan"], "freezes_used": info["freezes_used"],
                                         "days": int(days.group(1)), "today": today.group(0)})
    decision = results["check_freeze"]
    if not decision["eligible"]:
        return ("final", f"REJECTED: {decision['reason']}.")
    if "approve_freeze" not in results:
        return ("tool", "approve_freeze", {"member_id": info["member_id"], "days": int(days.group(1)),
                                           "fee_inr": decision["fee_inr"]})
    return ("final", f"APPROVED: freeze for {results['approve_freeze']['days']} days, "
                     f"fee INR {results['approve_freeze']['fee_inr']}.")


# ====================================================================== THE LOOP
def run_agent(request, scenario="demo", token_limit=3000, time_limit_s=10):
    run_id = str(uuid.uuid4())[:8]
    STATE.mkdir(exist_ok=True)
    messages = [{"role": "system", "content": "AGENT: handle membership freeze requests using the tools."},
                {"role": "user", "content": request}]
    steps, tokens, start = [], 0, time.monotonic()

    def finish(status, text):
        return {"scenario": scenario, "status": status, "answer": text, "tools_used": [s["tool"] for s in steps],
                "steps": len(steps), "tokens": tokens}

    for _ in range(MAX_STEPS + 1):
        if time.monotonic() - start > time_limit_s:
            return finish("HANDOVER_TIMEOUT", "Time limit reached. Handing over to a human agent.")
        decision = agent_brain(messages)
        tokens += sum(len(str(m["content"]).split()) for m in messages)        # rough token count
        if tokens > token_limit:
            return finish("HANDOVER_TOKEN_LIMIT", "Token limit reached. Handing over to a human agent.")
        if decision[0] == "final":
            return finish(decision[1].split(":")[0], decision[1])
        if len(steps) >= MAX_STEPS:                                             # step cap reached
            summary = f"Handing over: {len(steps)} steps used without resolving. Last tool: {steps[-1]['tool']}."
            return finish("HANDOVER_MAX_STEPS", summary)
        _, name, args = decision
        observation = TOOLS[name](**args)
        steps.append({"tool": name, "args": args})
        messages.append({"role": "assistant", "content": f"call {name}"})
        messages.append({"role": "tool", "name": name, "content": json.dumps(observation, default=str)})
        (STATE / f"{scenario}_{run_id}.json").write_text(                       # checkpoint after every step
            json.dumps({"run_id": run_id, "request": request, "steps": steps, "tokens": tokens}), encoding="utf-8")
    return finish("HANDOVER_MAX_STEPS", "Handing over to a human agent.")
