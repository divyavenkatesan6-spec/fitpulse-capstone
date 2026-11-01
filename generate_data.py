"""FitPulse synthetic data generator (all records are fictional).
Run from the project root:  python generate_data.py

Writes: data/raw/*.csv and data/policies/*.txt
The raw data is intentionally messy so Part A has real cleaning work to do.
"""
import csv
import os
import random
from datetime import datetime, timedelta

rng = random.Random(7)
RAW, POL = "data/raw", "data/policies"
os.makedirs(RAW, exist_ok=True)
os.makedirs(POL, exist_ok=True)

CITIES = ["Chennai", "Mumbai", "Bengaluru", "Delhi", "Pune"]
PLANS = [("Student", 799), ("Basic", 999), ("Standard", 1499), ("Premium", 2499)]
WORKOUTS = [("Cardio", 7.5), ("Strength", 6.0), ("Yoga", 4.0), ("HIIT", 10.0), ("Zumba", 7.0)]  # kcal/min


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def messy_city(c):
    r = rng.random()
    if r < 0.18:
        return c.lower()
    if r < 0.30:
        return c.upper()
    if r < 0.40:
        return f" {c} "
    return c


def messy_gender(g):
    options = {"male": ["male", "Male", "M", "m"], "female": ["female", "Female", "F", "f"]}
    return rng.choice(options[g])


# ---------------------------------------------------------------- plans
write_csv(f"{RAW}/plans.csv", ["plan", "monthly_fee_inr"], PLANS)

# ---------------------------------------------------------------- members
# Hidden behaviour type drives everything else: committed, casual, or drifting (likely to churn).
N = 1000
behaviour = {}
members = []
bad_ages = {15: 250, 90: -3, 180: 0, 260: 140, 400: 999}   # impossible ages planted on purpose
for i in range(N):
    mid = f"M{i + 1:04d}"
    r = rng.random()
    behaviour[mid] = "committed" if r < 0.30 else "casual" if r < 0.65 else "drifting"
    join = datetime(2024, 6, 1) + timedelta(days=rng.randint(0, 330))
    age = bad_ages.get(i, rng.randint(18, 62))
    if i not in bad_ages and rng.random() < 0.05:
        age = ""                                         # missing age
    gender = "" if rng.random() < 0.03 else messy_gender(rng.choice(["male", "female"]))
    plan = rng.choices([p[0] for p in PLANS], weights=[15, 35, 35, 15])[0]
    members.append([mid, messy_city(rng.choice(CITIES)), join.strftime("%Y-%m-%d"), age, gender, plan])
write_csv(f"{RAW}/members.csv", ["member_id", "city", "join_date", "age", "gender", "plan"], members)

# ---------------------------------------------------------------- check-ins (Jan-Jun 2025)
# visits per week by behaviour; drifting members taper off after March
START, END = datetime(2025, 1, 1), datetime(2025, 6, 30)
weekly = {"committed": 4.0, "casual": 1.5, "drifting": 2.5}
checkins = []
cid = 500000
for mid, kind in behaviour.items():
    day = START
    while day <= END:
        rate = weekly[kind] / 7
        if kind == "drifting":
            months_in = (day.year - 2025) * 12 + day.month - 1          # 0..5
            rate *= max(0.0, 1 - months_in * 0.22)                       # fades out
        if rng.random() < rate:
            cid += 1
            wk, kcal_min = rng.choice(WORKOUTS)
            dur = max(15, min(120, int(rng.gauss(55, 15))))
            ts = day + timedelta(hours=rng.randint(6, 21), minutes=rng.randint(0, 59))
            calories = int(dur * kcal_min * rng.uniform(0.85, 1.15))
            if rng.random() < 0.06:
                calories = ""                                            # missing calories
            checkins.append([f"CK{cid}", mid, ts.strftime("%Y-%m-%d %H:%M:%S"), wk, dur, calories])
        day += timedelta(days=1)

# planted problems: negative durations, "N/A" calories, exact duplicate rows
for k in range(len(checkins)):
    if k % 150 == 11:
        checkins[k][4] = -abs(checkins[k][4])
    elif k % 211 == 5:
        checkins[k][5] = "N/A"
dups = [list(r) for r in rng.sample(checkins, 120)]
checkins += dups
rng.shuffle(checkins)
write_csv(f"{RAW}/checkins.csv", ["checkin_id", "member_id", "checkin_ts", "workout_type", "duration_min", "calories"], checkins)

# ---------------------------------------------------------------- payments (monthly, Jan-Jun 2025)
fee = dict(PLANS)
plan_of = {m[0]: m[5] for m in members}
payments = []
pid = 800000
for mid in behaviour:
    for month in range(1, 7):
        pid += 1
        r = rng.random()
        status = "failed" if r < 0.06 else "refunded" if r < 0.08 else "paid"
        amount = fee[plan_of[mid]]
        payments.append([f"PAY{pid}", mid, f"2025-{month:02d}", amount, status])
for k in range(len(payments)):
    if k % 400 == 3:
        payments[k][3] = "error"                          # unreadable amount
extra = [list(r) for r in rng.sample(payments, 40)]       # duplicate payment rows
payments += extra
rng.shuffle(payments)
write_csv(f"{RAW}/payments.csv", ["payment_id", "member_id", "month", "amount_inr", "status"], payments)

# ---------------------------------------------------------------- support tickets
T = {
    "billing": [
        "I was charged twice for {m} this month. Please refund the extra amount. Email me at {e}.",
        "My payment failed but money was deducted. Call me on {p} to fix the bill.",
        "The invoice shows the wrong plan fee for member {m}. Contact {e}.",
    ],
    "freeze": [
        "I am travelling for work and want to freeze my membership for 30 days. Member {m}. Call {p}.",
        "Please pause my plan for two months because of a knee injury. Write to {e}.",
        "I need to freeze my membership, I am moving abroad temporarily. Call {p}.",
    ],
    "equipment": [
        "The treadmill near the entrance is broken and the screen is dead. Please fix it. {e}",
        "Two cable machines in the strength area are out of order and the AC is not working. Call {p}.",
        "The shower in the men's locker room has no hot water. Member {m}.",
    ],
    "harassment": [
        "A trainer made inappropriate comments to me during my session. This is harassment. Call {p}.",
        "I feel unsafe because another member keeps following me and staring at me. Member {m}, email {e}.",
        "There was an abusive argument in the changing room and I was threatened. Contact {e}.",
    ],
}
first_member = {}
for r in members:
    first_member[r[0]] = r[0]
ids = sorted(first_member)
cats = list(T)
tickets = []
for n in range(1, 161):
    cat = cats[(n - 1) % 4]
    mid = rng.choice(ids)
    phone = rng.choice("6789") + "".join(rng.choice("0123456789") for _ in range(9))
    text = rng.choice(T[cat]).format(m=mid, e=f"member{n}@example.com", p=phone)
    tickets.append([f"T{n:04d}", mid, text, cat])
for n in range(161, 165):                                 # empty tickets
    tickets.append([f"T{n:04d}", rng.choice(ids), "", "none"])
filler = "The machine is still not fixed and nobody has replied to me yet. "
for n in range(165, 169):                                 # overlong tickets
    tickets.append([f"T{n:04d}", rng.choice(ids), ("The equipment is broken. " + filler * 40).strip(), "equipment"])
rng.shuffle(tickets)
write_csv(f"{RAW}/tickets.csv", ["ticket_id", "member_id", "text", "true_category"], tickets)

# ---------------------------------------------------------------- policy documents + eval questions
DOCS = {
    "membership_policy.txt": (
        "Membership Policy. Members may freeze a membership for a minimum of 15 days and a maximum of 60 days in a calendar year. "
        "A freeze request must be raised at least 3 days before the next billing date. A freeze fee of INR 200 is charged for each freeze. "
        "Premium members can freeze for free once per year. Memberships can be cancelled with 30 days written notice. "
        "Student plan members must show a valid student identity card at the front desk every 6 months."),
    "payment_policy.txt": (
        "Payment Policy. Monthly fees are charged on the first day of each month by UPI, card or auto debit. "
        "If a payment fails the system retries it after 3 days and again after 7 days. "
        "A late fee of INR 150 applies if a failed payment is not cleared within 10 days. "
        "If a member is charged twice for the same month, the extra amount is refunded within 7 working days. "
        "Refunds are never paid in cash."),
    "facility_policy.txt": (
        "Facility Policy. Gyms are open from 5 AM to 11 PM on all days. Guests are allowed only with a day pass costing INR 300. "
        "Members must wipe equipment after use and carry a separate pair of indoor shoes. "
        "Locker items are cleared every night at closing time. Children below 16 years are not allowed in the strength area."),
    "safety_policy.txt": (
        "Safety Policy. Every new member must complete a health questionnaire before the first workout. "
        "Members with a medical condition must submit a doctor's note before joining high intensity classes. "
        "Trainers are certified and renew their first aid certificate every 2 years. "
        "In case of injury the front desk must be informed immediately and an incident report is filed within 24 hours."),
    "conduct_policy.txt": (
        "Conduct Policy. Harassment of any kind leads to immediate suspension of the person involved. "
        "A harassment complaint must be forwarded to the regional manager within 4 hours and treated as high priority. "
        "Complaints from members are acknowledged within 24 hours. "
        "If an automated assistant cannot resolve a request within 5 steps, it must hand over to a human staff member with a short summary."),
}
for fn, body in DOCS.items():
    with open(f"{POL}/{fn}", "w", encoding="utf-8") as f:
        f.write(body + "\n")

QA = [
    ("What is the maximum number of days a membership can be frozen in a year?", "membership_policy.txt", "maximum of 60 days"),
    ("How much is the freeze fee?", "membership_policy.txt", "INR 200"),
    ("How many days before the billing date must a freeze be requested?", "membership_policy.txt", "at least 3 days"),
    ("How much notice is needed to cancel a membership?", "membership_policy.txt", "30 days"),
    ("What is the late fee for a failed payment?", "payment_policy.txt", "INR 150"),
    ("How soon is a double charge refunded?", "payment_policy.txt", "7 working days"),
    ("What does a day pass for guests cost?", "facility_policy.txt", "INR 300"),
    ("Are children below 16 allowed in the strength area?", "facility_policy.txt", "not allowed"),
    ("What must new members complete before the first workout?", "safety_policy.txt", "health questionnaire"),
    ("How soon must an incident report be filed after an injury?", "safety_policy.txt", "within 24 hours"),
    ("How fast must a harassment complaint reach the regional manager?", "conduct_policy.txt", "within 4 hours"),
    ("After how many steps must an automated assistant hand over to a human?", "conduct_policy.txt", "5 steps"),
    ("What is the name of the gym's mascot?", "NONE", ""),
    ("Which team won the football league last year?", "NONE", ""),
]
write_csv(f"{RAW}/policy_eval.csv", ["question", "expected_doc", "expected_fact"], QA)

print(f"generated: members={len(members)} checkins_raw={len(checkins)} "
      f"payments_raw={len(payments)} tickets={len(tickets)}")
