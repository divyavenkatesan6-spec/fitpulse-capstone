-- FitPulse: 7 SQL queries on data/fitpulse.db (SQLite)
-- Each query is separated by a semicolon.

-- Q1. JOIN + GROUP BY: revenue from paid payments by city and plan
SELECT m.city, m.plan, SUM(p.amount_inr) AS revenue_inr
FROM payments p
JOIN members m ON p.member_id = m.member_id
WHERE p.status = 'paid'
GROUP BY m.city, m.plan
ORDER BY m.city, revenue_inr DESC;

-- Q2. Monthly revenue and number of paid payments
SELECT month, COUNT(*) AS paid_payments, SUM(amount_inr) AS revenue_inr
FROM payments
WHERE status = 'paid'
GROUP BY month
ORDER BY month;

-- Q3. Failed-payment rate by plan (conditional counting)
SELECT m.plan, COUNT(*) AS payments,
       ROUND(100.0 * SUM(p.status = 'failed') / COUNT(*), 2) AS failed_pct
FROM payments p
JOIN members m ON p.member_id = m.member_id
GROUP BY m.plan
ORDER BY failed_pct DESC;

-- Q4. Workout popularity: visits, average duration and calories per workout type
SELECT workout_type, COUNT(*) AS visits,
       ROUND(AVG(duration_min), 1) AS avg_minutes,
       ROUND(AVG(calories), 0) AS avg_calories
FROM checkins
GROUP BY workout_type
ORDER BY visits DESC;

-- Q5. CTE: members who visited MORE than the average member
WITH visits AS (
    SELECT member_id, COUNT(*) AS n_visits
    FROM checkins
    GROUP BY member_id
)
SELECT member_id, n_visits
FROM visits
WHERE n_visits > (SELECT AVG(n_visits) FROM visits)
ORDER BY n_visits DESC
LIMIT 10;

-- Q6. Window function RANK: top 3 most active members in each city
WITH visits AS (
    SELECT m.city, m.member_id, COUNT(*) AS n_visits
    FROM checkins c
    JOIN members m ON c.member_id = m.member_id
    GROUP BY m.city, m.member_id
),
ranked AS (
    SELECT *, RANK() OVER (PARTITION BY city ORDER BY n_visits DESC) AS rnk
    FROM visits
)
SELECT city, member_id, n_visits, rnk
FROM ranked
WHERE rnk <= 3
ORDER BY city, rnk;

-- Q7. Window function: running total of monthly revenue
WITH monthly AS (
    SELECT month, SUM(amount_inr) AS revenue_inr
    FROM payments
    WHERE status = 'paid'
    GROUP BY month
)
SELECT month, revenue_inr,
       SUM(revenue_inr) OVER (ORDER BY month) AS running_total_inr
FROM monthly
ORDER BY month;
