"""Turn the task lists below into tasks/dev.json and tasks/test.json.

This script REFUSES to overwrite task files that differ from what it would
generate. The files on disk are the definition of every historical number in
results/; regenerating them silently would invalidate all of it without an
error. Run with --force only when replacing the task set is the intent, and
expect earlier numbers to become incomparable.

Rules for writing a task:

Every task states its id explicitly, never relying on list position. Position
numbering slips when editing -- one revision shifted everything up by one, so
questions no longer matched their gold SQL while the self-checks stayed green,
because they only verify that the SQL runs.

A question is phrased the way a real user would ask, not paraphrased back from
the SQL. Reverse-engineered questions carry traces of the GROUP BY and ORDER BY
structure, which hands the agent the answer.

A question must determine its answer uniquely. "Which cities have more than 30
users?" does not: city names alone, or names with counts? Both are reasonable,
and if the agent picks one while the gold picked the other, a correct answer
is scored wrong.

Anything about spending or purchasing must say whether unfinished orders count.
"Who has spent the most" -- do cancelled orders count? Both readings hold, the
agent sometimes filters on status and sometimes does not, and the task looks
flaky when the ambiguity is in the question. Write "Include orders of every
status." or "Only count completed orders." check_tasks.py enforces this.

Anything with a LIMIT needs a total order in ORDER BY. Otherwise, when ranks n
and n+1 tie, which one makes the cut is decided by the query plan -- undefined
behaviour, and an unreproducible false failure. Where that is not achievable,
state the tie-break rule in the question. check_tasks.py enforces this too.

`ordered` is True only when the ordering is itself part of the answer: "most
expensive first" is True, "top 3 products" is False, because that asks for a
set.

    python build_tasks.py
    python build_tasks.py --force     # replace the task set deliberately
"""
import argparse
import json
import sys
from pathlib import Path
from . import paths

OUT_DIR = paths.TASKS_DIR

TASKS = [
    # ==================== easy ====================
    {
        "id": "sql_001",
        "question": "How many users do we have?",
        "gold_sql": "SELECT COUNT(*) FROM users",
        "difficulty": "easy",
        "tags": ["count"],
    },
    {
        "id": "sql_002",
        "question": "Show me the name and price of all the electronics we sell, most expensive first.",
        "gold_sql": "SELECT name, price FROM products WHERE category = 'Electronics' ORDER BY price DESC",
        "difficulty": "easy",
        "tags": ["filter", "sort"],
        "ordered": True,
    },
    {
        "id": "sql_003",
        "question": "How many of our users are based in Beijing?",
        "gold_sql": "SELECT COUNT(*) FROM users WHERE city = 'Beijing'",
        "difficulty": "easy",
        "tags": ["count", "filter"],
    },
    {
        "id": "sql_004",
        "question": "What are the names and prices of our three most expensive products?",
        "gold_sql": "SELECT name, price FROM products ORDER BY price DESC LIMIT 3",
        "difficulty": "easy",
        "tags": ["sort", "limit"],
        "ordered": False,
    },
    {
        "id": "sql_005",
        "question": "How many people signed up in 2025 or later?",
        "gold_sql": "SELECT COUNT(*) FROM users WHERE signup_date >= '2025-01-01'",
        "difficulty": "easy",
        "tags": ["count", "date"],
    },

    # ==================== medium: aggregation ====================
    {
        "id": "sql_006",
        "question": "For each city, how many users are there? Show the city and the count.",
        "gold_sql": "SELECT city, COUNT(*) AS n FROM users GROUP BY city",
        "difficulty": "medium",
        "tags": ["group"],
    },
    {
        "id": "sql_007",
        "question": "For each product category, how many products are there and what is the average price?",
        "gold_sql": "SELECT category, COUNT(*) AS n, ROUND(AVG(price), 2) AS avg_price "
                    "FROM products GROUP BY category",
        "difficulty": "medium",
        "tags": ["group", "aggregate"],
    },
    {
        "id": "sql_008",
        "question": "How many orders do we have in each status? Show the status and the count.",
        "gold_sql": "SELECT status, COUNT(*) AS n FROM orders GROUP BY status",
        "difficulty": "medium",
        "tags": ["group"],
    },
    {
        "id": "sql_009",
        "question": "On average, how many line items does an order contain?",
        "gold_sql": "SELECT ROUND(AVG(item_count), 2) FROM ("
                    "  SELECT order_id, COUNT(*) AS item_count FROM order_items GROUP BY order_id)",
        "difficulty": "medium",
        "tags": ["group", "subquery", "aggregate"],
    },
    {
        "id": "sql_010",
        "question": "List the names of cities that have more than 30 users. Only the city names.",
        "gold_sql": "SELECT city FROM users GROUP BY city HAVING COUNT(*) > 30",
        "difficulty": "medium",
        "tags": ["group", "having"],
    },

    # ==================== medium · JOIN ====================
    {
        "id": "sql_011",
        "question": "List the order id and the customer's name for the first 10 orders by order id.",
        "gold_sql": "SELECT o.id, u.name FROM orders o "
                    "JOIN users u ON o.user_id = u.id ORDER BY o.id LIMIT 10",
        "difficulty": "medium",
        "tags": ["join2", "sort", "limit"],
        "ordered": True,
    },
    {
        "id": "sql_012",
        # Ranks 5 and 6 both have 10 orders; without a tie-break, which one
        # makes the top five is undefined. The question has to state the rule,
        # which is also what a real user asking for this should specify.
        "question": "Who are our five most frequent customers? Show the customer name and "
                    "their order count. Break ties by customer name (A to Z).",
        "gold_sql": "SELECT u.name, COUNT(*) AS order_count FROM orders o "
                    "JOIN users u ON o.user_id = u.id GROUP BY u.id "
                    "ORDER BY order_count DESC, u.name ASC LIMIT 5",
        "difficulty": "medium",
        "tags": ["join2", "group", "limit", "tie_break"],
        "ordered": False,
    },
    {
        "id": "sql_013",
        # At LIMIT 5 rank 5 was tied (two products at 269), so the answer was
        # not unique. At 3 the top values are 290/289/276, with no tie.
        "question": "What are our three best-selling products by units sold? "
                    "Show the product name and the units sold.",
        "gold_sql": "SELECT p.name, SUM(oi.quantity) AS total_sold FROM order_items oi "
                    "JOIN products p ON oi.product_id = p.id GROUP BY p.id "
                    "ORDER BY total_sold DESC LIMIT 3",
        "difficulty": "medium",
        "tags": ["join2", "group", "limit"],
        "ordered": False,
    },

    # ==================== hard ====================
    {
        "id": "sql_014",
        # The first version never said how many cities, and the agent added
        # its own LIMIT 5, returning one row fewer than the gold. Reading
        # "which cities generate the most revenue" as "the top few" is
        # perfectly reasonable. A superlative plus a sort has to say how many
        # rows are wanted.
        "question": "Rank every city by revenue, highest first. Only count completed "
                    "orders. Show the city and the revenue.",
        "gold_sql": "SELECT u.city, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue "
                    "FROM orders o "
                    "JOIN users u ON o.user_id = u.id "
                    "JOIN order_items oi ON oi.order_id = o.id "
                    "WHERE o.status = 'completed' GROUP BY u.city ORDER BY revenue DESC",
        "difficulty": "hard",
        "tags": ["join3", "aggregate", "filter", "unit_price"],
        "ordered": True,
    },
    {
        "id": "sql_015",
        "question": "How many users have never placed an order?",
        "gold_sql": "SELECT COUNT(*) FROM users u LEFT JOIN orders o ON u.id = o.user_id "
                    "WHERE o.id IS NULL",
        "difficulty": "hard",
        "tags": ["left_join", "null"],
    },
    {
        "id": "sql_016",
        "question": "What share of our total revenue does each product category account for? "
                    "Show the category, its revenue, and its percentage of the total.",
        "gold_sql": "SELECT p.category, "
                    "  ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue, "
                    "  ROUND(100.0 * SUM(oi.quantity * oi.unit_price) / "
                    "    (SELECT SUM(quantity * unit_price) FROM order_items), 2) AS pct "
                    "FROM order_items oi JOIN products p ON oi.product_id = p.id "
                    "GROUP BY p.category",
        "difficulty": "hard",
        "tags": ["join2", "subquery", "aggregate", "unit_price"],
    },
    {
        "id": "sql_017",
        "question": "Which five customers spend the most per order on average? "
                    "Show the customer name and their average order value. "
                    "Include orders of every status.",
        "gold_sql": "SELECT u.name, ROUND(AVG(order_total), 2) AS avg_order_value FROM ("
                    "  SELECT o.id, o.user_id, SUM(oi.quantity * oi.unit_price) AS order_total "
                    "  FROM orders o JOIN order_items oi ON oi.order_id = o.id GROUP BY o.id) t "
                    "JOIN users u ON t.user_id = u.id GROUP BY u.id "
                    "ORDER BY avg_order_value DESC LIMIT 5",
        "difficulty": "hard",
        "tags": ["subquery", "join2", "aggregate", "unit_price", "group_by_id"],
        "ordered": False,
    },
    {
        "id": "sql_018",
        "question": "How many customers have bought both electronics and books? "
                    "Include orders of every status.",
        "gold_sql": "SELECT COUNT(*) FROM ("
                    "  SELECT o.user_id FROM orders o "
                    "  JOIN order_items oi ON oi.order_id = o.id "
                    "  JOIN products p ON oi.product_id = p.id "
                    "  WHERE p.category IN ('Electronics', 'Books') "
                    "  GROUP BY o.user_id HAVING COUNT(DISTINCT p.category) = 2)",
        "difficulty": "hard",
        "tags": ["join3", "having", "subquery"],
    },

    # ==================== hard: added to lower the baseline ====================
    {
        "id": "sql_019",
        # The first version was ambiguous twice over, each costing a correct
        # answer: it never fixed the label format, so the agent returned
        # 'H1-2025' against a gold of 'H1', both defensible; and it never said
        # whether unfinished orders count, so the agent filtered on status
        # while a neighbouring task explicitly says to -- the task set
        # disagreeing with itself.
        "question": "Compare total revenue in the first half of 2025 versus the second half "
                    "of 2025. Include orders of every status. Label the two periods "
                    "exactly 'H1' and 'H2'.",
        "gold_sql": "SELECT CASE WHEN o.order_date < '2025-07-01' THEN 'H1' ELSE 'H2' END AS half, "
                    "ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue "
                    "FROM orders o JOIN order_items oi ON oi.order_id = o.id "
                    "WHERE o.order_date >= '2025-01-01' AND o.order_date < '2026-01-01' "
                    "GROUP BY half",
        "difficulty": "hard",
        "tags": ["date_window", "case", "join2", "aggregate"],
    },
    {
        "id": "sql_020",
        "question": "For each city, who is the customer that has spent the most? "
                    "Show the city, the customer name, and the amount. "
                    "Include orders of every status.",
        "gold_sql": "SELECT city, name, spent FROM ("
                    "  SELECT u.city, u.name, SUM(oi.quantity * oi.unit_price) AS spent, "
                    "    ROW_NUMBER() OVER (PARTITION BY u.city "
                    "      ORDER BY SUM(oi.quantity * oi.unit_price) DESC) AS rn "
                    "  FROM users u JOIN orders o ON o.user_id = u.id "
                    "  JOIN order_items oi ON oi.order_id = o.id GROUP BY u.id) "
                    "WHERE rn = 1",
        "difficulty": "hard",
        "tags": ["window_function", "per_group_top", "join3"],
        # Explicitly False. The gold has an ORDER BY that a window function
        # needs, and build() would infer ordered=True from it -- but the
        # question asks for no particular order, only one row per city. The
        # inferred True would fail correct answers.
        "ordered": False,
    },
    {
        "id": "sql_021",
        "question": "How many users placed their first order within 30 days of signing up?",
        "gold_sql": "SELECT COUNT(DISTINCT o.user_id) FROM orders o JOIN users u ON u.id = o.user_id "
                    "WHERE julianday(o.order_date) - julianday(u.signup_date) <= 30",
        "difficulty": "hard",
        "tags": ["date_arithmetic", "join2", "distinct"],
    },
    {
        "id": "sql_022",
        # A trap: it asks about the current price, so products.price is
        # required and order_items.unit_price is wrong
        "question": "If we repriced every completed order at today's product prices, "
                    "what would the total value be?",
        "gold_sql": "SELECT ROUND(SUM(oi.quantity * p.price), 2) FROM orders o "
                    "JOIN order_items oi ON oi.order_id = o.id "
                    "JOIN products p ON p.id = oi.product_id "
                    "WHERE o.status = 'completed'",
        "difficulty": "hard",
        "tags": ["trap_current_price", "join3", "aggregate"],
    },
    {
        "id": "sql_023",
        "question": "How many orders contain more than three different products?",
        "gold_sql": "SELECT COUNT(*) FROM (SELECT order_id FROM order_items "
                    "GROUP BY order_id HAVING COUNT(DISTINCT product_id) > 3)",
        "difficulty": "hard",
        "tags": ["having", "distinct", "subquery"],
    },
    {
        "id": "sql_024",
        "question": "Which customers have spent more than twice the average customer's total spend? "
                    "Show their names and how much they spent. "
                    "Include orders of every status.",
        "gold_sql": "SELECT u.name, ROUND(t.spent, 2) FROM ("
                    "  SELECT o.user_id, SUM(oi.quantity * oi.unit_price) AS spent "
                    "  FROM orders o JOIN order_items oi ON oi.order_id = o.id "
                    "  GROUP BY o.user_id) t "
                    "JOIN users u ON u.id = t.user_id "
                    "WHERE t.spent > 2 * (SELECT AVG(s) FROM ("
                    "  SELECT SUM(oi.quantity * oi.unit_price) AS s "
                    "  FROM orders o JOIN order_items oi ON oi.order_id = o.id "
                    "  GROUP BY o.user_id))",
        "difficulty": "hard",
        "tags": ["nested_aggregate", "subquery", "join2"],
    },

    # ============ testing the agent, not the SQL ============
    # The tasks above measure how well the model writes SQL. These measure
    # whether it is a good agent: does it ask when the request is
    # underspecified, admit when the data cannot answer, and report an empty
    # result honestly instead of inventing one.
    #
    # check="behavior" tasks have no gold_sql, because the correct behaviour is
    # not to query and there is no result set to compare. They are scored on
    # structural signals only -- whether submit_answer was called, whether a
    # question was asked -- never on wording. What cannot be scored is declared
    # unscoreable rather than approximated; see verify.verify_behavior.
    {
        "id": "sql_025",
        "question": "Show me the top products.",
        # Top by what: units, revenue, price? All three are reasonable, and
        # the agent should not pick one on the user's behalf
        "check": "behavior",
        "expect": {"submitted": False, "asks_question": True},
        "difficulty": "hard",
        "tags": ["ambiguous", "behavior"],
    },
    {
        "id": "sql_026",
        "question": "How are we doing this quarter?",
        # No metric defined at all. A good agent asks which one is wanted
        "check": "behavior",
        "expect": {"submitted": False, "asks_question": True},
        "difficulty": "hard",
        "tags": ["ambiguous", "behavior"],
    },
    {
        "id": "sql_027",
        "question": "What is our profit margin on each product category?",
        # There is no cost column anywhere, so margin cannot be computed. The
        # correct behaviour is to say the data does not support the question
        "check": "behavior",
        "expect": {"submitted": False, "min_tool_calls": 1},
        "difficulty": "hard",
        "tags": ["out_of_scope", "behavior"],
    },
    {
        "id": "sql_028",
        "question": "Which marketing channel brings in the most users?",
        # The users table has no acquisition-channel column
        "check": "behavior",
        "expect": {"submitted": False, "min_tool_calls": 1},
        "difficulty": "hard",
        "tags": ["out_of_scope", "behavior"],
    },
    {
        "id": "sql_029",
        "question": "How many of our users are based in Paris?",
        # There is no Paris in the database. COUNT(*) returns a single row of
        # zero, and the question is whether it reports the zero or invents
        "gold_sql": "SELECT COUNT(*) FROM users WHERE city = 'Paris'",
        "difficulty": "medium",
        "tags": ["empty_result", "count"],
    },
    {
        "id": "sql_030",
        "question": "List the names of all users who signed up before 2020.",
        # Genuinely zero rows. allow_empty=True says that is intended
        "gold_sql": "SELECT name FROM users WHERE signup_date < '2020-01-01'",
        "difficulty": "medium",
        "tags": ["empty_result", "filter"],
        "allow_empty": True,
    },
    {
        "id": "sql_031",
        "question": "What would our total revenue from completed orders be if every item "
                    "had been sold at its current list price instead of the price actually paid?",
        # A trap: this one deliberately wants products.price rather than
        # unit_price, against the domain rule stated in the system prompt. It
        # tests whether the agent reads the question or applies rules blindly
        "gold_sql": "SELECT ROUND(SUM(oi.quantity * p.price), 2) FROM orders o "
                    "JOIN order_items oi ON oi.order_id = o.id "
                    "JOIN products p ON p.id = oi.product_id "
                    "WHERE o.status = 'completed'",
        "difficulty": "hard",
        "tags": ["trap", "list_price", "join3"],
    },
]


# ==================== generated tasks ====================
# A model produced candidates in bulk, which were then filtered:
#   1. sanity: the gold runs, is non-empty, returns <= 100 rows, and the
#      question contains no SQL keywords
#   2. cross-validation: a model solves the question independently and the
#      task is kept only if its result matches the gold. This is the only
#      check that catches a gold answering a different question than the one
#      asked -- such tasks run fine and return a plausible number of rows, so
#      nothing else notices. It rejected 5.
#   3. the full check_tasks suite
#   4. reading a sample of the questions by hand
# 90 candidates in, 47 kept.
#
# What is automated is the quality checking, not the writing of tasks.
GENERATED_TASKS = [
    {
        "id": "sql_101",
        "question": "Show me the names and signup dates of the 5 most recently signed-up users. If two users signed up on the same day, list the one with the smaller user id first.",
        "gold_sql": "SELECT name, signup_date FROM users ORDER BY signup_date DESC, id ASC LIMIT 5",
        "difficulty": "easy",
        "tags": ["single-table", "sort", "limit"],
        "ordered": True,
    },
    {
        "id": "sql_102",
        "question": "For each product category, what is the total quantity of items sold? Include orders of every status. Show the category and the total quantity.",
        "gold_sql": "SELECT p.category, SUM(oi.quantity) AS total_qty FROM order_items oi JOIN products p ON oi.product_id = p.id JOIN orders o ON oi.order_id = o.id GROUP BY p.category ORDER BY total_qty DESC",
        "difficulty": "medium",
        "tags": ["two-table-join", "aggregation", "group-by", "sort"],
        "ordered": False,
    },
    {
        "id": "sql_103",
        "question": "Which users have placed orders in more than 3 different months? Show the user's name and the number of distinct months they ordered in. Only include completed orders.",
        "gold_sql": "SELECT u.name, COUNT(DISTINCT substr(o.order_date, 1, 7)) AS month_count FROM users u JOIN orders o ON u.id = o.user_id WHERE o.status = 'completed' GROUP BY u.id, u.name HAVING COUNT(DISTINCT substr(o.order_date, 1, 7)) > 3 ORDER BY month_count DESC, u.name ASC",
        "difficulty": "hard",
        "tags": ["two-table-join", "subquery", "having", "date-function"],
        "ordered": False,
    },
    {
        "id": "sql_104",
        "question": "What is the average order value for orders placed in 2025? Include orders of every status. Show the average as a single number rounded to 2 decimal places.",
        "gold_sql": "SELECT ROUND(AVG(order_total), 2) AS avg_order_value FROM (SELECT o.id, SUM(oi.quantity * oi.unit_price) AS order_total FROM orders o JOIN order_items oi ON o.id = oi.order_id WHERE o.order_date LIKE '2025%' GROUP BY o.id) sub",
        "difficulty": "medium",
        "tags": ["subquery", "aggregation", "date-filter"],
        "ordered": False,
    },
    {
        "id": "sql_105",
        "question": "List each product name along with its total revenue from completed orders. Only show products that have generated more than 10,000 in revenue.",
        "gold_sql": "SELECT p.name, SUM(oi.quantity * oi.unit_price) AS revenue FROM products p JOIN order_items oi ON p.id = oi.product_id JOIN orders o ON oi.order_id = o.id WHERE o.status = 'completed' GROUP BY p.id, p.name HAVING SUM(oi.quantity * oi.unit_price) > 10000 ORDER BY revenue DESC, p.name ASC",
        "difficulty": "medium",
        "tags": ["two-table-join", "aggregation", "having", "sort"],
        "ordered": False,
    },
    {
        "id": "sql_106",
        "question": "For each product category, what is the total revenue from completed orders? Show the category and the revenue, rounded to 2 decimal places.",
        "gold_sql": "SELECT p.category, ROUND(SUM(oi.unit_price * oi.quantity), 2) AS revenue FROM order_items oi JOIN orders o ON oi.order_id = o.id JOIN products p ON oi.product_id = p.id WHERE o.status = 'completed' GROUP BY p.category ORDER BY revenue DESC, p.category ASC",
        "difficulty": "medium",
        "tags": ["join", "aggregation", "filter"],
        "ordered": False,
    },
    {
        "id": "sql_107",
        "question": "List the names of users who have placed more than 5 orders in total (any status) but have never placed an order with status 'cancelled'. Show only the user names.",
        "gold_sql": "SELECT u.name FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name HAVING COUNT(*) > 5 AND SUM(CASE WHEN o.status = 'cancelled' THEN 1 ELSE 0 END) = 0 ORDER BY u.name ASC",
        "difficulty": "hard",
        "tags": ["join", "aggregation", "having", "filter"],
        "ordered": False,
    },
    {
        "id": "sql_108",
        "question": "Show the names of products in the Electronics category that have been ordered at least once. Return only the product names.",
        "gold_sql": "SELECT DISTINCT p.name FROM products p JOIN order_items oi ON p.id = oi.product_id WHERE p.category = 'Electronics' ORDER BY p.name ASC",
        "difficulty": "easy",
        "tags": ["join", "distinct", "filter"],
        "ordered": False,
    },
    {
        "id": "sql_109",
        "question": "What is the total revenue from completed orders placed by users from Chengdu? Show the total revenue as a single number rounded to 2 decimal places.",
        "gold_sql": "SELECT ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_revenue FROM users u JOIN orders o ON u.id = o.user_id JOIN order_items oi ON o.id = oi.order_id WHERE u.city = 'Chengdu' AND o.status = 'completed'",
        "difficulty": "easy",
        "tags": ["join", "aggregation", "filter", "single value"],
        "ordered": False,
    },
    {
        "id": "sql_110",
        "question": "Which products have been ordered by users from more than 2 different cities? Show the product name and the number of distinct cities. Include orders of every status.",
        "gold_sql": "SELECT p.name, COUNT(DISTINCT u.city) AS city_count FROM products p JOIN order_items oi ON p.id = oi.product_id JOIN orders o ON oi.order_id = o.id JOIN users u ON o.user_id = u.id GROUP BY p.id, p.name HAVING COUNT(DISTINCT u.city) > 2 ORDER BY city_count DESC, p.name ASC",
        "difficulty": "hard",
        "tags": ["join", "aggregation", "having", "distinct"],
        "ordered": False,
    },
    {
        "id": "sql_111",
        "question": "Show the total quantity sold for each product name, including all orders regardless of status.",
        "gold_sql": "SELECT p.name, SUM(oi.quantity) AS total_quantity FROM products p JOIN order_items oi ON p.id = oi.product_id GROUP BY p.id, p.name ORDER BY total_quantity DESC, p.name ASC",
        "difficulty": "easy",
        "tags": ["join", "aggregation", "group by", "sorting"],
        "ordered": False,
    },
    {
        "id": "sql_112",
        "question": "For each product category, show the category and the number of distinct products in that category that have been ordered at least once. Only count orders with status 'completed'.",
        "gold_sql": "SELECT p.category, COUNT(DISTINCT p.id) AS distinct_products_ordered FROM products p JOIN order_items oi ON p.id = oi.product_id JOIN orders o ON oi.order_id = o.id WHERE o.status = 'completed' GROUP BY p.category ORDER BY distinct_products_ordered DESC, p.category ASC",
        "difficulty": "medium",
        "tags": ["join", "filter", "distinct", "aggregation", "sorting"],
        "ordered": False,
    },
    {
        "id": "sql_113",
        "question": "What is the average total spend per user for users who have placed at least one completed order? Show the average as a single number rounded to 2 decimal places. Use unit_price * quantity for each line item.",
        "gold_sql": "SELECT ROUND(AVG(user_spend), 2) AS avg_user_spend FROM (SELECT o.user_id, SUM(oi.unit_price * oi.quantity) AS user_spend FROM orders o JOIN order_items oi ON o.id = oi.order_id WHERE o.status = 'completed' GROUP BY o.user_id) AS spend_per_user",
        "difficulty": "medium",
        "tags": ["subquery", "aggregation", "filter", "rounding"],
        "ordered": False,
    },
    {
        "id": "sql_114",
        "question": "Show the names of the 3 products with the highest total revenue from completed orders. If two products have the same total revenue, list the one with the smaller product id first. Return the product name and the total revenue rounded to 2 decimal places.",
        "gold_sql": "SELECT p.name, ROUND(SUM(oi.unit_price * oi.quantity), 2) AS revenue FROM products p JOIN order_items oi ON p.id = oi.product_id JOIN orders o ON oi.order_id = o.id WHERE o.status = 'completed' GROUP BY p.id, p.name ORDER BY revenue DESC, p.id ASC LIMIT 3",
        "difficulty": "medium",
        "tags": ["join", "aggregation", "limit", "sorting", "filter"],
        "ordered": True,
    },
    {
        "id": "sql_115",
        "question": "Which products have been ordered in every month of 2025? Show the product name and the number of distinct months in 2025 in which they were ordered. Only include completed orders.",
        "gold_sql": "SELECT p.name, COUNT(DISTINCT substr(o.order_date, 1, 7)) AS months_ordered FROM products p JOIN order_items oi ON p.id = oi.product_id JOIN orders o ON oi.order_id = o.id WHERE o.status = 'completed' AND o.order_date LIKE '2025%' GROUP BY p.id, p.name HAVING COUNT(DISTINCT substr(o.order_date, 1, 7)) = 12 ORDER BY p.name ASC",
        "difficulty": "hard",
        "tags": ["join", "filter", "group by", "having", "date extraction", "sorting"],
        "ordered": False,
    },
    {
        "id": "sql_116",
        "question": "Show the names of the 5 users with the most completed orders. If two users have the same number of completed orders, list the one with the smaller user id first. Return user name and the count of completed orders.",
        "gold_sql": "SELECT u.name, COUNT(o.id) AS completed_orders FROM users u JOIN orders o ON o.user_id = u.id WHERE o.status = 'completed' GROUP BY u.id, u.name ORDER BY completed_orders DESC, u.id ASC LIMIT 5",
        "difficulty": "medium",
        "tags": ["join", "aggregation", "limit", "tie-breaker"],
        "ordered": True,
    },
    {
        "id": "sql_117",
        "question": "For each product, show the product name and the number of distinct users who have ordered it in completed orders. Only include products that have been ordered by at least 3 distinct users.",
        "gold_sql": "SELECT p.name, COUNT(DISTINCT o.user_id) AS distinct_users FROM products p JOIN order_items oi ON oi.product_id = p.id JOIN orders o ON o.id = oi.order_id WHERE o.status = 'completed' GROUP BY p.id, p.name HAVING COUNT(DISTINCT o.user_id) >= 3 ORDER BY distinct_users DESC, p.name ASC",
        "difficulty": "medium",
        "tags": ["join", "aggregation", "having"],
        "ordered": False,
    },
    {
        "id": "sql_118",
        "question": "What is the total revenue from completed orders for each month in 2025? Show the month (as 'YYYY-MM') and the revenue rounded to 2 decimal places.",
        "gold_sql": "SELECT substr(o.order_date, 1, 7) AS month, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue FROM orders o JOIN order_items oi ON oi.order_id = o.id WHERE o.status = 'completed' AND o.order_date LIKE '2025-%' GROUP BY substr(o.order_date, 1, 7) ORDER BY month ASC",
        "difficulty": "medium",
        "tags": ["date", "aggregation", "join"],
        "ordered": False,
    },
    {
        "id": "sql_119",
        "question": "List the names of users whose total spend from completed orders is above the average total spend from completed orders across all users. Show user name and total spend rounded to 2 decimal places.",
        "gold_sql": "WITH user_spend AS (SELECT u.id, u.name, SUM(oi.quantity * oi.unit_price) AS total_spend FROM users u JOIN orders o ON o.user_id = u.id JOIN order_items oi ON oi.order_id = o.id WHERE o.status = 'completed' GROUP BY u.id, u.name) SELECT name, ROUND(total_spend, 2) AS total_spend FROM user_spend WHERE total_spend > (SELECT AVG(total_spend) FROM user_spend) ORDER BY total_spend DESC, name ASC",
        "difficulty": "hard",
        "tags": ["subquery", "cte", "aggregation"],
        "ordered": False,
    },
    {
        "id": "sql_120",
        "question": "For each product category, show the category and the average unit price actually paid for items in that category, rounded to 2 decimal places. Only include completed orders.",
        "gold_sql": "SELECT p.category, ROUND(AVG(oi.unit_price), 2) AS avg_paid_price FROM products p JOIN order_items oi ON oi.product_id = p.id JOIN orders o ON o.id = oi.order_id WHERE o.status = 'completed' GROUP BY p.category ORDER BY avg_paid_price DESC, p.category ASC",
        "difficulty": "easy",
        "tags": ["join", "aggregation", "group-by"],
        "ordered": False,
    },
    {
        "id": "sql_121",
        "question": "Show the names of the 3 products with the lowest total quantity sold across all orders (any status). If two products have the same total quantity, list the one with the larger product id first. Only include products that have been ordered at least once. Return product name and total quantity.",
        "gold_sql": "SELECT p.name, SUM(oi.quantity) AS total_qty FROM products p JOIN order_items oi ON oi.product_id = p.id GROUP BY p.id, p.name ORDER BY total_qty ASC, p.id DESC LIMIT 3",
        "difficulty": "medium",
        "tags": ["join", "aggregation", "limit", "tie-breaker"],
        "ordered": True,
    },
    {
        "id": "sql_122",
        "question": "List the user name and email of every user who has never placed a completed order.",
        "gold_sql": "SELECT u.name, u.email FROM users u LEFT JOIN orders o ON o.user_id = u.id AND o.status = 'completed' WHERE o.id IS NULL ORDER BY u.name ASC",
        "difficulty": "easy",
        "tags": ["left-join", "filter", "null"],
        "ordered": False,
    },
    {
        "id": "sql_123",
        "question": "Show the names and email addresses of users who have never placed any order.",
        "gold_sql": "SELECT u.name, u.email FROM users u LEFT JOIN orders o ON u.id = o.user_id WHERE o.id IS NULL ORDER BY u.name ASC",
        "difficulty": "easy",
        "tags": ["join", "anti-join", "filter"],
        "ordered": False,
    },
    {
        "id": "sql_124",
        "question": "What is the total revenue from all orders placed by users who signed up in 2024? Include orders of every status. Show the total revenue as a single number rounded to 2 decimal places.",
        "gold_sql": "SELECT ROUND(SUM(oi.quantity * oi.unit_price), 2) FROM orders o JOIN users u ON o.user_id = u.id JOIN order_items oi ON o.id = oi.order_id WHERE u.signup_date >= '2024-01-01' AND u.signup_date < '2025-01-01'",
        "difficulty": "medium",
        "tags": ["join", "aggregation", "date-filter", "revenue"],
        "ordered": False,
    },
    {
        "id": "sql_125",
        "question": "For each product, show the product name and the number of distinct orders that include it. Include only orders with status 'completed'. Only show products that have appeared in at least 2 distinct completed orders.",
        "gold_sql": "SELECT p.name, COUNT(DISTINCT oi.order_id) AS order_count FROM products p JOIN order_items oi ON p.id = oi.product_id JOIN orders o ON oi.order_id = o.id WHERE o.status = 'completed' GROUP BY p.name HAVING COUNT(DISTINCT oi.order_id) >= 2 ORDER BY order_count DESC, p.name ASC",
        "difficulty": "medium",
        "tags": ["join", "aggregation", "having", "distinct-count"],
        "ordered": False,
    },
    {
        "id": "sql_126",
        "question": "Which users have placed orders in both January 2025 and February 2025? Show the user name and the number of distinct orders they placed in each of those months. Only include completed orders.",
        "gold_sql": "SELECT u.name, SUM(CASE WHEN substr(o.order_date, 1, 7) = '2025-01' THEN 1 ELSE 0 END) AS jan_orders, SUM(CASE WHEN substr(o.order_date, 1, 7) = '2025-02' THEN 1 ELSE 0 END) AS feb_orders FROM users u JOIN orders o ON u.id = o.user_id WHERE o.status = 'completed' AND substr(o.order_date, 1, 7) IN ('2025-01', '2025-02') GROUP BY u.id, u.name HAVING SUM(CASE WHEN substr(o.order_date, 1, 7) = '2025-01' THEN 1 ELSE 0 END) > 0 AND SUM(CASE WHEN substr(o.order_date, 1, 7) = '2025-02' THEN 1 ELSE 0 END) > 0 ORDER BY u.name ASC",
        "difficulty": "hard",
        "tags": ["conditional-aggregation", "date-filter", "having", "join"],
        "ordered": False,
    },
    {
        "id": "sql_127",
        "question": "Show the order id, the customer's name, and the total value (sum of quantity * unit_price) for each order that has a total value higher than every order placed by users from Chengdu. Include orders of every status.",
        "gold_sql": "SELECT o.id, u.name, SUM(oi.quantity * oi.unit_price) AS total_value FROM orders o JOIN users u ON o.user_id = u.id JOIN order_items oi ON o.id = oi.order_id GROUP BY o.id, u.name HAVING SUM(oi.quantity * oi.unit_price) > (SELECT MAX(sub.total_value) FROM (SELECT o2.id, SUM(oi2.quantity * oi2.unit_price) AS total_value FROM orders o2 JOIN users u2 ON o2.user_id = u2.id JOIN order_items oi2 ON o2.id = oi2.order_id WHERE u2.city = 'Chengdu' GROUP BY o2.id) sub) ORDER BY o.id ASC",
        "difficulty": "hard",
        "tags": ["subquery", "correlated", "aggregation", "comparison"],
        "ordered": False,
    },
    {
        "id": "sql_128",
        "question": "Show the product name and the total revenue from completed orders for each product in the Electronics category. Only include products that have been ordered at least once.",
        "gold_sql": "SELECT p.name AS product_name, SUM(oi.quantity * oi.unit_price) AS total_revenue FROM products p JOIN order_items oi ON p.id = oi.product_id JOIN orders o ON oi.order_id = o.id WHERE p.category = 'Electronics' AND o.status = 'completed' GROUP BY p.id, p.name ORDER BY total_revenue DESC, p.name ASC",
        "difficulty": "medium",
        "tags": ["join", "aggregation", "filter", "revenue"],
        "ordered": False,
    },
    {
        "id": "sql_129",
        "question": "What is the average order value (sum of quantity * unit_price per order) for orders placed in 2024? Show the average as a single number rounded to 2 decimal places. Include orders of every status.",
        "gold_sql": "SELECT ROUND(AVG(order_value), 2) AS avg_order_value FROM (SELECT o.id, SUM(oi.quantity * oi.unit_price) AS order_value FROM orders o JOIN order_items oi ON o.id = oi.order_id WHERE o.order_date LIKE '2024-%' GROUP BY o.id) sub",
        "difficulty": "medium",
        "tags": ["subquery", "aggregation", "revenue", "date"],
        "ordered": False,
    },
    {
        "id": "sql_130",
        "question": "For each product, show the product name and the rank of its total quantity sold (across all orders, any status) among all products. Rank 1 for the highest total quantity. If two products have the same total quantity, give them the same rank and skip the next rank.",
        "gold_sql": "SELECT p.name AS product_name, RANK() OVER (ORDER BY SUM(oi.quantity) DESC) AS quantity_rank FROM products p JOIN order_items oi ON p.id = oi.product_id GROUP BY p.id, p.name ORDER BY quantity_rank ASC, p.name ASC",
        "difficulty": "hard",
        "tags": ["window function", "rank", "join", "aggregation"],
        "ordered": False,
    },
    {
        "id": "sql_131",
        "question": "What is the total revenue from completed orders for each product category? Show the category and the revenue rounded to 2 decimal places. Only include categories that have at least one completed order line item.",
        "gold_sql": "SELECT p.category, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue FROM products p JOIN order_items oi ON p.id = oi.product_id JOIN orders o ON oi.order_id = o.id WHERE o.status = 'completed' GROUP BY p.category HAVING SUM(oi.quantity * oi.unit_price) > 0 ORDER BY revenue DESC, p.category ASC",
        "difficulty": "medium",
        "tags": ["join", "aggregation", "having", "money"],
        "ordered": False,
    },
    {
        "id": "sql_132",
        "question": "Show the names of the 2 products with the highest number of distinct users who have ordered them in completed orders. If two products have the same number of distinct users, list the one with the smaller product id first. Return the product name and the count of distinct users.",
        "gold_sql": "SELECT p.name, COUNT(DISTINCT o.user_id) AS user_count FROM products p JOIN order_items oi ON p.id = oi.product_id JOIN orders o ON oi.order_id = o.id WHERE o.status = 'completed' GROUP BY p.id, p.name ORDER BY user_count DESC, p.id ASC LIMIT 2",
        "difficulty": "medium",
        "tags": ["join", "aggregation", "limit", "distinct"],
        "ordered": True,
    },
    {
        "id": "sql_133",
        "question": "List the names of users who have placed at least one order in 2025 but have never placed an order in 2024. Return only the user names.",
        "gold_sql": "SELECT DISTINCT u.name FROM users u JOIN orders o ON u.id = o.user_id WHERE o.order_date LIKE '2025%' AND u.id NOT IN (SELECT user_id FROM orders WHERE order_date LIKE '2024%') ORDER BY u.name ASC",
        "difficulty": "medium",
        "tags": ["subquery", "distinct", "date filter", "not in"],
        "ordered": False,
    },
    {
        "id": "sql_134",
        "question": "For each month in 2025, show the month (as 'YYYY-MM') and the number of distinct products that were ordered in completed orders during that month. Only include months that have at least one completed order.",
        "gold_sql": "SELECT substr(o.order_date, 1, 7) AS month, COUNT(DISTINCT oi.product_id) AS product_count FROM orders o JOIN order_items oi ON o.id = oi.order_id WHERE o.order_date LIKE '2025%' AND o.status = 'completed' GROUP BY substr(o.order_date, 1, 7) HAVING COUNT(DISTINCT oi.product_id) > 0 ORDER BY month ASC",
        "difficulty": "medium",
        "tags": ["aggregation", "date extraction", "distinct", "group by"],
        "ordered": False,
    },
    {
        "id": "sql_135",
        "question": "Show the user name and the total value of their orders placed in January 2025, but only for users who have placed at least one order in that month. Include all order statuses.",
        "gold_sql": "SELECT u.name, SUM(oi.quantity * oi.unit_price) AS total_value FROM users u JOIN orders o ON u.id = o.user_id JOIN order_items oi ON o.id = oi.order_id WHERE o.order_date LIKE '2025-01%' GROUP BY u.id, u.name ORDER BY total_value DESC, u.name ASC",
        "difficulty": "medium",
        "tags": ["join", "aggregation", "date-filter"],
        "ordered": False,
    },
    {
        "id": "sql_136",
        "question": "List the names of cities that have at least 2 users who have placed at least one completed order. Return only the city names.",
        "gold_sql": "SELECT u.city FROM users u JOIN orders o ON u.id = o.user_id WHERE o.status = 'completed' GROUP BY u.city HAVING COUNT(DISTINCT u.id) >= 2 ORDER BY u.city ASC",
        "difficulty": "medium",
        "tags": ["join", "aggregation", "distinct-count", "status-filter"],
        "ordered": False,
    },
    {
        "id": "sql_137",
        "question": "Show the product name and the total revenue from completed orders for the product with the highest total revenue. If there is a tie, show all tied products. Return product name and revenue rounded to 2 decimal places.",
        "gold_sql": "SELECT p.name, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue FROM products p JOIN order_items oi ON p.id = oi.product_id JOIN orders o ON oi.order_id = o.id WHERE o.status = 'completed' GROUP BY p.id, p.name HAVING SUM(oi.quantity * oi.unit_price) = (SELECT MAX(revenue) FROM (SELECT SUM(oi2.quantity * oi2.unit_price) AS revenue FROM order_items oi2 JOIN orders o2 ON oi2.order_id = o2.id WHERE o2.status = 'completed' GROUP BY oi2.product_id) AS sub) ORDER BY p.name ASC",
        "difficulty": "hard",
        "tags": ["subquery", "join", "aggregation", "status-filter"],
        "ordered": False,
    },
    {
        "id": "sql_138",
        "question": "Show the names of the 5 users with the highest total quantity of items ordered across all orders (any status). If two users have the same total quantity, list the one with the smaller user id first. Return user name and total quantity.",
        "gold_sql": "SELECT u.name, SUM(oi.quantity) AS total_quantity FROM users u JOIN orders o ON u.id = o.user_id JOIN order_items oi ON o.id = oi.order_id GROUP BY u.id, u.name ORDER BY total_quantity DESC, u.id ASC LIMIT 5",
        "difficulty": "medium",
        "tags": ["join", "aggregation", "limit", "tie-breaker"],
        "ordered": True,
    },
    {
        "id": "sql_139",
        "question": "For each month in 2024, show the month (as 'YYYY-MM') and the total revenue from completed orders placed in that month. Only include months that have at least one completed order.",
        "gold_sql": "SELECT strftime('%Y-%m', o.order_date) AS month, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue FROM orders o JOIN order_items oi ON o.id = oi.order_id WHERE o.status = 'completed' AND o.order_date LIKE '2024-%' GROUP BY month ORDER BY month ASC",
        "difficulty": "medium",
        "tags": ["date", "aggregation", "filter"],
        "ordered": False,
    },
    {
        "id": "sql_140",
        "question": "What is the average total spend per user from completed orders, but only counting users who have placed at least one completed order? Show the average as a single number rounded to 2 decimal places.",
        "gold_sql": "SELECT ROUND(AVG(user_spend), 2) AS avg_spend FROM (SELECT u.id, SUM(oi.quantity * oi.unit_price) AS user_spend FROM users u JOIN orders o ON u.id = o.user_id JOIN order_items oi ON o.id = oi.order_id WHERE o.status = 'completed' GROUP BY u.id) AS spend_per_user",
        "difficulty": "hard",
        "tags": ["subquery", "aggregation", "average"],
        "ordered": False,
    },
    {
        "id": "sql_141",
        "question": "For each product category, show the category and the number of distinct users who have ordered products in that category from completed orders. Only include categories with at least one such user.",
        "gold_sql": "SELECT p.category, COUNT(DISTINCT u.id) AS user_count FROM products p JOIN order_items oi ON p.id = oi.product_id JOIN orders o ON oi.order_id = o.id JOIN users u ON o.user_id = u.id WHERE o.status = 'completed' GROUP BY p.category HAVING user_count > 0 ORDER BY user_count DESC, p.category ASC",
        "difficulty": "medium",
        "tags": ["join", "aggregation", "distinct", "group by"],
        "ordered": False,
    },
    {
        "id": "sql_142",
        "question": "Show the names of users who have placed at least one order in every month from January 2025 through March 2025 (i.e., all three months). Only count completed orders. Return user names.",
        "gold_sql": "SELECT u.name FROM users u JOIN orders o ON u.id = o.user_id WHERE o.status = 'completed' AND o.order_date BETWEEN '2025-01-01' AND '2025-03-31' GROUP BY u.id, u.name HAVING COUNT(DISTINCT strftime('%Y-%m', o.order_date)) = 3 ORDER BY u.name ASC",
        "difficulty": "hard",
        "tags": ["date", "aggregation", "having", "distinct"],
        "ordered": False,
    },
    {
        "id": "sql_143",
        "question": "For each product, show the product name and the total quantity ordered across all orders of any status. Only include products that have been ordered at least once.",
        "gold_sql": "SELECT p.name, SUM(oi.quantity) AS total_qty FROM products p JOIN order_items oi ON p.id = oi.product_id GROUP BY p.id, p.name ORDER BY total_qty DESC, p.name ASC",
        "difficulty": "easy",
        "tags": ["aggregation", "join", "group by"],
        "ordered": False,
    },
    {
        "id": "sql_144",
        "question": "What is the total revenue from completed orders placed in the month of March 2025? Show the total as a single number rounded to 2 decimal places.",
        "gold_sql": "SELECT ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_revenue FROM orders o JOIN order_items oi ON o.id = oi.order_id WHERE o.status = 'completed' AND o.order_date LIKE '2025-03-%'",
        "difficulty": "easy",
        "tags": ["aggregation", "date filter", "status filter"],
        "ordered": False,
    },
    {
        "id": "sql_145",
        "question": "Show the names of the 3 products with the highest total revenue from completed orders. If two products have the same total revenue, list the one with the smaller product id first. Return product name and total revenue rounded to 2 decimal places.",
        "gold_sql": "SELECT p.name, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue FROM products p JOIN order_items oi ON p.id = oi.product_id JOIN orders o ON oi.order_id = o.id WHERE o.status = 'completed' GROUP BY p.id, p.name ORDER BY revenue DESC, p.id ASC LIMIT 3",
        "difficulty": "medium",
        "tags": ["aggregation", "join", "limit", "tie-breaker"],
        "ordered": True,
    },
    {
        "id": "sql_146",
        "question": "Show the names of products that have been ordered by users from more than 3 different cities. Only include completed orders. Return product name and the number of distinct cities.",
        "gold_sql": "SELECT p.name, COUNT(DISTINCT u.city) AS city_count FROM products p JOIN order_items oi ON p.id = oi.product_id JOIN orders o ON oi.order_id = o.id JOIN users u ON o.user_id = u.id WHERE o.status = 'completed' GROUP BY p.id, p.name HAVING COUNT(DISTINCT u.city) > 3 ORDER BY city_count DESC, p.name ASC",
        "difficulty": "medium",
        "tags": ["join", "having", "distinct count"],
        "ordered": False,
    },
    {
        "id": "sql_147",
        "question": "Show the names of the 3 products with the lowest total quantity ordered across all orders of any status. If two products have the same total quantity, list the one with the smaller product id first. Return product name and total quantity.",
        "gold_sql": "SELECT p.name, COALESCE(SUM(oi.quantity), 0) AS total_qty FROM products p LEFT JOIN order_items oi ON p.id = oi.product_id GROUP BY p.id, p.name ORDER BY total_qty ASC, p.id ASC LIMIT 3",
        "difficulty": "medium",
        "tags": ["aggregation", "left join", "limit", "tie-breaker"],
        "ordered": True,
    },
]


# ==================== held-out test set ====================
# Locked. Run once, when a final number is reported.
#
# Why it has to be independent of dev: several experiments and rounds of
# prompt tuning have run against dev, so those numbers are overfitted to it by
# construction. This set was generated separately and has taken part in no
# decision whatsoever.
#
# Generated exactly as dev was, through the same four filters. 40 candidates
# in, 24 kept.
TEST_TASKS = [
    {
        "id": "test_501",
        "question": "Show the names of users who signed up in 2024 and have placed at least one completed order. Return only the user names.",
        "gold_sql": "SELECT DISTINCT u.name FROM users u JOIN orders o ON u.id = o.user_id WHERE u.signup_date LIKE '2024%' AND o.status = 'completed'",
        "difficulty": "easy",
        "tags": ["join", "filter", "distinct"],
        "ordered": False,
    },
    {
        "id": "test_502",
        "question": "Show the email address of every user who has placed an order containing the product named 'Laptop Pro 14'. Include orders of every status. Return only the email addresses.",
        "gold_sql": "SELECT DISTINCT u.email FROM users u JOIN orders o ON u.id = o.user_id JOIN order_items oi ON o.id = oi.order_id JOIN products p ON oi.product_id = p.id WHERE p.name = 'Laptop Pro 14'",
        "difficulty": "easy",
        "tags": ["join", "filter", "distinct"],
        "ordered": False,
    },
    {
        "id": "test_503",
        "question": "For each city, show the city name and the total revenue from completed orders placed by users in that city. Only include cities with at least one completed order.",
        "gold_sql": "SELECT u.city, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue FROM users u JOIN orders o ON u.id = o.user_id JOIN order_items oi ON o.id = oi.order_id WHERE o.status = 'completed' GROUP BY u.city HAVING COUNT(o.id) > 0 ORDER BY revenue DESC, u.city ASC",
        "difficulty": "medium",
        "tags": ["join", "aggregation", "group by", "order by"],
        "ordered": False,
    },
    {
        "id": "test_504",
        "question": "Show the product name and the rank of its total quantity sold (from completed orders only) among all products. Rank 1 for the highest total quantity. Only include products that have been ordered at least once in completed orders.",
        "gold_sql": "SELECT p.name, RANK() OVER (ORDER BY SUM(oi.quantity) DESC) AS rank FROM products p JOIN order_items oi ON p.id = oi.product_id JOIN orders o ON oi.order_id = o.id WHERE o.status = 'completed' GROUP BY p.id, p.name HAVING COUNT(oi.id) > 0 ORDER BY rank ASC, p.name ASC",
        "difficulty": "hard",
        "tags": ["window function", "join", "aggregation"],
        "ordered": False,
    },
    {
        "id": "test_505",
        "question": "Show the names and signup dates of users who signed up in 2025.",
        "gold_sql": "SELECT name, signup_date FROM users WHERE signup_date >= '2025-01-01' AND signup_date < '2026-01-01' ORDER BY signup_date ASC, id ASC",
        "difficulty": "easy",
        "tags": ["single-table", "filter", "ordering"],
        "ordered": False,
    },
    {
        "id": "test_506",
        "question": "Show the names of products that have been. Only include completed orders. Return product name and the number of distinct users who ordered it.",
        "gold_sql": "SELECT p.name, COUNT(DISTINCT o.user_id) AS user_count FROM products p JOIN order_items oi ON p.id = oi.product_id JOIN orders o ON oi.order_id = o.id WHERE o.status = 'completed' GROUP BY p.id, p.name HAVING COUNT(DISTINCT o.user_id) >= 2 ORDER BY user_count DESC, p.id ASC",
        "difficulty": "medium",
        "tags": ["three-table-join", "aggregation", "having", "ordering"],
        "ordered": False,
    },
    {
        "id": "test_507",
        "question": "What is the total revenue from completed orders for each city? Show the city name and the revenue rounded to 2 decimal places. Only include cities that have at least one completed order.",
        "gold_sql": "SELECT u.city, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue FROM users u JOIN orders o ON u.id = o.user_id JOIN order_items oi ON o.id = oi.order_id WHERE o.status = 'completed' GROUP BY u.city HAVING COUNT(o.id) > 0 ORDER BY revenue DESC, u.city ASC",
        "difficulty": "medium",
        "tags": ["three-table-join", "aggregation", "rounding", "ordering"],
        "ordered": False,
    },
    {
        "id": "test_508",
        "question": "Show the product name and the total quantity sold (from completed orders only) for each product, along with the difference between that total quantity and the average total quantity sold per product (from completed orders only). Round the difference to 2 decimal places. Only include products that have been ordered at least once in completed orders.",
        "gold_sql": "SELECT p.name, SUM(oi.quantity) AS total_quantity, ROUND(SUM(oi.quantity) - (SELECT AVG(qty) FROM (SELECT SUM(oi2.quantity) AS qty FROM order_items oi2 JOIN orders o2 ON oi2.order_id = o2.id WHERE o2.status = 'completed' GROUP BY oi2.product_id) avg_qty), 2) AS diff_from_avg FROM products p JOIN order_items oi ON p.id = oi.product_id JOIN orders o ON oi.order_id = o.id WHERE o.status = 'completed' GROUP BY p.id, p.name ORDER BY diff_from_avg DESC, p.id ASC",
        "difficulty": "hard",
        "tags": ["subquery", "aggregation", "correlated", "ordering"],
        "ordered": False,
    },
    {
        "id": "test_509",
        "question": "Show the names and emails of users who live in Shanghai and have placed at least one order with status 'completed'. Return user name and email only.",
        "gold_sql": "SELECT DISTINCT u.name, u.email FROM users u JOIN orders o ON u.id = o.user_id WHERE u.city = 'Shanghai' AND o.status = 'completed'",
        "difficulty": "easy",
        "tags": ["join", "filter", "distinct"],
        "ordered": False,
    },
    {
        "id": "test_510",
        "question": "For each product, show the product name and the total revenue (sum of quantity * unit_price) from orders of every status. Only include products that have been ordered at least once.",
        "gold_sql": "SELECT p.name, SUM(oi.quantity * oi.unit_price) AS total_revenue FROM products p JOIN order_items oi ON p.id = oi.product_id GROUP BY p.id, p.name ORDER BY p.name ASC",
        "difficulty": "medium",
        "tags": ["join", "aggregation", "group by", "order by"],
        "ordered": False,
    },
    {
        "id": "test_511",
        "question": "Show the names of the 3 cities with the highest number of distinct users who have placed at least one completed order. If two cities have the same count, list the one with the smaller city name alphabetically first. Return city name and the count of distinct users.",
        "gold_sql": "SELECT u.city, COUNT(DISTINCT u.id) AS user_count FROM users u JOIN orders o ON u.id = o.user_id WHERE o.status = 'completed' GROUP BY u.city ORDER BY user_count DESC, u.city ASC LIMIT 3",
        "difficulty": "medium",
        "tags": ["join", "aggregation", "limit", "order by"],
        "ordered": True,
    },
    {
        "id": "test_512",
        "question": "Show the product name and the total revenue from completed orders for each product in the 'Electronics' category. Only include products that have been ordered at least once in completed orders. Round the revenue to 2 decimal places.",
        "gold_sql": "SELECT p.name, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue FROM products p JOIN order_items oi ON p.id = oi.product_id JOIN orders o ON oi.order_id = o.id AND o.status = 'completed' WHERE p.category = 'Electronics' GROUP BY p.id, p.name HAVING SUM(oi.quantity) IS NOT NULL ORDER BY revenue DESC, p.id ASC",
        "difficulty": "medium",
        "tags": ["join", "filter", "aggregation", "order by"],
        "ordered": False,
    },
    {
        "id": "test_513",
        "question": "Show the names of users who have placed at least one order in 2025 but have not placed any order in 2024, and also have placed at least one order with status 'completed' in 2025. Return only the user names, ordered alphabetically.",
        "gold_sql": "SELECT DISTINCT u.name FROM users u WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id AND o.order_date LIKE '2025-%' AND o.status = 'completed') AND NOT EXISTS (SELECT 1 FROM orders o2 WHERE o2.user_id = u.id AND o2.order_date LIKE '2024-%') ORDER BY u.name ASC",
        "difficulty": "hard",
        "tags": ["subquery", "exists", "not exists", "order by"],
        "ordered": False,
    },
    {
        "id": "test_514",
        "question": "Show the names of the 3 products with the highest list price. If two products have the same list price, list the one with the smaller product id first. Return product name and price.",
        "gold_sql": "SELECT name, price FROM products ORDER BY price DESC, id ASC LIMIT 3",
        "difficulty": "easy",
        "tags": ["single-table", "limit", "order-by", "products"],
        "ordered": True,
    },
    {
        "id": "test_515",
        "question": "For each product, show the product name and the total revenue from completed orders, but only for products that have been ordered at least once in completed orders and whose total revenue is above 5000. Round revenue to 2 decimal places.",
        "gold_sql": "SELECT p.name, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue FROM products p JOIN order_items oi ON p.id = oi.product_id JOIN orders o ON oi.order_id = o.id WHERE o.status = 'completed' GROUP BY p.id, p.name HAVING SUM(oi.quantity * oi.unit_price) > 5000 ORDER BY revenue DESC, p.id ASC",
        "difficulty": "medium",
        "tags": ["three-table-join", "aggregation", "having", "order-by"],
        "ordered": False,
    },
    {
        "id": "test_516",
        "question": "Show the names of users who have placed at least one completed order that contains more than 5 total items (sum of quantity). Return only the user names, ordered alphabetically.",
        "gold_sql": "SELECT DISTINCT u.name FROM users u JOIN orders o ON u.id = o.user_id JOIN order_items oi ON o.id = oi.order_id WHERE o.status = 'completed' GROUP BY u.id, u.name, o.id HAVING SUM(oi.quantity) > 5 ORDER BY u.name ASC",
        "difficulty": "medium",
        "tags": ["three-table-join", "aggregation", "having", "distinct"],
        "ordered": False,
    },
    {
        "id": "test_517",
        "question": "Show the product name and the number of distinct cities from which users have ordered that product, considering only completed orders. Only include products that have been.",
        "gold_sql": "SELECT p.name, COUNT(DISTINCT u.city) AS city_count FROM products p JOIN order_items oi ON p.id = oi.product_id JOIN orders o ON oi.order_id = o.id JOIN users u ON o.user_id = u.id WHERE o.status = 'completed' GROUP BY p.id, p.name HAVING COUNT(DISTINCT u.city) >= 2 ORDER BY city_count DESC, p.id ASC",
        "difficulty": "hard",
        "tags": ["four-table-join", "aggregation", "having", "distinct", "order-by"],
        "ordered": False,
    },
    {
        "id": "test_518",
        "question": "Show the user name and the total quantity of items they have ordered in completed orders, but only for users whose total quantity is greater than the average total quantity per user (computed across all users who have at least one completed order). Round the average to 2 decimal places.",
        "gold_sql": "WITH user_quant AS (SELECT u.id, u.name, SUM(oi.quantity) AS total_qty FROM users u JOIN orders o ON u.id = o.user_id JOIN order_items oi ON o.id = oi.order_id WHERE o.status = 'completed' GROUP BY u.id, u.name), avg_qty AS (SELECT AVG(total_qty) AS avg_val FROM user_quant) SELECT uq.name, uq.total_qty FROM user_quant uq CROSS JOIN avg_qty aq WHERE uq.total_qty > aq.avg_val ORDER BY uq.total_qty DESC, uq.id ASC",
        "difficulty": "hard",
        "tags": ["cte", "subquery", "aggregation", "having-alternative", "order-by"],
        "ordered": False,
    },
    {
        "id": "test_519",
        "question": "Show the names and emails of users who have never placed any order.",
        "gold_sql": "SELECT u.name, u.email FROM users u LEFT JOIN orders o ON u.id = o.user_id WHERE o.id IS NULL ORDER BY u.signup_date ASC, u.id ASC",
        "difficulty": "easy",
        "tags": ["join", "left join", "null check", "ordering"],
        "ordered": False,
    },
    {
        "id": "test_520",
        "question": "For each product, show the product name and the total quantity. Only include products that have been ordered at least once in completed orders. Show the percentage this quantity represents of the total quantity ordered across all products from completed orders, rounded to 2 decimal places.",
        "gold_sql": "SELECT p.name, SUM(oi.quantity) AS total_qty, ROUND(100.0 * SUM(oi.quantity) / (SELECT SUM(quantity) FROM order_items oi2 JOIN orders o2 ON oi2.order_id = o2.id WHERE o2.status = 'completed'), 2) AS pct FROM products p JOIN order_items oi ON p.id = oi.product_id JOIN orders o ON oi.order_id = o.id WHERE o.status = 'completed' GROUP BY p.id, p.name ORDER BY pct DESC, p.id ASC",
        "difficulty": "hard",
        "tags": ["join", "aggregation", "subquery", "percentage", "ordering"],
        "ordered": False,
    },
    {
        "id": "test_521",
        "question": "Show the names of users whose first completed order was in 2024. Return only the user names, ordered alphabetically.",
        "gold_sql": "SELECT u.name FROM users u WHERE (SELECT MIN(o.order_date) FROM orders o WHERE o.user_id = u.id AND o.status = 'completed') LIKE '2024%' ORDER BY u.name ASC",
        "difficulty": "medium",
        "tags": ["subquery", "min", "date filtering", "correlated subquery"],
        "ordered": False,
    },
    {
        "id": "test_522",
        "question": "Show the product name and the total revenue from completed orders for the product with the highest total revenue. If two products have the same highest total revenue, list the one with the smaller product id first. Return product name and total revenue rounded to 2 decimal places.",
        "gold_sql": "SELECT p.name, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue FROM products p JOIN order_items oi ON p.id = oi.product_id JOIN orders o ON oi.order_id = o.id WHERE o.status = 'completed' GROUP BY p.id, p.name ORDER BY revenue DESC, p.id ASC LIMIT 1",
        "difficulty": "medium",
        "tags": ["join", "aggregation", "limit", "ordering", "revenue"],
        "ordered": True,
    },
    {
        "id": "test_523",
        "question": "Show the names of users whose total spend from completed orders is above the average total spend per user (computed across all users who have at least one completed order). Round the average to 2 decimal places. Return user name and total spend rounded to 2 decimal places.",
        "gold_sql": "SELECT u.name, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_spend FROM users u JOIN orders o ON u.id = o.user_id JOIN order_items oi ON o.id = oi.order_id WHERE o.status = 'completed' GROUP BY u.id, u.name HAVING SUM(oi.quantity * oi.unit_price) > (SELECT AVG(spend) FROM (SELECT SUM(oi2.quantity * oi2.unit_price) AS spend FROM orders o2 JOIN order_items oi2 ON o2.id = oi2.order_id WHERE o2.status = 'completed' GROUP BY o2.user_id) t) ORDER BY total_spend DESC, u.id ASC",
        "difficulty": "hard",
        "tags": ["join", "aggregation", "subquery", "having", "average comparison"],
        "ordered": False,
    },
    {
        "id": "test_524",
        "question": "Show the names of the 3 products with the highest total quantity ordered from completed orders. If two products have the same total quantity, list the one with the smaller product id first. Return product name and total quantity.",
        "gold_sql": "SELECT p.name, SUM(oi.quantity) AS total_qty FROM products p JOIN order_items oi ON p.id = oi.product_id JOIN orders o ON oi.order_id = o.id WHERE o.status = 'completed' GROUP BY p.id, p.name ORDER BY total_qty DESC, p.id ASC LIMIT 3",
        "difficulty": "easy",
        "tags": ["join", "aggregation", "limit", "ordering"],
        "ordered": True,
    },
]


def build(source: list) -> list[dict]:
    """Fill in the default fields.

    Two kinds of task:
      check="result"     compared by result set, requires a gold_sql
      check="behavior"   compared by behaviour, no gold_sql, driven by expect

    When `ordered` is not stated, it is inferred from whether the gold has an
    ORDER BY.
    """
    out = []
    seen = set()
    for t in source:
        if t["id"] in seen:
            raise ValueError(f"duplicate task id: {t['id']}")
        seen.add(t["id"])

        check = t.get("check", "result")
        task = {
            "id": t["id"],
            "question": t["question"],
            "check": check,
            "difficulty": t["difficulty"],
            "tags": t.get("tags", []),
        }

        if check == "behavior":
            if "expect" not in t:
                raise ValueError(f"{t['id']}: behavior task needs an 'expect' dict")
            task["expect"] = t["expect"]
        else:
            sql = " ".join(t["gold_sql"].split())
            task["gold_sql"] = sql
            task["ordered"] = t.get("ordered", "ORDER BY" in sql.upper())
            # Declares that an empty gold is intended, or check_tasks calls it broken
            task["allow_empty"] = t.get("allow_empty", False)

        out.append(task)
    return out


def _describe_drift(old: list[dict], new: list[dict]) -> list[str]:
    """Say how two task sets differ, in terms a person can act on."""
    old_by_id = {t["id"]: t for t in old}
    new_by_id = {t["id"]: t for t in new}
    added = sorted(new_by_id.keys() - old_by_id.keys())
    removed = sorted(old_by_id.keys() - new_by_id.keys())
    changed = sorted(i for i in old_by_id.keys() & new_by_id.keys()
                     if old_by_id[i] != new_by_id[i])

    lines = [f"  on disk: {len(old)} tasks, would generate: {len(new)}"]
    for label, ids in (("only in the generated set", added),
                       ("only on disk", removed),
                       ("differ in content", changed)):
        if ids:
            shown = ", ".join(ids[:8]) + (" ..." if len(ids) > 8 else "")
            lines.append(f"  {label} ({len(ids)}): {shown}")
    for task_id in changed[:3]:
        for key in sorted(set(old_by_id[task_id]) | set(new_by_id[task_id])):
            a, b = old_by_id[task_id].get(key), new_by_id[task_id].get(key)
            if a != b:
                lines.append(f"    {task_id}.{key}:")
                lines.append(f"      on disk: {a!r}")
                lines.append(f"      would be: {b!r}")
    return lines


def _write_or_refuse(path: Path, tasks: list[dict], force: bool) -> bool:
    """Write the task file, unless doing so would silently change it.

    The files in tasks/ define every number in results/. Regenerating them is
    not a build step, it is a change of measuring instrument: nothing raises,
    nothing looks different afterwards, and every recorded number silently
    stops meaning what it says. So a difference has to be approved by hand.

    Returns True when the file is now what this script generates.
    """
    if path.exists() and not force:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = None
        if existing == tasks:
            print(f"unchanged  {path.name}  ({len(tasks)} tasks)")
            return True
        print(f"\nREFUSING to overwrite {path.name}: it differs from what this "
              f"script generates.")
        print("\n".join(_describe_drift(existing or [], tasks)))
        print(f"\n  The file on disk is what every number in results/ was "
              f"measured against.\n"
              f"  Overwriting it invalidates those numbers, without an error.\n"
              f"  Re-run with --force once you have decided that is what you "
              f"want.")
        return False

    path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(tasks)} tasks -> tasks/{path.name}"
          + ("  (locked: run once)" if path.name == "test.json" else ""))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="overwrite task files that differ from what this "
                             "script generates, invalidating historical results")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    # The held-out set is a separate file and enters none of the dev statistics.
    test = build(TEST_TASKS)
    ok = _write_or_refuse(OUT_DIR / "test.json", test, args.force)

    tasks = build(TASKS + GENERATED_TASKS)
    out = OUT_DIR / "dev.json"
    ok = _write_or_refuse(out, tasks, args.force) and ok

    dist: dict[str, int] = {}
    for t in tasks:
        dist[t["difficulty"]] = dist.get(t["difficulty"], 0) + 1

    print(f"\ngenerated set: {len(tasks)} tasks")
    for level in ("easy", "medium", "hard"):
        n = dist.get(level, 0)
        print(f"  {level:<8} {n:>3} {n / len(tasks):>6.0%}")
    # Behaviour tasks carry no ordered/gold_sql, so these use .get rather than
    # indexing -- otherwise adding a new kind of task raises KeyError here.
    n_behavior = sum(t["check"] == "behavior" for t in tasks)
    print(f"  check=result   {len(tasks) - n_behavior:>3}   "
          f"(ordered=True: {sum(t.get('ordered', False) for t in tasks)}, "
          f"allow_empty: {sum(t.get('allow_empty', False) for t in tasks)})")
    print(f"  check=behavior {n_behavior:>3}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
