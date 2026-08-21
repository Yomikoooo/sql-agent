"""Generate the SQLite database the agent queries (an e-commerce order shop).

The random seed is fixed, so every run produces byte-identical data. That
matters more than it looks: the gold answers in the eval set are tied to these
specific rows, and regenerating with different data invalidates all of them.

Usage:
    python seed_db.py
"""
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from . import paths

random.seed(42)                      # fixed seed: reproducible data

DB_PATH = paths.DB_PATH

SCHEMA = """
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS users;

CREATE TABLE users (
    id          INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    email       TEXT    NOT NULL,
    city        TEXT    NOT NULL,
    signup_date TEXT    NOT NULL      -- 'YYYY-MM-DD'
);

CREATE TABLE products (
    id       INTEGER PRIMARY KEY,
    name     TEXT    NOT NULL,
    category TEXT    NOT NULL,
    price    REAL    NOT NULL
);

CREATE TABLE orders (
    id         INTEGER PRIMARY KEY,
    user_id    INTEGER NOT NULL,
    order_date TEXT    NOT NULL,
    status     TEXT    NOT NULL,      -- 'completed' | 'pending' | 'cancelled'
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE order_items (
    id         INTEGER PRIMARY KEY,
    order_id   INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity   INTEGER NOT NULL,
    unit_price REAL    NOT NULL,      -- price at order time, may differ from current
    FOREIGN KEY (order_id)   REFERENCES orders(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);
"""

CITIES = ["Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Hangzhou", "Chengdu"]
FIRST = ["Wei", "Fang", "Min", "Jing", "Lei", "Yan", "Tao", "Hui", "Qiang", "Xia"]
LAST = ["Wang", "Li", "Zhang", "Liu", "Chen", "Yang", "Huang", "Zhao", "Wu", "Zhou"]

PRODUCTS = [
    ("Laptop Pro 14",      "Electronics", 9999.0),
    ("Laptop Air 13",      "Electronics", 6499.0),
    ("Smartphone X",       "Electronics", 4999.0),
    ("Wireless Earbuds",   "Electronics",  899.0),
    ("Mechanical Keyboard","Electronics",  599.0),
    ("USB-C Hub",          "Electronics",  249.0),
    ("Office Chair",       "Furniture",   1299.0),
    ("Standing Desk",      "Furniture",   2499.0),
    ("Desk Lamp",          "Furniture",    199.0),
    ("Bookshelf",          "Furniture",    799.0),
    ("Coffee Beans 1kg",   "Grocery",      129.0),
    ("Green Tea 500g",     "Grocery",       89.0),
    ("Protein Powder",     "Grocery",      299.0),
    ("Running Shoes",      "Sports",       699.0),
    ("Yoga Mat",           "Sports",       159.0),
    ("Dumbbell Set",       "Sports",       459.0),
    ("Novel: The Path",    "Books",         59.0),
    ("Python Cookbook",    "Books",        128.0),
    ("Design Patterns",    "Books",        108.0),
    ("Notebook A5",        "Books",         25.0),
]

STATUSES = ["completed"] * 7 + ["pending"] * 2 + ["cancelled"] * 1   # 70/20/10


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)

    # ---- users ----
    users = []
    start = date(2024, 1, 1)
    for uid in range(1, 201):                       # 200 users
        name = f"{random.choice(LAST)} {random.choice(FIRST)}"
        email = f"user{uid}@example.com"
        city = random.choice(CITIES)
        signup = start + timedelta(days=random.randint(0, 500))
        users.append((uid, name, email, city, signup.isoformat()))
    conn.executemany("INSERT INTO users VALUES (?,?,?,?,?)", users)

    # ---- products ----
    products = [(pid, n, c, p) for pid, (n, c, p) in enumerate(PRODUCTS, 1)]
    conn.executemany("INSERT INTO products VALUES (?,?,?,?)", products)

    # ---- orders + order_items ----
    orders, items = [], []
    item_id = 1
    for oid in range(1, 1001):                      # 1000 orders
        uid = random.randint(1, 200)
        # an order cannot predate the signup
        signup = date.fromisoformat(users[uid - 1][4])
        max_days = (date(2025, 12, 31) - signup).days
        odate = signup + timedelta(days=random.randint(0, max(max_days, 1)))
        orders.append((oid, uid, odate.isoformat(), random.choice(STATUSES)))

        for _ in range(random.randint(1, 4)):       # 1-4 products per order
            pid, _, _, price = random.choice(products)
            qty = random.randint(1, 3)
            # occasional discount, so unit_price differs from products.price
            unit = round(price * random.choice([1.0, 1.0, 1.0, 0.9, 0.8]), 2)
            items.append((item_id, oid, pid, qty, unit))
            item_id += 1

    conn.executemany("INSERT INTO orders VALUES (?,?,?,?)", orders)
    conn.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", items)
    conn.commit()

    print(f"database written: {DB_PATH}")
    for table in ("users", "products", "orders", "order_items"):
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:<12} {n:>6,} rows")
    conn.close()


if __name__ == "__main__":
    main()
