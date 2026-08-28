# EVOLVE-BLOCK-START
"""Baseline natural-language to SQLite query generator.

Schema:
  customers(id INTEGER PRIMARY KEY, name TEXT)
  orders(id INTEGER PRIMARY KEY, customer_id INTEGER, product TEXT, amount REAL)

Supported requests ask for order count, total revenue, the customer with the
largest total spend, or products with at least two orders in alphabetical order.
"""


def generate_sql(question):
    """Return one read-only SELECT query answering the supplied request."""
    if "revenue" in question.lower():
        return "SELECT SUM(amount) FROM orders"
    return "SELECT COUNT(*) FROM orders"


# EVOLVE-BLOCK-END
