"""Database helpers for the collector."""

import os

import psycopg


def connect() -> psycopg.Connection:
    """Open a connection using DATABASE_URL from the environment."""
    return psycopg.connect(os.environ["DATABASE_URL"])


if __name__ == "__main__":
    # Quick connection check: python -m collector.db
    with connect() as conn:
        version = conn.execute("SELECT version();").fetchone()[0]
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name;"
        ).fetchall()
    print(version)
    print("Tables:", [row[0] for row in tables])
