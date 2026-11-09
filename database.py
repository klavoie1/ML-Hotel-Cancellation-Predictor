"""
SQLite database for user accounts and the prediction log.
"""
import sqlite3
from pathlib import Path

DB_FILE = Path("app.db")


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_FILE)
    con.execute("CREATE TABLE IF NOT EXISTS users ("
                "username TEXT PRIMARY KEY, salt TEXT NOT NULL, hash TEXT NOT NULL)")
    return con
