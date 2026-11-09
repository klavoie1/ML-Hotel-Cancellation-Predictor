"""
User accounts with salted, hashed passwords (SHA256), stored in SQLite.

Passwords are never stored. Only a random salt and the hash are saved.

Add or change a user from the command line:
    python security.py add <username>
"""
import getpass
import hashlib
import hmac
import os
import sys

import database

ITERATIONS = 200_000

# Hashed and Salted password to keep it safer than storing just plain password or just its hash.
def hash_password(password: str, salt: bytes | None = None) -> dict:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return {"salt": salt.hex(), "hash": digest.hex()}


def verify_password(password: str, record: dict) -> bool:
    expected = hash_password(password, bytes.fromhex(record["salt"]))["hash"]
    return hmac.compare_digest(expected, record["hash"])

def check_login(username: str, password: str) -> bool:
    with database.connect() as con:
        row = con.execute("SELECT salt, hash FROM users WHERE username = ?",
                          (username,)).fetchone()
    return row is not None and verify_password(password, {"salt": row[0], "hash": row[1]})


# shouldn't be needed for the submission but in a real development adding users would be important.
def add_user(username: str, password: str) -> None:
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    record = hash_password(password)
    with database.connect() as con:
        con.execute("INSERT OR REPLACE INTO users (username, salt, hash) VALUES (?, ?, ?)",
                    (username, record["salt"], record["hash"]))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "add":
        add_user(sys.argv[2], getpass.getpass("New password: "))
        print(f"Saved user '{sys.argv[2]}'.")
    else:
        print("Usage: python security.py add <username>")
