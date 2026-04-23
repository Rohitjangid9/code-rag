"""Reset index DB in-place (safe on Windows where file-delete is blocked).

Wipes all tables so the next `cce index .` performs a full re-index.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(".cce/index.sqlite")
if not DB_PATH.exists():
    print("DB not found – nothing to reset")
    raise SystemExit(0)

con = sqlite3.connect(str(DB_PATH))
cur = con.cursor()

# Find all user tables (excludes FTS shadow tables automatically handled by DROP)
cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
tables = [r[0] for r in cur.fetchall()]
print(f"Tables found: {tables}")

for t in tables:
    try:
        cur.execute(f"DELETE FROM [{t}]")
        print(f"  Cleared {t}")
    except Exception as e:
        print(f"  Skip {t}: {e}")

con.commit()
con.execute("VACUUM")
con.close()
print("DB reset complete – run `cce index .` to rebuild from scratch.")
