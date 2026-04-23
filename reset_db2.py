"""Full DB reset: drop all tables/triggers/indexes then recreate schema.

Safe to run while the DB file is locked (operates via a new connection).
After this, run `cce index .` for a clean full re-index.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(".cce/index.sqlite")
if not DB_PATH.exists():
    print("DB not found – nothing to reset")
    raise SystemExit(0)

con = sqlite3.connect(str(DB_PATH))
cur = con.cursor()

# 1. Drop triggers
cur.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
for (t,) in cur.fetchall():
    cur.execute(f"DROP TRIGGER IF EXISTS [{t}]")
    print(f"Dropped trigger {t}")

# 2. Drop virtual (FTS) tables first
cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND sql LIKE '%fts5%'")
for (t,) in cur.fetchall():
    if not t.endswith(("_data", "_idx", "_content", "_docsize", "_config")):
        cur.execute(f"DROP TABLE IF EXISTS [{t}]")
        print(f"Dropped FTS table {t}")

# 3. Drop remaining user tables
cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
for (t,) in cur.fetchall():
    cur.execute(f"DROP TABLE IF EXISTS [{t}]")
    print(f"Dropped table {t}")

# 4. Drop indexes
cur.execute("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'")
for (t,) in cur.fetchall():
    cur.execute(f"DROP INDEX IF EXISTS [{t}]")
    print(f"Dropped index {t}")

con.commit()
con.execute("VACUUM")
con.close()
print("\nDB reset complete. Run `cce index .` to rebuild from scratch.")
