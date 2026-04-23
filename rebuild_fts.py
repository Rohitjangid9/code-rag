"""Rebuild FTS5 shadow tables after manual data deletion."""
import sqlite3

con = sqlite3.connect(".cce/index.sqlite")
for tbl in ["lex_fts", "lex_sym_fts", "symbols_fts"]:
    try:
        con.execute(f"INSERT INTO {tbl}({tbl}) VALUES('rebuild')")
        print(f"Rebuilt {tbl}")
    except Exception as e:
        print(f"{tbl}: {e}")
con.commit()
con.close()
print("done")
