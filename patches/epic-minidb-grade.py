#!/usr/bin/env python3
"""Hidden acceptance suite for the EPIC long-horizon mission: build minidb.

The model builds solution.py exposing `class DB` with `execute(sql) -> list`
(list of row-tuples for SELECT, or None for DDL/DML). We grade by running a
battery of SQL against a fresh DB and checking EXACT results. This file is
self-testing: run with --selftest to verify the suite passes a reference
implementation (proving the checks are correct before any model sees them).

Usage:
  epic-minidb-grade.py --selftest          # verify suite vs reference
  epic-minidb-grade.py --check <solution.py>  # grade a model's solution
"""
import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

# ── The acceptance battery: (sql, expected). expected None = no rows returned
# (DDL/DML). For SELECT, expected is a list of tuples in the required order. ──
SCRIPT = [
    ("CREATE TABLE users (id INT, name TEXT, age INT, city TEXT)", None),
    ("INSERT INTO users VALUES (1, 'Alice', 30, 'NYC')", None),
    ("INSERT INTO users VALUES (2, 'Bob', 25, 'LA')", None),
    ("INSERT INTO users VALUES (3, 'Carol', 35, 'NYC')", None),
    ("INSERT INTO users VALUES (4, 'Dave', 28, 'LA')", None),
    ("INSERT INTO users VALUES (5, 'Eve', 42, 'SF')", None),
    # basic select-all preserves insertion order
    ("SELECT id, name FROM users", [(1,'Alice'),(2,'Bob'),(3,'Carol'),(4,'Dave'),(5,'Eve')]),
    # projection subset
    ("SELECT name FROM users", [('Alice',),('Bob',),('Carol',),('Dave',),('Eve',)]),
    # WHERE equality on text
    ("SELECT name FROM users WHERE city = 'NYC'", [('Alice',),('Carol',)]),
    # WHERE numeric comparison
    ("SELECT name FROM users WHERE age > 30", [('Carol',),('Eve',)]),
    ("SELECT name FROM users WHERE age >= 30", [('Alice',),('Carol',),('Eve',)]),
    ("SELECT name FROM users WHERE age < 28", [('Bob',)]),
    ("SELECT name FROM users WHERE age != 30", [('Bob',),('Carol',),('Dave',),('Eve',)]),
    # AND / OR with precedence (AND binds tighter than OR)
    ("SELECT name FROM users WHERE city = 'LA' AND age > 26", [('Dave',)]),
    ("SELECT name FROM users WHERE city = 'SF' OR age < 26", [('Bob',),('Eve',)]),
    ("SELECT name FROM users WHERE city = 'NYC' AND age > 32 OR city = 'LA'",
     [('Bob',),('Carol',),('Dave',)]),
    # ORDER BY asc/desc
    ("SELECT name FROM users ORDER BY age", [('Bob',),('Dave',),('Alice',),('Carol',),('Eve',)]),
    ("SELECT name FROM users ORDER BY age DESC", [('Eve',),('Carol',),('Alice',),('Dave',),('Bob',)]),
    # ORDER BY + LIMIT
    ("SELECT name FROM users ORDER BY age DESC LIMIT 2", [('Eve',),('Carol',)]),
    # aggregates
    ("SELECT COUNT(*) FROM users", [(5,)]),
    ("SELECT COUNT(*) FROM users WHERE city = 'NYC'", [(2,)]),
    ("SELECT SUM(age) FROM users", [(160,)]),
    ("SELECT MIN(age) FROM users", [(25,)]),
    ("SELECT MAX(age) FROM users", [(42,)]),
    ("SELECT AVG(age) FROM users", [(32.0,)]),
    # GROUP BY with aggregate, ordered by group key for determinism
    ("SELECT city, COUNT(*) FROM users GROUP BY city ORDER BY city",
     [('LA',2),('NYC',2),('SF',1)]),
    ("SELECT city, SUM(age) FROM users GROUP BY city ORDER BY city",
     [('LA',53),('NYC',65),('SF',42)]),
    # UPDATE then observe
    ("UPDATE users SET age = 26 WHERE name = 'Bob'", None),
    ("SELECT age FROM users WHERE name = 'Bob'", [(26,)]),
    ("UPDATE users SET city = 'NYC' WHERE city = 'LA'", None),
    ("SELECT COUNT(*) FROM users WHERE city = 'NYC'", [(4,)]),
    # DELETE then observe
    ("DELETE FROM users WHERE age > 40", None),
    ("SELECT name FROM users ORDER BY name", [('Alice',),('Bob',),('Carol',),('Dave',)]),
    ("SELECT COUNT(*) FROM users", [(4,)]),
    # second table + INNER JOIN
    ("CREATE TABLE orders (oid INT, uid INT, amount INT)", None),
    ("INSERT INTO orders VALUES (10, 1, 100)", None),
    ("INSERT INTO orders VALUES (11, 1, 50)", None),
    ("INSERT INTO orders VALUES (12, 3, 75)", None),
    ("INSERT INTO orders VALUES (13, 99, 999)", None),   # no matching user
    ("SELECT users.name, orders.amount FROM users JOIN orders ON users.id = orders.uid ORDER BY orders.oid",
     [('Alice',100),('Alice',50),('Carol',75)]),
    ("SELECT COUNT(*) FROM users JOIN orders ON users.id = orders.uid", [(3,)]),
    # aggregate over a join with group-by
    ("SELECT users.name, SUM(orders.amount) FROM users JOIN orders ON users.id = orders.uid GROUP BY users.name ORDER BY users.name",
     [('Alice',150),('Carol',75)]),
]


def load_db(path):
    spec = importlib.util.spec_from_file_location("solution", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DB


def run_battery(DB):
    db = DB()
    passed = failed = 0
    fails = []
    for i, (sql, expected) in enumerate(SCRIPT):
        try:
            got = db.execute(sql)
        except Exception as e:
            failed += 1
            fails.append(f"[{i}] {sql[:60]!r} raised {type(e).__name__}: {e}")
            continue
        if expected is None:
            # DDL/DML: accept None or empty; just ensure no crash
            passed += 1
            continue
        # normalize: allow lists or tuples of rows; rows as tuples/lists
        norm = None
        try:
            norm = [tuple(r) for r in got]
        except Exception:
            fails.append(f"[{i}] {sql[:60]!r} returned non-iterable rows: {got!r}")
            failed += 1
            continue
        exp = [tuple(r) for r in expected]
        # float tolerance for AVG
        ok = len(norm) == len(exp) and all(
            all(_eq(a, b) for a, b in zip(rn, re)) for rn, re in zip(norm, exp)
        )
        if ok:
            passed += 1
        else:
            failed += 1
            fails.append(f"[{i}] {sql[:70]!r}\n     expected {exp}\n     got      {norm}")
    return passed, failed, fails


def _eq(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) < 1e-6
        except Exception:
            return a == b
    return a == b


REFERENCE = r'''
# Reference minidb — validates the acceptance suite. Not shown to the model.
import re
class DB:
    def __init__(self):
        self.tables = {}   # name -> {"cols":[...], "rows":[dict,...]}
    def execute(self, sql):
        s = sql.strip().rstrip(";")
        low = s.lower()
        if low.startswith("create table"):
            m = re.match(r"create table (\w+)\s*\((.*)\)", s, re.I|re.S)
            name = m.group(1); cols = [c.strip().split()[0] for c in m.group(2).split(",")]
            self.tables[name] = {"cols": cols, "rows": []}
            return None
        if low.startswith("insert into"):
            m = re.match(r"insert into (\w+)\s+values\s*\((.*)\)", s, re.I|re.S)
            name = m.group(1); vals = self._parse_vals(m.group(2))
            t = self.tables[name]
            self.tables[name]["rows"].append(dict(zip(t["cols"], vals)))
            return None
        if low.startswith("update"):
            m = re.match(r"update (\w+) set (\w+)\s*=\s*(.+?)(?:\s+where\s+(.*))?$", s, re.I|re.S)
            name, col, val = m.group(1), m.group(2), self._lit(m.group(3).strip())
            where = m.group(4)
            for row in self.tables[name]["rows"]:
                if where is None or self._where(where, row, name): row[col] = val
            return None
        if low.startswith("delete from"):
            m = re.match(r"delete from (\w+)(?:\s+where\s+(.*))?$", s, re.I|re.S)
            name, where = m.group(1), m.group(2)
            self.tables[name]["rows"] = [r for r in self.tables[name]["rows"]
                                         if not (where is None or self._where(where, r, name))]
            return None
        if low.startswith("select"):
            return self._select(s)
        raise ValueError("unsupported: " + s)
    def _parse_vals(self, s):
        return [self._lit(v.strip()) for v in self._split_top(s)]
    def _split_top(self, s):
        out=[]; depth=0; cur=""; q=None
        for ch in s:
            if q:
                cur+=ch
                if ch==q: q=None
            elif ch in "'\"": q=ch; cur+=ch
            elif ch=="(": depth+=1; cur+=ch
            elif ch==")": depth-=1; cur+=ch
            elif ch=="," and depth==0: out.append(cur); cur=""
            else: cur+=ch
        if cur.strip(): out.append(cur)
        return out
    def _lit(self, tok):
        tok=tok.strip()
        if tok[:1] in "'\"": return tok[1:-1]
        if re.fullmatch(r"-?\d+", tok): return int(tok)
        if re.fullmatch(r"-?\d+\.\d+", tok): return float(tok)
        return tok
    def _col(self, name, row, table):
        if "." in name:
            t,c = name.split(".")
            return row[t][c] if isinstance(row.get(t), dict) else row[c]
        return row[name]
    def _where(self, expr, row, table):
        # OR of ANDs of comparisons
        for orpart in re.split(r"\s+or\s+", expr, flags=re.I):
            if all(self._cmp(a, row, table) for a in re.split(r"\s+and\s+", orpart, flags=re.I)):
                return True
        return False
    def _cmp(self, c, row, table):
        m = re.match(r"(.+?)\s*(>=|<=|!=|=|>|<)\s*(.+)", c.strip())
        l, op, r = m.group(1).strip(), m.group(2), self._lit(m.group(3).strip())
        lv = self._col(l, row, table)
        if op=="=": return lv==r
        if op=="!=": return lv!=r
        if op==">": return lv>r
        if op=="<": return lv<r
        if op==">=": return lv>=r
        if op=="<=": return lv<=r
    def _select(self, s):
        m = re.match(r"select (.+?) from (\w+)(?:\s+join\s+(\w+)\s+on\s+(.+?))?"
                     r"(?:\s+where\s+(.+?))?(?:\s+group by\s+(.+?))?"
                     r"(?:\s+order by\s+(.+?))?(?:\s+limit\s+(\d+))?$", s, re.I|re.S)
        cols_s, t1, t2, on, where, groupby, orderby, limit = m.groups()
        if t2:
            rows=[]
            m2 = re.match(r"(.+?)\s*=\s*(.+)", on)
            la, ra = m2.group(1).strip(), m2.group(2).strip()
            for r1 in self.tables[t1]["rows"]:
                for r2 in self.tables[t2]["rows"]:
                    joined={t1:r1, t2:r2}
                    if self._col(la, joined, None)==self._col(ra, joined, None):
                        rows.append(joined)
            colctx=None
        else:
            rows=[dict(r) for r in self.tables[t1]["rows"]]; colctx=t1
        if where: rows=[r for r in rows if self._where(where, r, colctx)]
        agg = re.search(r"(count|sum|avg|min|max)\s*\(", cols_s, re.I)
        select_items=[c.strip() for c in cols_s.split(",")]
        if groupby:
            key=groupby.strip()
            groups={}
            for r in rows: groups.setdefault(self._col(key, r, colctx), []).append(r)
            out=[]
            for k,grp in groups.items():
                out.append(tuple(self._eval_item(it, grp, k, key, colctx) for it in select_items))
            if orderby: out=self._order(out, select_items, orderby)
            return out[:int(limit)] if limit else out
        if agg:
            return [tuple(self._agg1(it, rows, colctx) for it in select_items)]
        result=[tuple(self._col(c, r, colctx) for c in select_items) for r in rows]
        if orderby: result=self._order_rows(rows, select_items, orderby, colctx)
        return result[:int(limit)] if limit else result
    def _eval_item(self, it, grp, kval, key, ctx):
        if it==key or it==key.split(".")[-1]: return kval
        return self._agg(it, grp, ctx)[0] if False else self._agg1(it, grp, ctx)
    def _agg1(self, it, rows, ctx):
        m=re.match(r"(count|sum|avg|min|max)\s*\((.+?)\)", it, re.I)
        fn, arg = m.group(1).lower(), m.group(2).strip()
        if fn=="count": return len(rows)
        vals=[self._col(arg, r, ctx) for r in rows]
        if fn=="sum": return sum(vals)
        if fn=="avg": return sum(vals)/len(vals)
        if fn=="min": return min(vals)
        if fn=="max": return max(vals)
    def _agg(self, it, rows, ctx):
        return (self._agg1(it, rows, ctx),)
    def _order(self, out, items, orderby):
        key=orderby.strip(); desc=key.lower().endswith(" desc")
        key=re.sub(r"\s+(asc|desc)$","",key,flags=re.I).strip()
        idx=[i for i,it in enumerate(items) if it==key or it.split(".")[-1]==key]
        i=idx[0] if idx else 0
        return sorted(out, key=lambda r:r[i], reverse=desc)
    def _order_rows(self, rows, items, orderby, ctx):
        key=orderby.strip(); desc=key.lower().endswith(" desc")
        key=re.sub(r"\s+(asc|desc)$","",key,flags=re.I).strip()
        rows2=sorted(rows, key=lambda r:self._col(key, r, ctx), reverse=desc)
        return [tuple(self._col(c, r, ctx) for c in items) for r in rows2]
'''

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check", metavar="SOLUTION")
    args = ap.parse_args()
    if args.selftest:
        import tempfile, os
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(REFERENCE); ref = f.name
        DB = load_db(ref); os.unlink(ref)
        p, fl, fails = run_battery(DB)
        print(f"selftest: {p} passed, {fl} failed of {len(SCRIPT)}")
        for x in fails[:10]: print("  " + x)
        sys.exit(0 if fl == 0 else 1)
    if args.check:
        try:
            DB = load_db(args.check)
        except Exception as e:
            print(json.dumps({"pass": False, "score": 0, "error": f"import failed: {e}"})); sys.exit(1)
        p, fl, fails = run_battery(DB)
        total = len(SCRIPT)
        print(json.dumps({"pass": fl == 0, "score": p, "total": total,
                          "fails": fails[:8]}))
        sys.exit(0 if fl == 0 else 1)
    ap.error("need --selftest or --check")
