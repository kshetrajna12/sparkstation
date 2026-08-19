#!/usr/bin/env bash
# LONG-HORIZON epic mission through pi (real env, through the gateway per the
# always-through-sparkstation rule). The model builds a mini in-memory SQL
# database (minidb) over a multi-hour session; graded by a hidden acceptance
# suite (patches/epic-minidb-grade.py, self-validated 42/42 vs reference).
#
# Usage: epic-long-horizon-arm.sh [arm-label]   (default: whatever pi's
# default provider currently serves — the daily driver via the gateway).
set -uo pipefail
REPO="$HOME/src/github.com/sparkstation"
ARM="${1:-daily-driver}"
OUT="$HOME/.sparkstation/quant-ab/epic-$(date +%Y%m%d)"
LOG="$OUT/epic.log"; RESULTS="$OUT/results.jsonl"
PI_TIMEOUT="${PI_TIMEOUT:-14400}"   # 4h ceiling for the whole multi-turn build
mkdir -p "$OUT"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
cd "$REPO"

WS="$OUT/$ARM"; rm -rf "$WS"; mkdir -p "$WS"

PROMPT=$(cat <<'EOP'
You are building a small in-memory SQL database engine, entirely from scratch, in Python (standard library only — no sqlite, no external packages). Work ONLY in the current directory. Your deliverable is a single self-contained file named solution.py.

solution.py must define `class DB` with one public method `execute(self, sql: str)`. It runs one SQL statement and returns:
  - for SELECT: a list of row tuples, in the correct order
  - for CREATE TABLE / INSERT / UPDATE / DELETE: None

The SQL dialect to support (case-insensitive keywords; identifiers are case-sensitive):
  - CREATE TABLE name (col TYPE, col TYPE, ...)   — TYPE is INT | TEXT (you may ignore the type beyond parsing)
  - INSERT INTO name VALUES (v1, v2, ...)         — integer, float, and single-quoted text literals
  - SELECT <cols> FROM name [JOIN name2 ON a.x = b.y] [WHERE ...] [GROUP BY col] [ORDER BY col [ASC|DESC]] [LIMIT n]
      <cols> is a comma list of column names (possibly table-qualified like users.name) OR aggregate calls
      Aggregates: COUNT(*), SUM(col), AVG(col), MIN(col), MAX(col)
      WHERE supports comparisons =, !=, <, >, <=, >= combined with AND / OR (AND binds tighter than OR)
      JOIN is INNER JOIN only; qualified columns (table.col) resolve against the joined row
      GROUP BY groups rows by a column; the select list is the group key plus aggregate(s)
      AVG returns a float; SUM/MIN/MAX/COUNT return as appropriate
  - UPDATE name SET col = value [WHERE ...]
  - DELETE FROM name [WHERE ...]

Rows returned by SELECT preserve insertion order unless ORDER BY is given. Text is compared as strings, numbers numerically.

This is a substantial, multi-part build. Decompose it: get the tokenizer/parser and a storage model right first, then SELECT with projection and WHERE, then ORDER/LIMIT, then aggregates and GROUP BY, then JOIN, then UPDATE/DELETE. WRITE YOUR OWN TESTS as you go — create test_*.py files with cases for each feature and run them to verify before moving on. You are graded by a hidden test suite that exercises every feature above against a fresh DB, checking exact results, so correctness and edge-case handling matter more than speed. Take the time to get it right; verify thoroughly before you finish.
EOP
)

log "=== EPIC long-horizon arm: $ARM (timeout ${PI_TIMEOUT}s) ==="
log "workspace: $WS"
t0=$(date +%s)
( cd "$WS" && timeout "$PI_TIMEOUT" pi -p "$PROMPT" > "$OUT/$ARM-pi.txt" 2>&1 )
rc=$?; t1=$(date +%s)
wall=$((t1-t0))
log "pi finished: rc=$rc wall=${wall}s ($(($wall/60))min)"

if [ -f "$WS/solution.py" ]; then
  v=$(python3 "$REPO/patches/epic-minidb-grade.py" --check "$WS/solution.py" 2>&1)
  gr=$?
else
  v='{"pass": false, "score": 0, "error": "no solution.py produced"}'; gr=1
fi
echo "{\"arm\":\"$ARM\",\"mission\":\"epic-minidb\",\"wall_s\":$wall,\"pi_rc\":$rc,\"grade\":$v}" >> "$RESULTS"
log "GRADE[$ARM]: $v"
log "=== EPIC arm $ARM complete ==="
