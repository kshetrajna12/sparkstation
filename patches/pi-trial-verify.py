#!/usr/bin/env python3
"""Plant/verify helpers for the pi-agentic trial arms (RealtorZero clones).

Modes:
  plant-a  <clone>   plant the None-beds coercion regression into comps.py
  verify-a <clone>   end-to-end: build a real sqlite fixture, run find_comps,
                     assert Maricopa (beds=None) comps pass bed filters again
  verify-b <clone>   verify median_ppsf() added to tools/comps.py + a test file

All imports are done against the CLONE with realtorzero.config stubbed to a
temp DATA_DIR — nothing touches the real repo or real data.
"""
import asyncio
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

PLANT_OLD = """        # NULL beds means the county doesn't publish bed counts (e.g. Maricopa) —
        # unknown must pass the filter, not be coerced to a failing value.
        if r["beds"] is not None:
            if min_beds is not None and r["beds"] < min_beds:
                continue
            if max_beds is not None and r["beds"] > max_beds:
                continue"""
PLANT_NEW = """        beds = r["beds"] if r["beds"] is not None else 0
        if min_beds is not None and beds < min_beds:
            continue
        if max_beds is not None and beds > max_beds:
            continue"""


def load_comps(clone: Path, data_dir: Path):
    """Import <clone>/src/realtorzero/tools/comps.py with config stubbed.

    The realtorzero/realtorzero.tools packages get REAL __path__ entries into
    the clone so any sibling modules the agent created (e.g. a geo_math.py
    extracted during a refactor mission) import normally — only config is
    stubbed. Without this, a correct multi-file refactor crashed the grader
    (caught live 2026-08-17 during the dsv4 escalation arm).
    """
    for name in list(sys.modules):
        if name == "realtorzero" or name.startswith("realtorzero."):
            del sys.modules[name]
    pkg = types.ModuleType("realtorzero")
    pkg.__path__ = [str(clone / "src/realtorzero")]
    tools = types.ModuleType("realtorzero.tools")
    tools.__path__ = [str(clone / "src/realtorzero/tools")]
    cfg = types.ModuleType("realtorzero.config")
    cfg.DATA_DIR = data_dir
    cfg.HTTP_USER_AGENT = "trial"
    sys.modules["realtorzero"] = pkg
    sys.modules["realtorzero.tools"] = tools
    sys.modules["realtorzero.config"] = cfg
    path = clone / "src/realtorzero/tools/comps.py"
    spec = importlib.util.spec_from_file_location("realtorzero.tools.comps", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fixture_rows():
    # around (33.45, -112.07); one Maricopa-style row with beds=NULL
    return [
        (1, "10 A St", 33.4501, -112.0701, "2026-05-01", 500000, None, 2.0, 1800, 6000, 1998, "sfr", "maricopa"),
        (2, "20 B St", 33.4510, -112.0710, "2026-04-01", 450000, 2, 1.0, 1200, 5000, 1980, "sfr", "kc"),
        (3, "30 C St", 33.4490, -112.0690, "2026-03-01", 650000, 4, 3.0, 2400, 7000, 2005, "sfr", "kc"),
        (4, "40 D St", 34.4500, -112.0700, "2026-05-01", 700000, 3, 2.0, 2000, 6000, 2001, "sfr", "kc"),
    ]


def build_db(mod):
    conn = mod.get_db()
    conn.executemany(
        "INSERT INTO sales VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], "x")[:13]
         for r in [row + () for row in fixture_rows()]],
    )
    conn.commit()
    conn.close()


def verify_a(clone: Path) -> dict:
    with tempfile.TemporaryDirectory() as td:
        mod = load_comps(clone, Path(td))
        build_db(mod)
        out = asyncio.run(mod.find_comps(33.45, -112.07, radius_miles=5.0,
                                         since="2026-01-01", min_beds=3))
        comps = out.get("comps", [])
        ids = {c.get("id") for c in comps}
        checks = {
            "null_beds_passes_filter": 1 in ids,
            "low_beds_excluded": 2 not in ids,
            "matching_included": 3 in ids,
            "out_of_radius_excluded": 4 not in ids,
        }
        return {"pass": all(checks.values()), "checks": checks}


def verify_b(clone: Path) -> dict:
    with tempfile.TemporaryDirectory() as td:
        mod = load_comps(clone, Path(td))
        checks = {"function_exists": hasattr(mod, "median_ppsf")}
        if checks["function_exists"]:
            f = mod.median_ppsf
            try:
                # ppsf values 200/300/400 -> median unambiguously 300.
                # (Original check used values [200,300,200] but asserted 300 —
                # a GRADER bug that failed correct implementations; caught
                # live on the dsv4 arm 2026-08-17 22:14.)
                checks["basic"] = abs(f([
                    {"sale_price": 400000, "sqft": 2000},
                    {"sale_price": 330000, "sqft": 1100},
                    {"sale_price": 1000000, "sqft": 2500},
                ]) - 300.0) < 1e-6
                checks["ignores_bad_sqft"] = abs(f([
                    {"sale_price": 400000, "sqft": 2000},
                    {"sale_price": 999999, "sqft": 0},
                    {"sale_price": 999999, "sqft": None},
                ]) - 200.0) < 1e-6
            except Exception as e:
                checks["raised"] = str(e)[:120]
        tests = list(clone.rglob("test*media*ppsf*")) + list(clone.rglob("test*comps*"))
        checks["test_file_added"] = any(
            t for t in tests if t.suffix == ".py" and "median_ppsf" in t.read_text(errors="ignore")
        )
        return {"pass": all(v is True for k, v in checks.items() if k != "raised")
                and checks.get("function_exists", False),
                "checks": {k: (v if isinstance(v, bool) else v) for k, v in checks.items()}}


if __name__ == "__main__":
    mode, clone = sys.argv[1], Path(sys.argv[2])
    if mode == "plant-a":
        p = clone / "src/realtorzero/tools/comps.py"
        s = p.read_text()
        assert PLANT_OLD in s, "plant anchor not found — comps.py changed upstream"
        p.write_text(s.replace(PLANT_OLD, PLANT_NEW))
        print(json.dumps({"planted": True}))
    elif mode == "verify-a":
        print(json.dumps(verify_a(clone)))
    elif mode == "verify-b":
        print(json.dumps(verify_b(clone)))
    elif mode == "verify-c":
        import re
        notes = clone / "MARKET_NOTES.md"
        checks = {"file_exists": notes.exists()}
        if notes.exists():
            txt = notes.read_text(errors="ignore")
            m = re.search(r"([2-9]\.\d{1,3})\s*%", txt)
            checks["plausible_rate"] = bool(m)
            checks["source_url"] = "http" in txt
            checks["mentions_source"] = bool(re.search(r"freddie|pmms|mortgage", txt, re.I))
        print(json.dumps({"pass": all(v for v in checks.values()), "checks": checks}))
    else:
        raise SystemExit(f"unknown mode {mode}")
