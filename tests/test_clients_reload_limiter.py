"""clients.yaml hot reload must not strand admission capacity.

The proxy resolves a client once per request and releases on that same object after a reload may have replaced it. These tests drive
the real Registry over a fictional YAML file (fake keys; reload by an explicit mtime advance, no sleep) under loud guards: any open
outside the test's directory or any socket connect fails the test. No gateway import, network, real config or key.

Run only this file:  .venv/bin/python -m pytest -p no:cacheprovider tests/test_clients_reload_limiter.py
"""
import builtins
import contextlib
import io
import os
import socket
import time
from pathlib import Path

import pytest

from gateway.clients import DENY_CONCURRENCY, DENY_RATE_LIMITED, Registry

OWNER, LAB = "sk-fictional-owner-0001", "sk-fictional-lab-0002"


class Loud(BaseException):
    pass


@contextlib.contextmanager
def guarded(directory: Path):
    real_open, real_io_open, real_os_open, real_connect = builtins.open, io.open, os.open, socket.socket.connect
    root = directory.resolve()

    def allowed(path):
        if isinstance(path, int):
            return False
        p = Path(os.fspath(path)).resolve()
        return p == root or root in p.parents

    def g_open(file, *a, **k):
        if not allowed(file):
            raise Loud(f"open outside the test directory: {file}")
        return real_open(file, *a, **k)

    def g_os_open(path, *a, **k):
        if not allowed(path):
            raise Loud(f"open outside the test directory: {path}")
        return real_os_open(path, *a, **k)

    def g_connect(self, *a, **k):
        raise Loud("socket connect attempted")

    builtins.open, io.open, os.open, socket.socket.connect = g_open, g_open, g_os_open, g_connect
    try:
        yield
    finally:
        builtins.open, io.open, os.open, socket.socket.connect = real_open, real_io_open, real_os_open, real_connect


class Clients:
    """A fictional clients.yaml whose reloads are driven by explicit mtime advances."""

    def __init__(self, directory: Path, owner_concurrency=1, owner_rpm=0, lab_allow=("default",), owner_keys=(OWNER,)):
        self.path, self.mtime = directory / "clients.yaml", 1_800_000_000.0
        self.write(owner_concurrency, owner_rpm, lab_allow, owner_keys)

    def write(self, owner_concurrency=1, owner_rpm=0, lab_allow=("default",), owner_keys=(OWNER,), raw=None):
        keys = ", ".join(f'"{k}"' for k in owner_keys)
        allow = "\n".join(f"      - {a}" for a in lab_allow)
        self.path.write_text(raw if raw is not None else (
            "enforce_auth: true\nclients:\n"
            f"  - name: sparky-worker-owner\n    keys: [{keys}]\n    allow:\n      - default\n"
            f"    rpm: {owner_rpm}\n    concurrency: {owner_concurrency}\n"
            f"  - name: sparky-worker-lab\n    keys: [\"{LAB}\"]\n    allow:\n{allow}\n    rpm: 24\n    concurrency: 1\n"))
        self.mtime += 10
        os.utime(self.path, (self.mtime, self.mtime))


def admit(client):
    ok, reason, _ = client.admit("default", time.monotonic())
    return ok, reason


@pytest.fixture
def env(tmp_path):
    with guarded(tmp_path):
        yield tmp_path


def test_unchanged_owner_release_after_lab_only_reload_frees_the_slot(env):
    f = Clients(env, owner_concurrency=1)
    reg = Registry(str(f.path))
    lease = reg.resolve(OWNER)
    assert admit(lease) == (True, None)
    f.write(owner_concurrency=1, lab_allow=("default", "reflex"))  # only LAB's allow list changes
    reg.maybe_reload()
    lease.release()  # the request ends on the object it was admitted with
    current = reg.resolve(OWNER)
    assert current.inflight == 0
    assert admit(current) == (True, None)


def test_changed_lab_release_after_its_own_allow_edit_frees_the_slot(env):
    f = Clients(env)
    reg = Registry(str(f.path))
    lease = reg.resolve(LAB)
    assert admit(lease) == (True, None)
    f.write(lab_allow=("default", "reflex"))
    reg.maybe_reload()
    current = reg.resolve(LAB)
    assert current.allows_model("reflex") and not lease.allows_model("reflex")  # the new policy is in force
    assert admit(current) == (False, DENY_CONCURRENCY)  # the lease is still in flight: concurrency 1 holds
    lease.release()
    assert current.inflight == 0
    assert admit(current) == (True, None)


def test_overlapping_leases_across_reloads_keep_exact_capacity(env):
    f = Clients(env, owner_concurrency=2)
    reg = Registry(str(f.path))
    first = reg.resolve(OWNER)
    assert admit(first) == (True, None)
    f.write(owner_concurrency=2, lab_allow=("default", "reflex"))
    reg.maybe_reload()
    second = reg.resolve(OWNER)
    assert admit(second) == (True, None)
    assert admit(second) == (False, DENY_CONCURRENCY)  # exactly two in flight across both objects
    f.write(owner_concurrency=2, lab_allow=("default",))
    reg.maybe_reload()
    first.release()
    second.release()
    current = reg.resolve(OWNER)
    assert current.inflight == 0
    assert admit(current) == (True, None) and admit(current) == (True, None)
    assert admit(current) == (False, DENY_CONCURRENCY)


def test_rpm_window_is_preserved_across_reload(env):
    f = Clients(env, owner_concurrency=0, owner_rpm=2)
    reg = Registry(str(f.path))
    c = reg.resolve(OWNER)
    assert admit(c) == (True, None) and admit(c) == (True, None)
    c.release(); c.release()
    f.write(owner_concurrency=0, owner_rpm=2, lab_allow=("default", "reflex"))
    reg.maybe_reload()
    assert admit(reg.resolve(OWNER)) == (False, DENY_RATE_LIMITED)


def test_policy_and_key_updates_are_fresh_snapshots_sharing_the_lease_count(env):
    f = Clients(env, owner_concurrency=1)
    reg = Registry(str(f.path))
    lease = reg.resolve(OWNER)
    assert admit(lease) == (True, None)
    f.write(owner_concurrency=2, owner_keys=(OWNER, "sk-fictional-owner-0003"))
    reg.maybe_reload()
    current = reg.resolve("sk-fictional-owner-0003")
    assert current is reg.resolve(OWNER)
    assert (lease.concurrency, current.concurrency) == (1, 2)  # each object keeps its own policy snapshot
    assert current.inflight == 1  # but the in-flight count is the client's
    assert admit(current) == (True, None)
    assert admit(current) == (False, DENY_CONCURRENCY)
    lease.release()
    assert current.inflight == 1


def test_parse_failure_keeps_the_previous_registry_and_its_count(env):
    f = Clients(env)
    reg = Registry(str(f.path))
    lease = reg.resolve(OWNER)
    assert admit(lease) == (True, None)
    f.write(raw="clients: [unclosed\n")
    reg.maybe_reload()
    assert reg.resolve(OWNER) is lease
    lease.release()
    assert admit(reg.resolve(OWNER)) == (True, None)


def test_guard_refuses_a_path_outside_the_test_directory(env):
    with pytest.raises(Loud):
        Registry("/etc/hostname")
