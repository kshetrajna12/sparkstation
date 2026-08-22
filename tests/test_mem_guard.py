"""Unit tests for the pre-launch memory-headroom guard
(supervisor/launchers/host_memory.py).

All cases mock the host probe — no ssh, no docker, no real launch.
"""
import pytest

from supervisor.launchers import host_memory
from supervisor.launchers.base import LaunchError

HOST = "worker1"


def test_meminfo_parse():
    text = (
        "MemTotal:       123456788 kB\n"
        "MemFree:         4000000 kB\n"
        "MemAvailable:   36000000 kB\n"
    )
    assert host_memory._mem_available_from_meminfo(text) == pytest.approx(34.34, abs=0.01)


def test_meminfo_parse_missing_line():
    assert host_memory._mem_available_from_meminfo("MemTotal: 100 kB\n") is None


def test_meminfo_parse_bad_number():
    assert host_memory._mem_available_from_meminfo("MemAvailable: oops kB\n") is None


def test_fresh_worker1_passes_daily_driver(monkeypatch):
    # Fresh worker1 (~117 GB MemAvailable); daily driver asks 98 GB + 10 GB
    # headroom = 108 GB required. Must not raise.
    monkeypatch.setattr(host_memory, "host_mem_available_gb", lambda host: 117.0)
    host_memory.check_memory_headroom(HOST, 98.0)


def test_packed_host_refuses_overcommit(monkeypatch):
    # worker1 already running the 98 GB daily driver (~34 GB MemAvailable):
    # a second big model must be refused with host + GB numbers in the error.
    monkeypatch.setattr(host_memory, "host_mem_available_gb", lambda host: 34.0)
    with pytest.raises(LaunchError) as excinfo:
        host_memory.check_memory_headroom(HOST, 98.0)
    msg = str(excinfo.value)
    assert HOST in msg            # names the host
    assert "34.0 GB" in msg       # reports available
    assert "108.0 GB" in msg      # reports required (98 + 10)


def test_small_model_passes_packed_host(monkeypatch):
    # bge-m3-class model (4 GB + 10 GB headroom) fits on a packed host.
    monkeypatch.setattr(host_memory, "host_mem_available_gb", lambda host: 34.0)
    host_memory.check_memory_headroom(HOST, 4.0)


def test_unmeasurable_memory_is_noop(monkeypatch):
    # Probe failure must not block launches — an unreachable host fails the
    # docker launch itself; a hard-fail here would just double the error.
    monkeypatch.setattr(host_memory, "host_mem_available_gb", lambda host: None)
    host_memory.check_memory_headroom(HOST, 98.0)


def test_none_memory_gb_is_noop(monkeypatch):
    # No declared allocation → nothing to check (would also block the probe).
    monkeypatch.setattr(host_memory, "host_mem_available_gb", lambda host: 1.0)
    host_memory.check_memory_headroom(HOST, None)
