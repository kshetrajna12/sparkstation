"""
Pre-launch host-memory headroom check for docker-based launchers.

DGX Spark has unified CPU+GPU memory: a model container that overcommits
it does not fail fast — it OOM-kills mid weight-load, or starves the OS
and wedges the whole node (2026-08-18: a 0.88 mem-fraction-static on
worker1 hung the machine; recovery required unplugging it).
ResourceManager.allocate_model only enforces the budget for host="primary",
so remote hosts (worker1, ...) had no pre-launch check at all.

This module adds that gate: read the target host's MemAvailable (the
kernel's estimate of reclaimable RAM — what a new container would actually
compete for) and refuse to launch when the model's declared allocation plus
a fixed headroom does not fit.

Host transport mirrors the launchers' own: local roles read
/proc/meminfo directly; remote roles are reached over the same passwordless
SSH that DOCKER_HOST=ssh://... relies on (see
supervisor.cluster_helpers.docker_env_for_host). When the probe is
inconclusive (host unreachable, no ssh_user configured, ...) the check is
skipped with a warning — a host we cannot reach will fail the docker launch
itself, and a hard-fail here would add a second, less-actionable error.
"""
import logging
import subprocess

from supervisor.launchers.base import LaunchError

logger = logging.getLogger(__name__)

# Headroom (GB) that must stay free on the launch host beyond the model's
# declared allocation: OS, page cache, docker/exporter sidecars, and
# KV-cache growth beyond the static reservation. 10 GB keeps the daily
# driver (98 GB on a 121 GB worker1) launchable on a fresh node while
# refusing a second ~90 GB co-tenant.
MEM_HEADROOM_GB = 10.0

_SSH_OPTS = [
    "-o", "BatchMode=yes",          # never prompt for a password/sshpass
    "-o", "ConnectTimeout=10",
    "-o", "StrictHostKeyChecking=accept-new",
]


def _mem_available_from_meminfo(text: str) -> float | None:
    """Parse MemAvailable (kB) out of /proc/meminfo text; return GB or None."""
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1]) / (1024.0 ** 2)  # kB -> GB
                except ValueError:
                    return None
    return None


def _mem_available_local() -> float | None:
    try:
        with open("/proc/meminfo") as f:
            return _mem_available_from_meminfo(f.read())
    except OSError as e:
        logger.warning(f"Could not read /proc/meminfo: {e}")
        return None


def _mem_available_ssh(user: str, ip: str) -> float | None:
    """MemAvailable (GB) on a remote host, or None when unreachable."""
    try:
        result = subprocess.run(
            ["ssh", *_SSH_OPTS, f"{user}@{ip}", "cat /proc/meminfo"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            logger.warning(
                f"ssh meminfo probe failed for {user}@{ip}: "
                f"rc={result.returncode} {result.stderr.strip()[:200]}"
            )
            return None
        return _mem_available_from_meminfo(result.stdout)
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning(f"ssh meminfo probe failed for {user}@{ip}: {e}")
        return None


def host_mem_available_gb(host: str) -> float | None:
    """MemAvailable (GB) for a cluster role, or None when unmeasurable.

    Resolution matches docker_env_for_host(): an unknown role, a role with
    no ip, or a loopback ip is local; everything else goes over ssh with
    the role's ssh_user.
    """
    try:
        from supervisor.models_config import get_cluster_config
        entry = get_cluster_config().hosts.get(host)
    except Exception as e:
        logger.warning(f"Could not resolve cluster config for host '{host}': {e}")
        entry = None

    if entry is None or entry.ip is None or entry.ip in ("127.0.0.1", "localhost", "::1"):
        return _mem_available_local()
    if not entry.ssh_user:
        logger.warning(f"Cluster host '{host}' has no ssh_user — cannot probe memory")
        return None
    return _mem_available_ssh(entry.ssh_user, entry.ip)


def check_memory_headroom(host: str, memory_gb: float | None) -> None:
    """Refuse to launch when the host cannot spare memory_gb + headroom.

    Raises LaunchError (which the callers' existing paths turn into a
    FAILED model / ModelLaunchError) with the host name and both GB
    numbers. No-op when memory_gb is unset or the probe is inconclusive —
    this is a safety backstop in front of docker run, not a scheduler.
    """
    if memory_gb is None:
        return
    available = host_mem_available_gb(host)
    if available is None:
        logger.warning(
            f"Host '{host}': MemAvailable unmeasurable — skipping headroom "
            f"check (launch proceeds; cgroup cap / OS OOM remain the backstop)"
        )
        return
    required = memory_gb + MEM_HEADROOM_GB
    if available < required:
        raise LaunchError(
            f"Host '{host}' has {available:.1f} GB available but this model "
            f"requires {memory_gb:.1f} GB + {MEM_HEADROOM_GB:.0f} GB headroom "
            f"({required:.1f} GB total). Refusing launch to avoid "
            f"overcommitting unified memory (node-wedge risk). Free memory "
            f"on '{host}' and retry."
        )
    logger.info(
        f"Memory headroom OK on host '{host}': {available:.1f} GB available, "
        f"{required:.1f} GB required ({memory_gb:.1f} GB model + "
        f"{MEM_HEADROOM_GB:.0f} GB headroom)"
    )
