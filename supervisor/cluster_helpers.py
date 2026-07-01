"""
Shared helpers for cluster-aware operations.

Every launcher (vllm, sglang, clip, face, species, flux) and the registry's
reconcile loop needs to talk to a specific host's Docker daemon and build URLs
that the gateway can reach. Rather than duplicate the `os.environ.copy();
env['DOCKER_HOST'] = ...` boilerplate everywhere, funnel it through these two
helpers.

For a `host` in the "primary" role (or any role with no `ip` or a loopback ip),
these are no-ops — the local Docker socket is used, and base URLs are built
against the supervisor's own bind address. Pre-cluster deployments (no
`cluster:` block in models.yaml) still work unchanged.
"""
import os
from typing import Dict

from supervisor.config import settings
from supervisor.models_config import get_cluster_config


def docker_env_for_host(host: str) -> Dict[str, str]:
    """Env-var overrides to pass to a subprocess `docker …` call for a role.

    Returns {} for local (Docker CLI uses the default unix socket) and
    {'DOCKER_HOST': 'ssh://user@ip'} for remote. Docker CLI natively supports
    the `ssh://` transport (docker/cli PR #1014, 2018) — no separate agent
    needed on the worker beyond passwordless SSH + docker group membership.
    """
    cluster = get_cluster_config()
    docker_host = cluster.docker_host_env(host)
    if docker_host is None:
        return {}
    return {"DOCKER_HOST": docker_host}


def merged_env(host: str) -> Dict[str, str]:
    """Convenience: os.environ + docker_env_for_host(host). Pass to
    subprocess.run/Popen as env=..."""
    env = os.environ.copy()
    env.update(docker_env_for_host(host))
    return env


def base_url_for_host(host: str, port: int, scheme: str = "http") -> str:
    """URL to embed in a model container's ModelInstance.base_url.

    Used by the health-check loop and the gateway to reach the model. Must be
    routable from the supervisor process, so for a remote host we return the
    cluster.hosts[host].ip (which, on our setup, is the QSFP-side address, so
    all in-cluster traffic goes over the 200 GbE direct link, not the LAN).
    """
    cluster = get_cluster_config()
    ip = cluster.host_ip_for_url(host, fallback=settings.host)
    return f"{scheme}://{ip}:{port}"
