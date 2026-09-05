"""
Load and parse models.yaml (+ optional .sparkstation.local.yaml overlay).

Structural refactor (iter-6, 2026-06-30):
  - Models are defined ONCE in `models:` (dict keyed by alias) instead of being
    copy-pasted across `autoload:` and every profile.
  - Profiles are `dict[alias -> override_dict]`. Empty `{}` means "use base
    spec as-is"; any keys set override the base at resolve time. This lets
    different profiles put the same model on different hosts, allocate
    different memory, etc., without duplicating the whole entry.
  - Cluster host details (IPs, ssh users, hostnames) live in a gitignored
    `.sparkstation.local.yaml` that deep-merges over the public models.yaml,
    so the checked-in file stays free of machine-specific info.

Loader precedence: public `models.yaml` (base) → `.sparkstation.local.yaml`
(overrides). Both files are optional; either or both can be absent.
"""
import logging
from pathlib import Path
from typing import Set, List, Dict, Any, Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


LOCAL_CONFIG_FILENAME = ".sparkstation.local.yaml"
ROOT_BREADCRUMB = Path.home() / ".sparkstation" / "root"


def find_sparkstation_root() -> Path:
    """Locate the sparkstation repo root from ANY working directory.

    Order: walk up from cwd looking for models.yaml (works inside the repo);
    $SPARKSTATION_HOME; the ~/.sparkstation/root breadcrumb (written by the
    CLI whenever it runs from the real repo). Falls back to cwd. Without
    this, `sparkstation init` in another project silently loaded an EMPTY
    config and generated a CLAUDE.md with no models (2026-08-15).
    """
    import os
    cwd = Path.cwd()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "models.yaml").exists():
            return candidate
    env = os.environ.get("SPARKSTATION_HOME")
    if env and (Path(env) / "models.yaml").exists():
        return Path(env)
    try:
        crumb = Path(ROOT_BREADCRUMB.read_text().strip())
        if (crumb / "models.yaml").exists():
            return crumb
    except OSError:
        pass
    return cwd


def _deep_merge(base: dict, over: dict) -> dict:
    """Recursive dict merge; `over` wins for scalars, dicts merge, lists replace."""
    result = dict(base)
    for k, v in (over or {}).items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


class ClusterHost(BaseModel):
    """One node in the cluster, referenced by a role name.

    Role names (`primary`, `worker1`, ...) are logical labels used by every
    file in the repo. Real IPs / SSH users / hostnames go in the gitignored
    `.sparkstation.local.yaml` so the public repo stays free of topology.

    - `ip: None` (default) means "the local machine" — Docker talks to the
      unix socket, no ssh transport.
    - Non-local IPs require `ssh_user`; the launcher builds
      `DOCKER_HOST=ssh://<ssh_user>@<ip>` and reuses passwordless SSH keys.
    """
    ip: Optional[str] = None
    ssh_user: Optional[str] = None
    label: Optional[str] = None  # human-friendly name for logs
    # Path to HF cache on this host, defaults to ~/.cache/huggingface. Set
    # explicitly if you moved the cache elsewhere (e.g. NFS mount).
    hf_cache_path: Optional[str] = None


class ClusterConfig(BaseModel):
    """Top-level cluster block. Default is single-host (`primary` only)."""
    hosts: Dict[str, ClusterHost] = Field(
        default_factory=lambda: {"primary": ClusterHost()}
    )

    def is_local(self, host: str) -> bool:
        entry = self.hosts.get(host)
        if entry is None or entry.ip is None:
            return True
        return entry.ip in ("127.0.0.1", "localhost", "::1")

    def docker_host_env(self, host: str) -> Optional[str]:
        entry = self.hosts.get(host)
        if entry is None or entry.ip is None:
            return None
        if entry.ip in ("127.0.0.1", "localhost", "::1"):
            return None
        if not entry.ssh_user:
            raise ValueError(
                f"cluster host '{host}' has ip={entry.ip!r} but no ssh_user — "
                f"set cluster.hosts.{host}.ssh_user in {LOCAL_CONFIG_FILENAME}"
            )
        return f"ssh://{entry.ssh_user}@{entry.ip}"

    def host_ip_for_url(self, host: str, fallback: str = "127.0.0.1") -> str:
        entry = self.hosts.get(host)
        if entry is None or entry.ip is None:
            return fallback
        return entry.ip


class ModelDefinition(BaseModel):
    """Profile-independent BASE spec for a model.

    Everything a launcher might need lives here EXCEPT the alias (which is
    the dict key in `models:`) and profile-specific overrides (which live in
    each `profiles.<name>.<alias>` dict). Profile overrides deep-merge over
    this at resolve time.
    """
    name: str  # HuggingFace model name
    backend: str
    model_type: str = "chat"
    host: str = "primary"  # default; profiles typically override
    quantization: Optional[str] = "fp8"
    memory_gb: Optional[float] = None
    idle_timeout_minutes: int = 30
    auto_suspend_enabled: bool = False
    speculative_model: Optional[str] = None
    num_speculative_tokens: int = 5
    speculative_method: Optional[str] = None
    speculative_extra: Dict[str, Any] = Field(default_factory=dict)
    default: bool = False
    # CAPABILITY: does this model accept image input at all? Independent of
    # vision_default — a profile has exactly ONE vision_default (what the
    # "vision" alias resolves to) but may run SEVERAL vision-capable models
    # (2026-09-05: under `voice`, both qwen-flash-next and gemma4-2b see).
    # Published to the gateway as model_info.supports_vision so clients can
    # discover capability instead of hardcoding which alias is multimodal.
    vision: bool = False
    vision_default: bool = False  # profile's designated vision model → gateway "vision" alias
    extra_args: Dict[str, Any] = Field(default_factory=dict)
    docker_image: Optional[str] = None
    env_vars: Dict[str, str] = Field(default_factory=dict)
    volumes: List[str] = Field(default_factory=list)


class ModelConfigYAML(BaseModel):
    """Resolved model config (base + profile override + alias merged).

    This is what launchers and main.py autoload iterate over. Preserved
    unchanged from the pre-refactor shape so the rest of the code doesn't
    need to know profiles work by override anymore.
    """
    name: str
    alias: Optional[str] = None
    backend: str
    model_type: str = "chat"
    host: str = "primary"
    quantization: Optional[str] = "fp8"
    memory_gb: Optional[float] = None
    idle_timeout_minutes: int = 30
    auto_suspend_enabled: bool = False
    speculative_model: Optional[str] = None
    num_speculative_tokens: int = 5
    speculative_method: Optional[str] = None
    speculative_extra: Dict[str, Any] = Field(default_factory=dict)
    default: bool = False
    # CAPABILITY: does this model accept image input at all? Independent of
    # vision_default — a profile has exactly ONE vision_default (what the
    # "vision" alias resolves to) but may run SEVERAL vision-capable models
    # (2026-09-05: under `voice`, both qwen-flash-next and gemma4-2b see).
    # Published to the gateway as model_info.supports_vision so clients can
    # discover capability instead of hardcoding which alias is multimodal.
    vision: bool = False
    vision_default: bool = False  # profile's designated vision model → gateway "vision" alias
    extra_args: Dict[str, Any] = Field(default_factory=dict)
    docker_image: Optional[str] = None
    env_vars: Dict[str, str] = Field(default_factory=dict)
    volumes: List[str] = Field(default_factory=list)


class ModelsConfiguration(BaseModel):
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    models: Dict[str, ModelDefinition] = Field(default_factory=dict)
    # profiles[profile_name][alias] = override dict (empty means "use base as-is")
    profiles: Dict[str, Dict[str, Dict[str, Any]]] = Field(default_factory=dict)
    default_profile: Optional[str] = None


def _load_raw(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Failed to read {path}: {e}")
        return {}


def load_models_config(config_path: Optional[Path] = None) -> ModelsConfiguration:
    """Load models.yaml + optional .sparkstation.local.yaml overlay.

    The local overlay's `cluster.hosts` deep-merges over the base — it's the
    only place real IPs / ssh users / hostnames should appear. It may also
    override anything else in models.yaml (models, profiles, default_profile)
    on a per-machine basis if the user chooses.
    """
    if config_path is None:
        config_path = find_sparkstation_root() / "models.yaml"

    base = _load_raw(config_path)
    local = _load_raw(config_path.parent / LOCAL_CONFIG_FILENAME)
    merged = _deep_merge(base, local)

    if not merged:
        logger.warning(f"No models config found at {config_path}, using defaults")
        return ModelsConfiguration()

    try:
        cluster = ClusterConfig(**(merged.get("cluster") or {}))
        models = {
            alias: ModelDefinition(**defn)
            for alias, defn in (merged.get("models") or {}).items()
        }
        # Profiles: leave as raw override dicts; resolution happens in
        # get_profile_models(). Empty dicts and None both mean "use base".
        raw_profiles = merged.get("profiles") or {}
        profiles = {}
        for pname, entries in raw_profiles.items():
            profiles[pname] = {alias: (override or {}) for alias, override in (entries or {}).items()}

        return ModelsConfiguration(
            cluster=cluster,
            models=models,
            profiles=profiles,
            default_profile=merged.get("default_profile"),
        )
    except Exception as e:
        logger.error(f"Failed to parse merged models config: {e}")
        return ModelsConfiguration()


def _resolve(alias: str, base_defn: ModelDefinition, override: Dict[str, Any]) -> ModelConfigYAML:
    """Merge base + override → resolved ModelConfigYAML with alias set.

    Dict-valued fields (extra_args, env_vars, speculative_extra) deep-merge
    rather than replace so profiles can add a couple of extra_args without
    respecifying the whole block.
    """
    merged = base_defn.model_dump()
    for k, v in (override or {}).items():
        if k in ("extra_args", "env_vars", "speculative_extra") and isinstance(v, dict):
            merged[k] = _deep_merge(merged.get(k) or {}, v)
        else:
            merged[k] = v
    merged["alias"] = alias
    # A profile's designated VLM is vision-capable by definition — imply the
    # capability so a spec never has to carry both flags in agreement.
    if merged.get("vision_default"):
        merged["vision"] = True
    return ModelConfigYAML(**merged)


def get_cluster_config() -> ClusterConfig:
    return load_models_config().cluster


def get_profile_models(profile_name: str) -> List[ModelConfigYAML]:
    """Resolved list of models for a profile (base spec + per-profile overrides).

    Order is preserved from the YAML — first entry loads first. Aliases the
    profile references that aren't in the top-level `models:` dict are
    logged and skipped.
    """
    cfg = load_models_config()
    if profile_name not in cfg.profiles:
        logger.error(f"Profile not found: {profile_name}")
        return []
    resolved = []
    for alias, override in cfg.profiles[profile_name].items():
        base = cfg.models.get(alias)
        if base is None:
            logger.warning(
                f"Profile '{profile_name}' references unknown model alias '{alias}' — skipping"
            )
            continue
        resolved.append(_resolve(alias, base, override))
    return resolved


def find_model_by_alias(alias: str, profile_name: Optional[str] = None) -> Optional[ModelConfigYAML]:
    """Look up a model by alias, resolved through a profile if given.

    - `profile_name` set: return the profile's resolved entry for this alias,
      or None if the profile doesn't enable it.
    - `profile_name` None: fall through to `default_profile` from YAML; if
      that fails, return the base spec unchanged.
    """
    cfg = load_models_config()
    profile = profile_name or cfg.default_profile
    if profile and profile in cfg.profiles and alias in cfg.profiles[profile]:
        base = cfg.models.get(alias)
        if base is None:
            return None
        return _resolve(alias, base, cfg.profiles[profile][alias])
    base = cfg.models.get(alias)
    if base is None:
        return None
    return _resolve(alias, base, {})


def get_default_model_alias(profile_name: Optional[str] = None) -> Optional[str]:
    """Alias of the model marked `default: true` in the resolved profile."""
    profile = profile_name
    if profile is None:
        profile = load_models_config().default_profile
    if profile is None:
        return None
    for m in get_profile_models(profile):
        if m.default:
            return m.alias or m.name
    return None


def get_vision_model_alias(profile_name: Optional[str] = None) -> Optional[str]:
    """Alias of the model marked `vision_default: true` in the resolved profile.

    Published by gateway_sync as the "vision" alias so vision consumers (e.g.
    the ds4f-vision MCP) never hardcode a model name — coding profile resolves
    to qwen3-vl-4b, image-indexing to qwen3.8-27b (natively vision-capable).
    """
    profile = profile_name
    if profile is None:
        profile = load_models_config().default_profile
    if profile is None:
        return None
    for m in get_profile_models(profile):
        if m.vision_default:
            return m.alias or m.name
    return None


def get_vision_capable_aliases(profile_name: Optional[str] = None) -> Set[str]:
    """Aliases of EVERY vision-capable model in the resolved profile.

    Distinct from get_vision_model_alias(), which names the single model the
    "vision" alias resolves to. A profile routinely runs more than one model
    that accepts images (under `voice`: qwen-flash-next AND gemma4-2b), and
    clients need to know about all of them — publishing only the designated
    one is what left consumers hardcoding capability by model name.
    """
    profile = profile_name
    if profile is None:
        profile = load_models_config().default_profile
    if profile is None:
        return set()
    return {
        (m.alias or m.name) for m in get_profile_models(profile) if m.vision
    }


def list_profiles() -> List[str]:
    return list(load_models_config().profiles.keys())


# ─── Backward-compat shim (may be removed once callers migrate) ─────────────


def get_autoload_models() -> List[ModelConfigYAML]:
    """Deprecated. Kept so pre-refactor callers still resolve to *something*.

    Returns the default_profile's models if set; else the first profile's
    models; else an empty list. Prefer `get_profile_models(name)`.
    """
    cfg = load_models_config()
    fallback = cfg.default_profile or (next(iter(cfg.profiles), None))
    if fallback is None:
        return []
    return get_profile_models(fallback)
