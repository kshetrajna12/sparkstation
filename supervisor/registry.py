"""
Model registry for tracking running model instances.
Uses SQLAlchemy for persistence.
"""
import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select, update, delete
from supervisor.models import ModelInstanceDB, ModelInstance, ModelStatus, HealthStatus, Backend
from supervisor.config import settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for tracking model instances with SQLite persistence.
    """

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or settings.supervisor_database_url
        # timeout=5 → SQLite busy_timeout. Health checks fan out N concurrent
        # registry.update()s (plus the auto-suspend loop, restart watcher, and
        # record_request); SQLite is single-writer, and with the default
        # busy_timeout of 0 any colliding write fails instantly ("database is
        # locked"). WAL mode further lets readers proceed during a write.
        self.engine = create_async_engine(
            self.database_url, echo=False, connect_args={"timeout": 5}
        )
        if self.database_url.startswith("sqlite"):
            from sqlalchemy import event

            @event.listens_for(self.engine.sync_engine, "connect")
            def _set_sqlite_pragmas(dbapi_conn, _record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.close()

        self.async_session = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def initialize(self):
        """Initialize database tables + apply lightweight forward-compat migrations.

        `create_all` only creates tables that don't exist — it won't add new columns
        to an existing table. For SQLite, adding an optional column to an existing
        DB is safe via `ALTER TABLE ADD COLUMN`. We do it idempotently here so
        upgrading past pre-cluster deploys (where `host` didn't exist) doesn't
        require a manual `sqlite3` step or a DB wipe.
        """
        from sqlalchemy import text
        async with self.engine.begin() as conn:
            await conn.run_sync(ModelInstanceDB.metadata.create_all)

            # Idempotent per-column adds. Keeps the list small on purpose —
            # if the schema drifts far from the original, do a proper migration.
            pending_columns = [
                ("host", "VARCHAR"),
            ]
            result = await conn.execute(text("PRAGMA table_info(model_instances)"))
            existing = {row[1] for row in result.fetchall()}
            for col_name, col_type in pending_columns:
                if col_name in existing:
                    continue
                try:
                    await conn.execute(
                        text(f"ALTER TABLE model_instances ADD COLUMN {col_name} {col_type}")
                    )
                    logger.info(f"Registry migration: added model_instances.{col_name}")
                except Exception as e:
                    logger.warning(f"Registry migration skipped for {col_name}: {e}")
        logger.info("Model registry initialized")

    async def create(self, instance: ModelInstance) -> ModelInstance:
        """
        Create a new model instance.

        Args:
            instance: Model instance to create

        Returns:
            Created instance
        """
        async with self.async_session() as session:
            db_instance = ModelInstanceDB(
                id=instance.id,
                model_name=instance.model_name,
                model_alias=instance.model_alias,
                backend=instance.backend,
                model_type=instance.model_type,
                host=instance.host or "primary",
                status=instance.status,
                health_status=instance.health_status,
                port=instance.port,
                gpu_ids=instance.gpu_ids,
                base_url=instance.base_url,
                pid=instance.pid,
                container_id=instance.container_id,
                systemd_service=instance.systemd_service,
                started_at=instance.started_at,
                last_request_time=instance.last_request_time,
                auto_suspend_enabled=instance.auto_suspend_enabled,
                idle_timeout_minutes=instance.idle_timeout_minutes,
                memory_gb=instance.memory_gb,
                saved_config=instance.saved_config,
                extra_args=instance.extra_args,
                restart_count=instance.restart_count,
                last_restart_time=instance.last_restart_time,
            )
            session.add(db_instance)
            await session.commit()
            logger.info(f"Created model instance: {instance.id}")
            return instance

    async def get(self, model_id: str) -> Optional[ModelInstance]:
        """
        Get model instance by ID.

        Args:
            model_id: Model instance ID

        Returns:
            Model instance or None if not found
        """
        async with self.async_session() as session:
            result = await session.execute(select(ModelInstanceDB).where(ModelInstanceDB.id == model_id))
            db_instance = result.scalar_one_or_none()
            if db_instance:
                return self._to_pydantic(db_instance)
            return None

    async def get_by_alias(self, alias: str) -> Optional[ModelInstance]:
        """Get model instance by alias.

        model_alias has no unique constraint and purge/recycle paths can
        transiently leave duplicates — scalar_one_or_none() would raise
        MultipleResultsFound and 500 every caller. Prefer the newest row.
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(ModelInstanceDB)
                .where(ModelInstanceDB.model_alias == alias)
                .order_by(ModelInstanceDB.started_at.desc())
            )
            db_instances = result.scalars().all()
            if len(db_instances) > 1:
                logger.warning(
                    f"{len(db_instances)} registry rows share alias '{alias}' — using newest"
                )
            if db_instances:
                return self._to_pydantic(db_instances[0])
            return None

    async def list_all(self) -> List[ModelInstance]:
        """List all model instances."""
        async with self.async_session() as session:
            result = await session.execute(select(ModelInstanceDB))
            db_instances = result.scalars().all()
            return [self._to_pydantic(db_inst) for db_inst in db_instances]

    async def list_by_status(self, status: ModelStatus) -> List[ModelInstance]:
        """List models by status."""
        async with self.async_session() as session:
            result = await session.execute(
                select(ModelInstanceDB).where(ModelInstanceDB.status == status.value)
            )
            db_instances = result.scalars().all()
            return [self._to_pydantic(db_inst) for db_inst in db_instances]

    async def list_running(self) -> List[ModelInstance]:
        """List running models."""
        return await self.list_by_status(ModelStatus.RUNNING)

    async def list_suspended(self) -> List[ModelInstance]:
        """List suspended models."""
        return await self.list_by_status(ModelStatus.SUSPENDED)

    async def list_by_type(self, model_type: str) -> List[ModelInstance]:
        """List models by type (chat or embedding)."""
        async with self.async_session() as session:
            result = await session.execute(
                select(ModelInstanceDB).where(ModelInstanceDB.model_type == model_type)
            )
            db_instances = result.scalars().all()
            return [self._to_pydantic(db_inst) for db_inst in db_instances]

    async def update(self, instance: ModelInstance) -> ModelInstance:
        """
        Update model instance.

        Args:
            instance: Updated instance

        Returns:
            Updated instance
        """
        async with self.async_session() as session:
            await session.execute(
                update(ModelInstanceDB)
                .where(ModelInstanceDB.id == instance.id)
                .values(
                    status=instance.status,
                    health_status=instance.health_status,
                    pid=instance.pid,
                    container_id=instance.container_id,
                    # started_at changes on restart/resume relaunches; without
                    # persisting it, uptime and stuck-STARTING timeouts would
                    # be computed from the ORIGINAL launch time.
                    started_at=instance.started_at,
                    last_health_check=instance.last_health_check,
                    last_request_time=instance.last_request_time,
                    stopped_at=instance.stopped_at,
                    saved_config=instance.saved_config,
                    auto_suspend_enabled=instance.auto_suspend_enabled,
                    idle_timeout_minutes=instance.idle_timeout_minutes,
                    restart_count=instance.restart_count,
                    last_restart_time=instance.last_restart_time,
                )
            )
            await session.commit()
            logger.debug(f"Updated model instance: {instance.id}")
            return instance

    async def delete(self, model_id: str) -> bool:
        """
        Delete model instance.

        Args:
            model_id: Model instance ID

        Returns:
            True if deleted, False if not found
        """
        async with self.async_session() as session:
            result = await session.execute(
                delete(ModelInstanceDB).where(ModelInstanceDB.id == model_id)
            )
            await session.commit()
            deleted = result.rowcount > 0
            if deleted:
                logger.info(f"Deleted model instance: {model_id}")
            return deleted

    async def record_request(self, model_id: str):
        """Record that a model served a request (update last_request_time)."""
        async with self.async_session() as session:
            await session.execute(
                update(ModelInstanceDB)
                .where(ModelInstanceDB.id == model_id)
                .values(last_request_time=datetime.now())
            )
            await session.commit()
            logger.debug(f"Recorded request for model: {model_id}")

    def _to_pydantic(self, db_instance: ModelInstanceDB) -> ModelInstance:
        """Convert SQLAlchemy model to Pydantic model."""
        from supervisor.models import ModelType
        return ModelInstance(
            id=db_instance.id,
            model_name=db_instance.model_name,
            model_alias=db_instance.model_alias,
            backend=Backend(db_instance.backend),
            model_type=ModelType(db_instance.model_type) if db_instance.model_type else ModelType.CHAT,
            # Legacy rows (from before cluster support) have NULL host; treat as
            # "primary" so pre-cluster deployments keep working after upgrade.
            host=db_instance.host or "primary",
            status=ModelStatus(db_instance.status),
            health_status=HealthStatus(db_instance.health_status),
            port=db_instance.port,
            gpu_ids=db_instance.gpu_ids,
            base_url=db_instance.base_url,
            pid=db_instance.pid,
            container_id=db_instance.container_id,
            systemd_service=db_instance.systemd_service,
            started_at=db_instance.started_at,
            stopped_at=db_instance.stopped_at,
            last_health_check=db_instance.last_health_check,
            last_request_time=db_instance.last_request_time,
            auto_suspend_enabled=db_instance.auto_suspend_enabled,
            idle_timeout_minutes=db_instance.idle_timeout_minutes,
            memory_gb=db_instance.memory_gb,
            saved_config=db_instance.saved_config,
            extra_args=db_instance.extra_args,
            restart_count=db_instance.restart_count or 0,
            last_restart_time=db_instance.last_restart_time,
        )

    async def reconcile_state(self) -> Dict[str, Any]:
        """
        Reconcile database state with actual running containers, across
        every host in the cluster.

        For each model in the registry, checks the docker daemon on the model's
        assigned host (via DOCKER_HOST=ssh://... for remote roles). Also enumerates
        sparkstation-* containers on every host to catch orphans.

        Called on startup to fix inconsistencies.
        """
        import asyncio
        import functools
        import subprocess
        from supervisor.cluster_helpers import merged_env
        from supervisor.models_config import get_cluster_config

        # docker-over-SSH calls can take seconds each; run them off the event
        # loop so background tasks (health checks, API) aren't stalled.
        async def _run(cmd, timeout, env):
            return await asyncio.to_thread(
                functools.partial(subprocess.run, cmd, capture_output=True, text=True, timeout=timeout, env=env)
            )

        logger.info("Starting database reconciliation...")
        summary = {
            "checked": 0,
            "fixed_stopped": [],
            "fixed_running": [],
            "removed_orphaned": [],
        }

        # Get all models from database
        models = await self.list_all()
        summary["checked"] = len(models)

        for model in models:
            # Only reconcile Docker-based models
            if not model.container_id:
                continue

            # Per-model host env so remote containers get inspected on the
            # right daemon (not the local one).
            model_env = merged_env(model.host or "primary")

            # Check if container exists and is running
            try:
                result = await _run(
                    ["docker", "inspect", "-f", "{{.State.Running}}", model.container_id],
                    timeout=10,  # SSH transport adds latency vs local socket
                    env=model_env,
                )

                container_running = result.stdout.strip() == "true"
                db_thinks_running = model.status in [ModelStatus.STARTING, ModelStatus.RUNNING]

                # Fix mismatches
                if container_running and not db_thinks_running:
                    # Container is running but DB says stopped - mark as running
                    logger.warning(f"Container {model.container_id[:12]} is running on host={model.host or 'primary'} but DB says stopped. Fixing...")
                    model.status = ModelStatus.RUNNING
                    model.health_status = HealthStatus.UNKNOWN
                    await self.update(model)
                    summary["fixed_running"].append(model.id)

                elif not container_running and db_thinks_running:
                    # Container stopped but DB says running - mark as stopped
                    logger.warning(f"Container {model.container_id[:12]} stopped on host={model.host or 'primary'} but DB says running. Fixing...")
                    model.status = ModelStatus.STOPPED
                    model.health_status = HealthStatus.UNHEALTHY
                    model.stopped_at = datetime.now()
                    await self.update(model)
                    summary["fixed_stopped"].append(model.id)

            except subprocess.TimeoutExpired:
                logger.error(f"Timeout checking container {model.container_id[:12]} on host={model.host or 'primary'}")
            except Exception as e:
                # Container doesn't exist - mark as stopped
                logger.warning(f"Container {model.container_id[:12]} doesn't exist on host={model.host or 'primary'}: {e}. Marking as stopped...")
                model.status = ModelStatus.STOPPED
                model.health_status = HealthStatus.UNHEALTHY
                model.stopped_at = datetime.now()
                await self.update(model)
                summary["fixed_stopped"].append(model.id)

        # Check for orphaned containers on every host. Loop through
        # cluster.hosts so we catch stragglers left on the worker too, not just
        # the primary. `--no-trunc` is essential — see bug #6 note in earlier
        # commit for why short-ID matching wiped every legitimate container.
        cluster = get_cluster_config()
        known_container_ids = {m.container_id for m in models if m.container_id}
        for host_role in cluster.hosts:
            host_env = merged_env(host_role)
            try:
                result = await _run(
                    ["docker", "ps", "-a", "--filter", "name=sparkstation-", "--no-trunc", "--format", "{{.ID}}:{{.Names}}"],
                    timeout=15,
                    env=host_env,
                )

                if result.returncode != 0:
                    logger.warning(f"docker ps failed on host={host_role}: {result.stderr[:200]}")
                    continue
                if not result.stdout.strip():
                    continue

                for line in result.stdout.strip().split("\n"):
                    if ":" not in line:
                        continue
                    container_id, name = line.split(":", 1)

                    if container_id not in known_container_ids:
                        logger.warning(f"Found orphaned container {name} ({container_id[:12]}) on host={host_role}. Removing...")
                        await _run(
                            ["docker", "rm", "-f", container_id],
                            timeout=15,
                            env=host_env,
                        )
                        summary["removed_orphaned"].append(f"{host_role}:{name}")

            except Exception as e:
                logger.error(f"Failed to check for orphaned containers on host={host_role}: {e}")

        logger.info(f"Reconciliation complete: {summary}")
        return summary

    @staticmethod
    def generate_id(model_name: str) -> str:
        """Generate unique model ID."""
        # Use last part of model name + short UUID
        name_part = model_name.split("/")[-1].lower().replace("_", "-")
        uuid_part = str(uuid.uuid4())[:8]
        return f"{name_part}-{uuid_part}"
