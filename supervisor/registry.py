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
        self.database_url = database_url or settings.database_url
        self.engine = create_async_engine(self.database_url, echo=False)
        self.async_session = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def initialize(self):
        """Initialize database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(ModelInstanceDB.metadata.create_all)
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
        """Get model instance by alias."""
        async with self.async_session() as session:
            result = await session.execute(
                select(ModelInstanceDB).where(ModelInstanceDB.model_alias == alias)
            )
            db_instance = result.scalar_one_or_none()
            if db_instance:
                return self._to_pydantic(db_instance)
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
        return ModelInstance(
            id=db_instance.id,
            model_name=db_instance.model_name,
            model_alias=db_instance.model_alias,
            backend=Backend(db_instance.backend),
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

    @staticmethod
    def generate_id(model_name: str) -> str:
        """Generate unique model ID."""
        # Use last part of model name + short UUID
        name_part = model_name.split("/")[-1].lower().replace("_", "-")
        uuid_part = str(uuid.uuid4())[:8]
        return f"{name_part}-{uuid_part}"
