#!/usr/bin/env python3
"""
Daily maintenance script for Sparkstation.

Performs automated cleanup and health checks:
- Cleans up old log files
- Vacuums SQLite database
- Detects stale/zombie models
- Checks for port leaks
- Generates resource usage report

Usage:
    python scripts/maintenance.py [--dry-run] [--verbose]

Recommended to run via cron:
    0 3 * * * /path/to/.venv/bin/python /path/to/scripts/maintenance.py
"""
import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from supervisor.config import settings
from supervisor.registry import ModelRegistry
from supervisor.resources import ResourceManager
from supervisor.models import ModelStatus

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("maintenance")


class MaintenanceReport:
    """Track maintenance operations and results."""

    def __init__(self):
        self.timestamp = datetime.now()
        self.operations: List[Dict[str, Any]] = []
        self.errors: List[str] = []

    def add_operation(self, operation: str, details: Dict[str, Any]):
        """Record a maintenance operation."""
        self.operations.append({"operation": operation, "details": details})

    def add_error(self, error: str):
        """Record an error."""
        self.errors.append(error)

    def summary(self) -> str:
        """Generate summary report."""
        lines = [
            "=" * 60,
            f"Sparkstation Maintenance Report - {self.timestamp.isoformat()}",
            "=" * 60,
            "",
        ]

        for op in self.operations:
            lines.append(f"✓ {op['operation']}")
            for key, value in op["details"].items():
                lines.append(f"  - {key}: {value}")
            lines.append("")

        if self.errors:
            lines.append("ERRORS:")
            for error in self.errors:
                lines.append(f"  ✗ {error}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


async def cleanup_old_logs(report: MaintenanceReport, dry_run: bool = False):
    """Remove log files older than 30 days."""
    logger.info("Checking for old log files...")

    log_path = Path(settings.log_file_path)
    if not log_path.exists():
        report.add_operation("Log cleanup", {"status": "No logs found"})
        return

    log_dir = log_path.parent
    old_logs = []
    cutoff_date = datetime.now() - timedelta(days=30)

    # Find old rotated logs
    for log_file in log_dir.glob(f"{log_path.name}.*"):
        if log_file.stat().st_mtime < cutoff_date.timestamp():
            old_logs.append(log_file)

    if not old_logs:
        report.add_operation("Log cleanup", {"old_logs_found": 0})
        return

    if dry_run:
        report.add_operation(
            "Log cleanup (DRY RUN)",
            {"old_logs_found": len(old_logs), "would_delete": [str(f) for f in old_logs]},
        )
    else:
        deleted_size = sum(f.stat().st_size for f in old_logs)
        for log_file in old_logs:
            log_file.unlink()
        report.add_operation(
            "Log cleanup",
            {
                "deleted_count": len(old_logs),
                "freed_mb": round(deleted_size / 1024 / 1024, 2),
            },
        )


async def vacuum_database(report: MaintenanceReport, dry_run: bool = False):
    """Optimize SQLite database."""
    logger.info("Vacuuming database...")

    # Extract path from database_url (format: sqlite+aiosqlite:///./data/sparkstation.db)
    db_url = settings.database_url
    if not db_url.startswith("sqlite"):
        report.add_operation("Database vacuum", {"status": "Not a SQLite database"})
        return

    # Parse path from URL
    db_path_str = db_url.replace("sqlite+aiosqlite:///", "").replace("sqlite:///", "")
    db_path = Path(db_path_str)

    if not db_path.exists():
        report.add_operation("Database vacuum", {"status": "Database not found"})
        return

    if dry_run:
        report.add_operation(
            "Database vacuum (DRY RUN)", {"would_vacuum": str(db_path)}
        )
        return

    try:
        size_before = db_path.stat().st_size

        # Import here to avoid circular dependencies
        from sqlalchemy import create_engine, text

        # Use synchronous SQLite for VACUUM
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as conn:
            conn.execute(text("VACUUM"))
            conn.commit()

        size_after = db_path.stat().st_size
        saved_kb = round((size_before - size_after) / 1024, 2)

        report.add_operation(
            "Database vacuum",
            {
                "size_before_mb": round(size_before / 1024 / 1024, 2),
                "size_after_mb": round(size_after / 1024 / 1024, 2),
                "saved_kb": saved_kb,
            },
        )
    except Exception as e:
        report.add_error(f"Database vacuum failed: {e}")


async def detect_stale_models(report: MaintenanceReport, dry_run: bool = False):
    """Find models in STARTING state for >10 minutes or FAILED models."""
    logger.info("Checking for stale models...")

    registry = ModelRegistry()
    await registry.initialize()

    try:
        all_models = await registry.list_all()
        stale_models = []
        failed_models = []

        for model in all_models:
            # Check STARTING models stuck for >10 minutes
            if model.status == ModelStatus.STARTING:
                age_minutes = (datetime.now() - model.started_at).total_seconds() / 60
                if age_minutes > 10:
                    stale_models.append(
                        {
                            "id": model.id,
                            "name": model.model_name,
                            "status": model.status.value,
                            "age_minutes": round(age_minutes, 1),
                        }
                    )

            # Track FAILED models
            elif model.status == ModelStatus.FAILED:
                failed_models.append(
                    {
                        "id": model.id,
                        "name": model.model_name,
                        "restart_count": model.restart_count,
                    }
                )

        details = {
            "total_models": len(all_models),
            "stale_count": len(stale_models),
            "failed_count": len(failed_models),
        }

        if stale_models:
            details["stale_models"] = stale_models
        if failed_models:
            details["failed_models"] = failed_models

        report.add_operation("Stale model detection", details)

        # Suggest cleanup actions
        if stale_models and not dry_run:
            logger.warning(
                f"Found {len(stale_models)} stale STARTING models - manual intervention recommended"
            )
        if failed_models and not dry_run:
            logger.warning(
                f"Found {len(failed_models)} FAILED models - consider stopping via API"
            )

    except Exception as e:
        report.add_error(f"Stale model detection failed: {e}")


async def check_port_leaks(report: MaintenanceReport, dry_run: bool = False):
    """Check for allocated ports without corresponding running models."""
    logger.info("Checking for port leaks...")

    try:
        resource_manager = ResourceManager()
        registry = ModelRegistry()
        await registry.initialize()

        try:
            all_models = await registry.list_all()

            # Get ports that should be allocated
            expected_ports = {
                model.port for model in all_models if model.status in [ModelStatus.RUNNING, ModelStatus.STARTING]
            }

            # Get actually allocated ports
            allocated_ports = set(resource_manager.allocated_ports.values())

            # Find leaks
            leaked_ports = allocated_ports - expected_ports
            orphaned_models = expected_ports - allocated_ports

            details = {
                "total_allocated": len(allocated_ports),
                "expected_running": len(expected_ports),
                "leaked_ports": len(leaked_ports),
                "orphaned_models": len(orphaned_models),
            }

            if leaked_ports:
                details["leaked_port_list"] = sorted(list(leaked_ports))
            if orphaned_models:
                details["orphaned_port_list"] = sorted(list(orphaned_models))

            report.add_operation("Port leak detection", details)

            if leaked_ports:
                logger.warning(f"Found {len(leaked_ports)} leaked ports - requires manual cleanup")

        except Exception as e:
            report.add_error(f"Port leak detection failed: {e}")

    except Exception as e:
        report.add_error(f"Port leak detection outer failed: {e}")


async def generate_resource_report(report: MaintenanceReport, dry_run: bool = False):
    """Generate current resource usage snapshot."""
    logger.info("Generating resource usage report...")

    try:
        resource_manager = ResourceManager()
        registry = ModelRegistry()
        await registry.initialize()

        try:
            all_models = await registry.list_all()

            # Categorize by status
            by_status = {}
            for model in all_models:
                status = model.status.value
                by_status[status] = by_status.get(status, 0) + 1

            # Calculate memory usage
            total_memory = sum(
                model.memory_gb or 0
                for model in all_models
                if model.status in [ModelStatus.RUNNING, ModelStatus.STARTING]
            )

            status = resource_manager.get_resource_status()

            details = {
                "total_models": len(all_models),
                "by_status": by_status,
                "memory_used_gb": round(total_memory, 1),
                "memory_limit_gb": status["unified_memory_limit_gb"],
                "memory_utilization_pct": round(
                    (total_memory / status["unified_memory_limit_gb"]) * 100, 1
                )
                if status["unified_memory_limit_gb"] > 0
                else 0,
                "available_ports": status["available_ports"],
            }

            report.add_operation("Resource usage snapshot", details)

        except Exception as e:
            report.add_error(f"Resource report generation failed: {e}")

    except Exception as e:
        report.add_error(f"Resource report outer failed: {e}")


async def main():
    """Run all maintenance tasks."""
    parser = argparse.ArgumentParser(description="Sparkstation daily maintenance")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")

    report = MaintenanceReport()

    # Run maintenance tasks
    await cleanup_old_logs(report, args.dry_run)
    await vacuum_database(report, args.dry_run)
    await detect_stale_models(report, args.dry_run)
    await check_port_leaks(report, args.dry_run)
    await generate_resource_report(report, args.dry_run)

    # Print summary
    print(report.summary())

    # Exit with error code if errors occurred
    if report.errors:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
