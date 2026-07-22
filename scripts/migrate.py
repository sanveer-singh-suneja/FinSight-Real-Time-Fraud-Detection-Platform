#!/usr/bin/env python3
"""
FinSight Database Migration Runner.
Applies Alembic migrations and optionally seeds rule data.
Run: python scripts/migrate.py [--seed-rules]
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_migrations(dsn: str | None = None) -> None:
    """Apply all pending Alembic migrations."""
    import os
    from alembic.config import Config
    from alembic import command

    alembic_cfg = Config(str(Path(__file__).parent.parent / "alembic.ini"))

    if dsn:
        # Override with provided DSN (sync driver)
        sync_dsn = dsn.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
        alembic_cfg.set_main_option("sqlalchemy.url", sync_dsn)

    print("Running Alembic migrations...")
    command.upgrade(alembic_cfg, "head")
    print("✓ Migrations applied")


async def seed_rules() -> None:
    """Sync rule definitions from YAML to the database."""
    from api.services.rule_engine import RuleEngine
    from configs.settings import get_settings
    from database.repositories import RuleRepository
    from database.session import get_db_session

    settings = get_settings()
    engine = RuleEngine()

    async with get_db_session() as session:
        repo = RuleRepository(session)
        synced = 0
        for rule in engine._rules:
            await repo.upsert({
                "rule_id": rule["id"],
                "name": rule["name"],
                "description": rule.get("description", ""),
                "category": rule["category"],
                "severity": rule["severity"],
                "action": rule["action"],
                "conditions": rule.get("conditions"),
                "enabled": rule.get("enabled", True),
                "hit_count": 0,
            })
            synced += 1

    print(f"✓ Synced {synced} rules to database")


def main() -> None:
    parser = argparse.ArgumentParser(description="FinSight Database Setup")
    parser.add_argument("--dsn", help="Database DSN (overrides settings)")
    parser.add_argument("--seed-rules", action="store_true", help="Sync rules to DB")
    parser.add_argument("--rollback", help="Rollback to revision")
    args = parser.parse_args()

    if args.rollback:
        from alembic.config import Config
        from alembic import command
        alembic_cfg = Config("alembic.ini")
        command.downgrade(alembic_cfg, args.rollback)
        print(f"✓ Rolled back to {args.rollback}")
        return

    run_migrations(args.dsn)

    if args.seed_rules:
        asyncio.run(seed_rules())


if __name__ == "__main__":
    main()
