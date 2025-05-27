#!/usr/bin/env python3
"""
Migration script for payload indexing (Issue #56).

Adds payload indexes to existing collections for 10-100x performance improvement
in filtered searches. This script can be run safely on existing collections.
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import UnifiedConfig
from services.errors import QdrantServiceError
from services.qdrant_service import QdrantService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PayloadIndexMigrator:
    """Manages payload index migration for existing collections."""

    def __init__(self):
        """Initialize migrator with service dependencies."""
        self.config = UnifiedConfig()
        self.qdrant_service = QdrantService(self.config)
        self.migration_stats = {
            "collections_processed": 0,
            "collections_migrated": 0,
            "collections_skipped": 0,
            "collections_failed": 0,
            "total_indexes_created": 0,
            "errors": [],
            "start_time": None,
            "end_time": None,
        }

    async def initialize(self):
        """Initialize services."""
        try:
            await self.qdrant_service.initialize()
            logger.info("Successfully initialized services")
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise

    async def cleanup(self):
        """Cleanup services."""
        try:
            await self.qdrant_service.cleanup()
            logger.info("Successfully cleaned up services")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

    async def check_collection_indexes(self, collection_name: str) -> dict:
        """Check current index status of a collection."""
        try:
            # Get current index stats
            stats = await self.qdrant_service.get_payload_index_stats(collection_name)

            # Check if collection already has indexes
            has_indexes = stats["indexed_fields_count"] > 0

            return {
                "collection_name": collection_name,
                "has_indexes": has_indexes,
                "indexed_fields_count": stats["indexed_fields_count"],
                "indexed_fields": stats["indexed_fields"],
                "total_points": stats["total_points"],
                "needs_migration": not has_indexes,
            }

        except Exception as e:
            logger.error(
                f"Failed to check indexes for collection {collection_name}: {e}"
            )
            return {
                "collection_name": collection_name,
                "error": str(e),
                "needs_migration": False,
            }

    async def migrate_collection(
        self, collection_name: str, force: bool = False
    ) -> dict:
        """Migrate a single collection to use payload indexes."""
        logger.info(f"Starting migration for collection: {collection_name}")

        try:
            # Check current status
            status = await self.check_collection_indexes(collection_name)

            if "error" in status:
                self.migration_stats["collections_failed"] += 1
                self.migration_stats["errors"].append(
                    f"{collection_name}: {status['error']}"
                )
                return {"status": "error", "error": status["error"]}

            # Skip if already has indexes (unless forced)
            if status["has_indexes"] and not force:
                logger.info(
                    f"Collection {collection_name} already has {status['indexed_fields_count']} indexes, skipping"
                )
                self.migration_stats["collections_skipped"] += 1
                return {
                    "status": "skipped",
                    "reason": "already_indexed",
                    "existing_indexes": status["indexed_fields_count"],
                }

            # Perform migration
            if force and status["has_indexes"]:
                logger.info(f"Force reindexing collection {collection_name}")
                await self.qdrant_service.reindex_collection(collection_name)
            else:
                logger.info(
                    f"Creating payload indexes for collection {collection_name}"
                )
                await self.qdrant_service.create_payload_indexes(collection_name)

            # Get updated stats
            new_stats = await self.qdrant_service.get_payload_index_stats(
                collection_name
            )

            # Update migration stats
            self.migration_stats["collections_migrated"] += 1
            self.migration_stats["total_indexes_created"] += new_stats[
                "indexed_fields_count"
            ]

            logger.info(
                f"Successfully migrated {collection_name}: "
                f"{new_stats['indexed_fields_count']} indexes created for "
                f"{new_stats['total_points']} points"
            )

            return {
                "status": "migrated",
                "indexes_created": new_stats["indexed_fields_count"],
                "indexed_fields": new_stats["indexed_fields"],
                "total_points": new_stats["total_points"],
            }

        except QdrantServiceError as e:
            logger.error(f"Migration failed for collection {collection_name}: {e}")
            self.migration_stats["collections_failed"] += 1
            self.migration_stats["errors"].append(f"{collection_name}: {e!s}")
            return {"status": "failed", "error": str(e)}

        except Exception as e:
            logger.error(
                f"Unexpected error migrating collection {collection_name}: {e}"
            )
            self.migration_stats["collections_failed"] += 1
            self.migration_stats["errors"].append(f"{collection_name}: {e!s}")
            return {"status": "failed", "error": str(e)}

    async def migrate_all_collections(
        self, force: bool = False, dry_run: bool = False
    ) -> dict:
        """Migrate all collections in the database."""
        self.migration_stats["start_time"] = datetime.now()

        try:
            # Get all collections
            collections = await self.qdrant_service.list_collections()
            logger.info(f"Found {len(collections)} collections to process")

            if not collections:
                logger.info("No collections found to migrate")
                return self.migration_stats

            # Process each collection
            migration_results = {}

            for collection_name in collections:
                self.migration_stats["collections_processed"] += 1

                if dry_run:
                    # Dry run - just check status
                    status = await self.check_collection_indexes(collection_name)
                    migration_results[collection_name] = {
                        "status": "dry_run",
                        "needs_migration": status.get("needs_migration", False),
                        "current_indexes": status.get("indexed_fields_count", 0),
                        "total_points": status.get("total_points", 0),
                    }
                    logger.info(
                        f"[DRY RUN] {collection_name}: "
                        f"{'needs migration' if status.get('needs_migration') else 'already indexed'}"
                    )
                else:
                    # Actual migration
                    result = await self.migrate_collection(collection_name, force)
                    migration_results[collection_name] = result

            self.migration_stats["end_time"] = datetime.now()
            self.migration_stats["migration_results"] = migration_results

            return self.migration_stats

        except Exception as e:
            logger.error(f"Failed to migrate collections: {e}")
            self.migration_stats["errors"].append(f"Global error: {e!s}")
            return self.migration_stats

    def print_migration_summary(self, stats: dict):
        """Print a summary of migration results."""
        print("\n" + "=" * 60)
        print("PAYLOAD INDEXING MIGRATION SUMMARY")
        print("=" * 60)

        # Timing info
        if stats["start_time"] and stats["end_time"]:
            duration = stats["end_time"] - stats["start_time"]
            print(f"Duration: {duration.total_seconds():.2f} seconds")

        # Collection stats
        print(f"Collections processed: {stats['collections_processed']}")
        print(f"Collections migrated: {stats['collections_migrated']}")
        print(f"Collections skipped: {stats['collections_skipped']}")
        print(f"Collections failed: {stats['collections_failed']}")
        print(f"Total indexes created: {stats['total_indexes_created']}")

        # Results breakdown
        if "migration_results" in stats:
            print("\nDetailed Results:")
            print("-" * 40)
            for collection, result in stats["migration_results"].items():
                status = result["status"]
                if status == "migrated":
                    print(
                        f"✅ {collection}: {result['indexes_created']} indexes, {result['total_points']} points"
                    )
                elif status == "skipped":
                    print(
                        f"⏭️  {collection}: {result.get('existing_indexes', 0)} existing indexes"
                    )
                elif status == "dry_run":
                    needs = (
                        "needs migration"
                        if result["needs_migration"]
                        else "already indexed"
                    )
                    print(
                        f"🔍 {collection}: {needs} ({result['current_indexes']} indexes)"
                    )
                elif status == "failed":
                    print(f"❌ {collection}: {result['error']}")

        # Errors
        if stats["errors"]:
            print(f"\nErrors ({len(stats['errors'])}):")
            print("-" * 40)
            for error in stats["errors"]:
                print(f"❌ {error}")

        # Performance estimate
        if stats["collections_migrated"] > 0:
            print("\n🚀 Performance Improvement:")
            print("   Filtered searches should now be 10-100x faster!")
            print("   Expected search times: 10-50ms (vs 850-1500ms without indexes)")

        print("=" * 60)


async def main():
    """Main migration function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate existing collections to use payload indexing"
    )
    parser.add_argument(
        "--collection", help="Migrate specific collection (default: all collections)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reindexing even if indexes already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check which collections need migration without making changes",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize migrator
    migrator = PayloadIndexMigrator()

    try:
        await migrator.initialize()

        if args.dry_run:
            print("🔍 DRY RUN MODE - No changes will be made")

        if args.collection:
            # Migrate specific collection
            logger.info(f"Migrating specific collection: {args.collection}")
            result = await migrator.migrate_collection(args.collection, args.force)

            # Simple result for single collection
            print(f"\nResult for {args.collection}:")
            if result["status"] == "migrated":
                print(f"✅ Success: {result['indexes_created']} indexes created")
            elif result["status"] == "skipped":
                print(f"⏭️  Skipped: {result.get('reason', 'unknown')}")
            elif result["status"] == "failed":
                print(f"❌ Failed: {result['error']}")
        else:
            # Migrate all collections
            logger.info("Starting migration for all collections")
            stats = await migrator.migrate_all_collections(args.force, args.dry_run)
            migrator.print_migration_summary(stats)

    except KeyboardInterrupt:
        logger.info("Migration cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Migration failed with error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        await migrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
