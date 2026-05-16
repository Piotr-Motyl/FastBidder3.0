"""
Redis progress tracker for Celery jobs.

Stores job status, history, and results in Redis. Falls back to file storage on Redis failure.
"""

import gzip
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, cast

from redis import Redis
from redis.exceptions import RedisError

# Configure logger for this module
logger = logging.getLogger(__name__)


class RedisProgressTracker:
    """
    Track Celery job progress in Redis with history, heartbeat, and file fallback.

    Redis keys (job_id is always str):
        "progress:{job_id}"         → JSON progress dict (TTL: 1h)
        "result:{job_id}"           → JSON result dict (TTL: 24h; gzip-compressed if >1MB)
        "progress:{job_id}:history" → Redis LIST, last 10 updates (FIFO)

    Progress dict fields: status, progress (0-100), message, current_item, total_items,
        stage, eta_seconds, memory_mb, errors, last_heartbeat.

    On Redis failure: writes to {FALLBACK_DIR}/progress_{job_id}.json (graceful degradation).
    """

    def __init__(
        self,
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
        redis_db: int = 0,
    ) -> None:
        """
        Connect to Redis. TTLs: progress=1h, result=24h (env: REDIS_PROGRESS_TTL, REDIS_RESULT_TTL).
        Fallback dir: FALLBACK_DIR env or {project}/data/fallback.
        """
        self.redis_host = redis_host or os.getenv("REDIS_HOST", "localhost")
        self.redis_port = redis_port or int(os.getenv("REDIS_PORT", "6379"))
        self.redis_db = redis_db

        # Initialize Redis connection
        self.redis: Redis = Redis(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            decode_responses=True,  # Auto-decode bytes to str
        )

        # TTL configuration (Phase 2 - different TTLs for progress vs result)
        self.progress_ttl: int = int(os.getenv("REDIS_PROGRESS_TTL", "3600"))  # 1h
        self.result_ttl: int = int(os.getenv("REDIS_RESULT_TTL", "86400"))  # 24h

        # Fallback configuration (Phase 2 - error recovery)
        # Use project directory for fallback storage instead of /tmp
        default_fallback = Path(__file__).parent.parent.parent.parent / "data" / "fallback"
        self.fallback_dir = Path(os.getenv("FALLBACK_DIR", str(default_fallback)))

        # Try to create fallback directory, log warning if fails
        try:
            self.fallback_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Fallback directory initialized: {self.fallback_dir}")
        except PermissionError as e:
            logger.warning(
                f"Cannot create fallback directory {self.fallback_dir}: {e}. "
                "Fallback storage will not work. Set FALLBACK_DIR env variable to a writable path."
            )

        # History configuration (Phase 2)
        self.max_history_entries: int = 10

        # Compression threshold (Phase 2)
        self.compression_threshold_bytes: int = 1024 * 1024  # 1MB

    def _get_progress_key(self, job_id: str) -> str:
        return f"progress:{job_id}"

    def _get_result_key(self, job_id: str) -> str:
        return f"result:{job_id}"

    def _get_history_key(self, job_id: str) -> str:
        return f"progress:{job_id}:history"

    def _get_fallback_path(self, job_id: str) -> Path:
        return self.fallback_dir / f"progress_{job_id}.json"

    def _write_fallback(self, job_id: str, data: dict) -> None:
        try:
            fallback_path = self._get_fallback_path(job_id)
            with fallback_path.open("w") as f:
                json.dump(data, f, indent=2)
            logger.warning(f"Progress data written to fallback file: {fallback_path}")
        except Exception as e:
            logger.error(f"Failed to write fallback file for job {job_id}: {e}")

    def start_job(
        self, job_id: str, message: str = "Job started", total_items: int = 0
    ) -> None:
        """Initialize job as status=processing, progress=0%. Falls back to file on RedisError."""
        # Initialize progress_data before try block (for except block reference)
        progress_data = {}
        try:
            # Prepare progress data with initial status
            progress_data = {
                "status": "processing",
                "progress": 0,
                "message": message,
                "current_item": 0,
                "total_items": total_items,
                "stage": "START",
                "eta_seconds": 0,  # Will be calculated during processing
                "memory_mb": 0.0,
                "errors": [],
                "last_heartbeat": datetime.now().isoformat(),
            }

            # Prepare initial history entry
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "progress": 0,
                "message": message,
                "stage": "START",
            }

            # Atomic operation: set progress + push history using pipeline (MULTI/EXEC)
            pipe = self.redis.pipeline()
            pipe.setex(
                self._get_progress_key(job_id),
                self.progress_ttl,
                json.dumps(progress_data),
            )
            pipe.lpush(
                self._get_history_key(job_id), json.dumps(history_entry)
            )
            pipe.ltrim(
                self._get_history_key(job_id), 0, self.max_history_entries - 1
            )
            pipe.expire(self._get_history_key(job_id), self.progress_ttl)
            pipe.execute()

            logger.info(f"Job {job_id} started in Redis")

        except RedisError as e:
            logger.warning(f"Redis error in start_job for {job_id}: {e}")
            self._write_fallback(job_id, progress_data)

    def update_progress(
        self,
        job_id: str,
        progress: int,
        message: str,
        current_item: int = 0,
        total_items: int = 0,
        stage: str = "",
        eta_seconds: int = 0,
        memory_mb: float = 0.0,
        errors: Optional[list[str]] = None,
    ) -> None:
        """Update progress and append to history. Raises ValueError if progress not 0-100."""
        # Validate progress
        if not 0 <= progress <= 100:
            raise ValueError(f"Progress must be 0-100, got {progress}")

        # Initialize progress_data before try block (for except block reference)
        progress_data = {}
        try:
            # Get current progress to preserve some fields
            current = self.get_status(job_id)
            current_status = current["status"] if current else "processing"

            # Prepare updated progress data
            progress_data = {
                "status": current_status,
                "progress": progress,
                "message": message,
                "current_item": current_item,
                "total_items": total_items,
                "stage": stage,
                "eta_seconds": eta_seconds,
                "memory_mb": memory_mb,
                "errors": errors or [],
                "last_heartbeat": datetime.now().isoformat(),
            }

            # Prepare history entry
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "progress": progress,
                "message": message,
                "stage": stage,
            }

            # Atomic operation: update progress + push history
            pipe = self.redis.pipeline()
            pipe.setex(
                self._get_progress_key(job_id),
                self.progress_ttl,
                json.dumps(progress_data),
            )
            pipe.lpush(
                self._get_history_key(job_id), json.dumps(history_entry)
            )
            pipe.ltrim(
                self._get_history_key(job_id),
                0,
                self.max_history_entries - 1,
            )
            pipe.execute()

            logger.debug(f"Job {job_id} progress updated: {progress}%")

        except RedisError as e:
            logger.warning(f"Redis error in update_progress for {job_id}: {e}")
            self._write_fallback(job_id, progress_data)

    def heartbeat(self, job_id: str) -> None:
        """Refresh last_heartbeat timestamp without changing other fields. Call every ~30s during long ops."""
        try:
            # Get current progress
            current = self.get_status(job_id)
            if not current:
                logger.warning(
                    f"Cannot send heartbeat for unknown job {job_id}"
                )
                return

            # Update only heartbeat timestamp
            current["last_heartbeat"] = datetime.now().isoformat()

            # Write back with TTL refresh
            self.redis.setex(
                self._get_progress_key(job_id),
                self.progress_ttl,
                json.dumps(current),
            )

            logger.debug(f"Heartbeat sent for job {job_id}")

        except RedisError as e:
            logger.warning(f"Redis error in heartbeat for {job_id}: {e}")
            # Heartbeat failure is not critical - just log and continue

    def complete_job(self, job_id: str, result: Optional[dict] = None) -> None:
        """Set status=completed, progress=100%, store result (gzip-compressed if >1MB)."""
        # Pre-init so the except branch can reference it even if we crash early.
        progress_data: dict = {}
        try:
            # Prepare final progress data
            progress_data = {
                "status": "completed",
                "progress": 100,
                "message": "Job completed successfully",
                "current_item": 0,
                "total_items": 0,
                "stage": "COMPLETE",
                "eta_seconds": 0,
                "memory_mb": 0.0,
                "errors": [],
                "last_heartbeat": datetime.now().isoformat(),
            }

            # Serialize result
            result_json = json.dumps(result) if result else "{}"

            # Compress if large (>1MB)
            if len(result_json.encode()) > self.compression_threshold_bytes:
                logger.info(
                    f"Compressing result for job {job_id} (size: {len(result_json)} bytes)"
                )
                result_data = gzip.compress(result_json.encode())
                # Store with compression flag
                result_to_store = json.dumps(
                    {"compressed": True, "data": result_data.hex()}
                )
            else:
                result_to_store = result_json

            # Atomic operation: update progress + store result
            pipe = self.redis.pipeline()
            pipe.setex(
                self._get_progress_key(job_id),
                self.progress_ttl,
                json.dumps(progress_data),
            )
            pipe.setex(
                self._get_result_key(job_id),
                self.result_ttl,  # Longer TTL for results
                result_to_store,
            )
            pipe.execute()

            logger.info(f"Job {job_id} marked as completed")

        except RedisError as e:
            logger.warning(f"Redis error in complete_job for {job_id}: {e}")
            self._write_fallback(
                job_id, {"progress": progress_data, "result": result}
            )

    def fail_job(self, job_id: str, error_message: str) -> None:
        """Set status=failed, append error_message to errors list."""
        # Initialize progress_data before try block (for except block reference)
        progress_data = {}
        try:
            # Get current progress to preserve progress/message
            current = self.get_status(job_id)
            current_progress = current["progress"] if current else 0
            current_stage = current.get("stage", "") if current else ""
            current_errors = current.get("errors", []) if current else []

            # Prepare failed progress data
            progress_data = {
                "status": "failed",
                "progress": current_progress,  # Keep current progress
                "message": f"Job failed: {error_message}",
                "current_item": 0,
                "total_items": 0,
                "stage": current_stage,
                "eta_seconds": 0,
                "memory_mb": 0.0,
                "errors": current_errors + [error_message],  # Append error
                "last_heartbeat": datetime.now().isoformat(),
            }

            # Store failed status
            self.redis.setex(
                self._get_progress_key(job_id),
                self.progress_ttl,
                json.dumps(progress_data),
            )

            logger.error(f"Job {job_id} marked as failed: {error_message}")

        except RedisError as e:
            logger.warning(f"Redis error in fail_job for {job_id}: {e}")
            self._write_fallback(job_id, progress_data)

    def get_status(self, job_id: str) -> Optional[dict]:
        """
        Return progress dict for job, or None if not found/expired.

        If status=completed, merges result data (using_ai, ai_model) into returned dict.
        Falls back to file storage on RedisError.
        """
        try:
            # decode_responses=True + sync client → return type narrows to Optional[str].
            # The Redis stub types it as ResponseT (Awaitable | str | bytes | ...) so we cast.
            data = cast(Optional[str], self.redis.get(self._get_progress_key(job_id)))
            if not data:
                return None

            progress_data = json.loads(data)

            # Phase 4: If job is completed, merge result data (includes using_ai, ai_model)
            if progress_data.get("status") == "completed":
                result_data = cast(
                    Optional[str], self.redis.get(self._get_result_key(job_id))
                )
                if result_data:
                    try:
                        result = json.loads(result_data)
                        # Check if result was compressed
                        if isinstance(result, dict) and result.get("compressed"):
                            # Decompress
                            import gzip
                            decompressed = gzip.decompress(bytes.fromhex(result["data"]))
                            result = json.loads(decompressed.decode())

                        # Merge AI matching fields from result into progress_data
                        if isinstance(result, dict):
                            progress_data["using_ai"] = result.get("using_ai", False)
                            progress_data["ai_model"] = result.get("ai_model")
                    except Exception as e:
                        logger.warning(f"Failed to parse result data for {job_id}: {e}")

            return progress_data

        except RedisError as e:
            logger.warning(f"Redis error in get_status for {job_id}: {e}")

            # Try fallback file
            try:
                fallback_path = self._get_fallback_path(job_id)
                if fallback_path.exists():
                    with fallback_path.open("r") as f:
                        return json.load(f)
            except Exception as fallback_err:
                logger.error(
                    f"Failed to read fallback for {job_id}: {fallback_err}"
                )

            return None

    def get_history(self, job_id: str) -> list[dict]:
        """Return last 10 progress history entries (newest first). Returns [] on error."""
        try:
            # Sync client + decode_responses=True returns list[str]; cast for type-checker.
            history_data = cast(
                list[str],
                self.redis.lrange(
                    self._get_history_key(job_id), 0, self.max_history_entries - 1
                ),
            )

            # Deserialize each entry
            return [json.loads(entry) for entry in history_data]

        except RedisError as e:
            logger.warning(f"Redis error in get_history for {job_id}: {e}")
            return []

    def delete_status(self, job_id: str) -> None:
        """Delete all Redis keys for job (progress, result, history)."""
        try:
            # Delete all keys for this job
            pipe = self.redis.pipeline()
            pipe.delete(self._get_progress_key(job_id))
            pipe.delete(self._get_result_key(job_id))
            pipe.delete(self._get_history_key(job_id))
            pipe.execute()

            logger.info(f"Job {job_id} status deleted from Redis")

        except RedisError as e:
            logger.warning(f"Redis error in delete_status for {job_id}: {e}")

    def cleanup_old_jobs(
        self, progress_hours: int = 2, result_hours: int = 48
    ) -> int:
        """Scan Redis for keys without TTL and set them. Returns count of affected keys."""
        try:
            cleaned_count = 0

            # Scan for progress keys
            for key in self.redis.scan_iter(match="progress:*", count=100):
                # Skip history keys
                if ":history" in key:
                    continue

                # Check TTL
                ttl = self.redis.ttl(key)
                if ttl == -1:  # No TTL set (shouldn't happen but handle it)
                    self.redis.expire(key, self.progress_ttl)

                # For manual cleanup: check if data is too old
                # (This would require storing timestamp in data or using key timestamp)
                # For Phase 2 contract, we rely on TTL for now

            # Scan for result keys
            for key in self.redis.scan_iter(match="result:*", count=100):
                ttl = self.redis.ttl(key)
                if ttl == -1:
                    self.redis.expire(key, self.result_ttl)

            logger.info(f"Cleanup completed: {cleaned_count} jobs removed")
            return cleaned_count

        except RedisError as e:
            logger.error(f"Redis error in cleanup_old_jobs: {e}")
            return 0
