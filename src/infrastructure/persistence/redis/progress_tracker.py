"""
Redis Progress Tracker (Phase 2 - Detailed Contract)

Tracks Celery job progress in Redis for real-time status updates.
Used by Application Layer to update and query job status.

Responsibility:
    - Store job progress in Redis with extended metadata
    - Maintain progress history (last 10 updates)
    - Heartbeat tracking to show task is alive
    - Error recovery with file fallback
    - Compression for large results
    - TTL management with different timeouts for progress vs results
    - Atomic operations for data consistency

Architecture Notes:
    - Infrastructure Layer (external dependency on Redis)
    - Used by Celery tasks and Application Layer
    - Phase 2: Extended progress data, history, heartbeat, fallback
    - Phase 3: Will implement actual Redis operations

Phase 2 Extensions:
    - Extended progress data (eta, memory, errors)
    - History tracking (last 10 updates)
    - Heartbeat mechanism
    - Error recovery with file fallback
    - Compression for large results (>1MB)
    - Cleanup method for old jobs
    - MULTI/EXEC for atomic operations
"""

import gzip
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from redis import Redis
from redis.exceptions import RedisError

# Configure logger for this module
logger = logging.getLogger(__name__)


class RedisProgressTracker:
    """
    Track Celery job progress using Redis with extended features (Phase 2 Contract).

    This class provides interface for storing and retrieving job progress information
    with extended metadata, history tracking, and error recovery.

    Note on job_id parameter type:
        All methods use job_id as string (not UUID) for Redis compatibility.
        Redis keys must be strings, so we use str(job_id) throughout.
        Application Layer passes UUID, which is converted to string here.

    Storage Format (Phase 2 - Extended):
        Redis keys:
        - "progress:{job_id}" -> JSON dict with progress data
        - "result:{job_id}" -> JSON dict with final result (optionally gzipped)
        - "progress:{job_id}:history" -> Redis LIST with last 10 progress updates

        Progress data structure:
        {
            "status": "processing",       # queued/processing/completed/failed
            "progress": 45,                # 0-100 percentage
            "message": "Matching...",      # Human-readable current step
            "current_item": 450,           # Current record being processed
            "total_items": 1000,           # Total records to process
            "stage": "MATCHING",           # Current stage (START, FILES_LOADED, MATCHING, etc.)
            "eta_seconds": 120,            # Estimated time to completion
            "memory_mb": 512.5,            # Memory usage in MB
            "errors": [],                  # List of error messages
            "last_heartbeat": "2025-01-11T10:30:45.123"  # ISO timestamp
        }

        History entry structure (in Redis LIST):
        {
            "timestamp": "2025-01-11T10:30:45.123",  # ISO timestamp
            "progress": 45,                           # 0-100 percentage
            "message": "Matching descriptions...",
            "stage": "MATCHING"  # Current stage name
        }

    Business Rules:
        - TTL progress: 1 hour (3600s) - auto-expire after 1h
        - TTL result: 24 hours (86400s) - results kept longer
        - History: Max 10 entries (FIFO, oldest dropped)
        - Heartbeat: Update every 30s to show task is alive
        - Compression: Gzip results >1MB before storing
        - Atomic: MULTI/EXEC for operations modifying >1 key

    Error Recovery:
        - If Redis fails, fallback to file storage
        - Fallback path: /tmp/fastbidder/fallback/progress_{job_id}.json
        - Log warning but do NOT raise exception (graceful degradation)

    Phase 2 Scope:
        - Full contract with all Phase 2 extensions
        - Detailed docstrings for Phase 3 implementation
        - Error handling patterns
        - Compression logic
        - Cleanup mechanism

    Examples:
        >>> # In Celery task - start job
        >>> tracker = RedisProgressTracker()
        >>> tracker.start_job("job-123", "Starting matching process")

        >>> # Update progress during execution
        >>> tracker.update_progress(
        ...     "job-123",
        ...     progress=50,
        ...     message="Matching descriptions",
        ...     current_item=500,
        ...     total_items=1000,
        ...     stage="MATCHING"
        ... )

        >>> # Send heartbeat
        >>> tracker.heartbeat("job-123")

        >>> # Complete job
        >>> tracker.complete_job("job-123", {"matches_count": 950, "result_file_id": "..."})

        >>> # In Application Layer - get status
        >>> status = tracker.get_status("job-123")
        >>> print(status["progress"])  # 100
    """

    def __init__(
        self,
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
        redis_db: int = 0,
    ) -> None:
        """
        Initialize Redis connection for progress tracking.

        Args:
            redis_host: Redis hostname (default from env: REDIS_HOST)
            redis_port: Redis port (default from env: REDIS_PORT)
            redis_db: Redis database number (default 0)

        Raises:
            RedisError: If connection cannot be established
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
        self.fallback_dir = Path(os.getenv("FALLBACK_DIR", "/tmp/fastbidder/fallback"))
        self.fallback_dir.mkdir(parents=True, exist_ok=True)

        # History configuration (Phase 2)
        self.max_history_entries: int = 10

        # Compression threshold (Phase 2)
        self.compression_threshold_bytes: int = 1024 * 1024  # 1MB

    def _get_progress_key(self, job_id: str) -> str:
        """
        Generate Redis key for job progress.

        Args:
            job_id: Unique job identifier (UUID string)

        Returns:
            Redis key in format "progress:{job_id}"

        Examples:
            >>> tracker._get_progress_key("abc-123")
            'progress:abc-123'
        """
        return f"progress:{job_id}"

    def _get_result_key(self, job_id: str) -> str:
        """
        Generate Redis key for job result.

        Args:
            job_id: Unique job identifier

        Returns:
            Redis key in format "result:{job_id}"

        Examples:
            >>> tracker._get_result_key("abc-123")
            'result:abc-123'
        """
        return f"result:{job_id}"

    def _get_history_key(self, job_id: str) -> str:
        """
        Generate Redis key for job history.

        Args:
            job_id: Unique job identifier

        Returns:
            Redis key in format "progress:{job_id}:history"

        Examples:
            >>> tracker._get_history_key("abc-123")
            'progress:abc-123:history'
        """
        return f"progress:{job_id}:history"

    def _get_fallback_path(self, job_id: str) -> Path:
        """
        Get fallback file path for job when Redis is unavailable.

        Args:
            job_id: Unique job identifier

        Returns:
            Path to fallback JSON file

        Examples:
            >>> tracker._get_fallback_path("abc-123")
            Path('/tmp/fastbidder/fallback/progress_abc-123.json')
        """
        return self.fallback_dir / f"progress_{job_id}.json"

    def _write_fallback(self, job_id: str, data: dict) -> None:
        """
        Write progress data to fallback file when Redis fails.

        Args:
            job_id: Unique job identifier
            data: Progress data to write

        Examples:
            >>> tracker._write_fallback("job-123", {"status": "processing", "progress": 50})
        """
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
        """
        Initialize job status in Redis (Phase 2 - Extended).

        Sets initial status to "processing" with 0% progress and initializes
        extended metadata fields.

        Uses MULTI/EXEC for atomic initialization of:
        - Progress data (progress:{job_id})
        - History with initial entry (progress:{job_id}:history)

        Args:
            job_id: Unique job identifier
            message: Initial status message (default: "Job started")
            total_items: Total items to process (0 if unknown)

        Error Handling:
            - On RedisError: Write to fallback file and log warning
            - Do NOT raise exception (graceful degradation)

        Examples:
            >>> tracker = RedisProgressTracker()
            >>> tracker.start_job("job-123", "Starting matching process", total_items=1000)
        """
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
        """
        Update job progress with extended metadata (Phase 2 - Full).

        Updates progress data and adds entry to history.
        Uses MULTI/EXEC for atomic updates.

        Args:
            job_id: Unique job identifier
            progress: Progress percentage (0-100)
            message: Current step description
            current_item: Current record being processed
            total_items: Total records to process
            stage: Current stage name (START, MATCHING, etc.)
            eta_seconds: Estimated time to completion
            memory_mb: Memory usage in MB
            errors: List of error messages (optional)

        Raises:
            ValueError: If progress not in range 0-100

        Error Handling:
            - On RedisError: Write to fallback file and log warning
            - Do NOT raise exception (graceful degradation)

        Examples:
            >>> tracker.update_progress(
            ...     "job-123",
            ...     progress=50,
            ...     message="Matching descriptions",
            ...     current_item=500,
            ...     total_items=1000,
            ...     stage="MATCHING",
            ...     eta_seconds=120,
            ...     memory_mb=512.5
            ... )
        """
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
        """
        Send heartbeat to show task is alive (Phase 2 - New).

        Updates last_heartbeat timestamp and refreshes TTL without changing
        other progress data. Used to show task is running even if progress
        hasn't changed.

        Should be called every ~30 seconds during long-running operations.

        Args:
            job_id: Unique job identifier

        Error Handling:
            - On RedisError: Log warning but do NOT raise
            - Heartbeat failure is not critical

        Examples:
            >>> # In Celery task during long operation
            >>> tracker.heartbeat("job-123")  # Call every 30s
        """
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
        """
        Mark job as completed with optional result (Phase 2 - Extended).

        Sets status to "completed", progress to 100%, and stores result.
        Uses compression for large results (>1MB).
        Uses MULTI/EXEC for atomic updates.

        Args:
            job_id: Unique job identifier
            result: Optional result data (e.g., matched count, file path)
                Will be compressed if size >1MB

        Error Handling:
            - On RedisError: Write to fallback file and log warning
            - Do NOT raise exception (graceful degradation)

        Examples:
            >>> tracker.complete_job(
            ...     "job-123",
            ...     {
            ...         "status": "completed",
            ...         "matches_count": 950,
            ...         "result_file_id": "result-uuid",
            ...         "processing_time": 45.2
            ...     }
            ... )
        """
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
        """
        Mark job as failed with error message.

        Sets status to "failed" and adds error to errors list.

        Args:
            job_id: Unique job identifier
            error_message: Error description for debugging

        Error Handling:
            - On RedisError: Write to fallback file and log warning

        Examples:
            >>> tracker.fail_job(
            ...     "job-123",
            ...     "File not found: working_file.xlsx"
            ... )
        """
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
        Retrieve current job status from Redis.

        Returns full progress data with extended metadata.

        Args:
            job_id: Unique job identifier

        Returns:
            Progress dict if found, None if job not found or expired

        Error Handling:
            - On RedisError: Try to read from fallback file
            - If both fail: return None

        Examples:
            >>> status = tracker.get_status("job-123")
            >>> if status:
            ...     print(f"Progress: {status['progress']}%")
            ...     print(f"Stage: {status['stage']}")
            ...     print(f"ETA: {status['eta_seconds']}s")
            ...     print(f"Memory: {status['memory_mb']}MB")
        """
        try:
            # Get progress data from Redis
            data = self.redis.get(self._get_progress_key(job_id))
            if not data:
                return None

            return json.loads(data)

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
        """
        Retrieve progress history for job (Phase 2 - New).

        Returns last 10 progress updates for debugging.

        Args:
            job_id: Unique job identifier

        Returns:
            List of history entries (newest first), empty list if none

        Examples:
            >>> history = tracker.get_history("job-123")
            >>> for entry in history:
            ...     print(f"{entry['timestamp']}: {entry['progress']}% - {entry['stage']} - {entry['message']}")
        """
        try:
            # Get history list from Redis (LRANGE returns list of JSON strings)
            history_data = self.redis.lrange(
                self._get_history_key(job_id), 0, self.max_history_entries - 1
            )

            # Deserialize each entry
            return [json.loads(entry) for entry in history_data]

        except RedisError as e:
            logger.warning(f"Redis error in get_history for {job_id}: {e}")
            return []

    def delete_status(self, job_id: str) -> None:
        """
        Delete job status, result, and history from Redis (manual cleanup).

        Deletes all keys associated with job:
        - progress:{job_id}
        - result:{job_id}
        - progress:{job_id}:history

        Args:
            job_id: Unique job identifier

        Examples:
            >>> tracker.delete_status("job-123")
        """
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
        """
        Cleanup old progress and result data (Phase 2 - New).

        Scans Redis for expired jobs and removes them.
        Should be called periodically (e.g., hourly cron job).

        Note: This is in addition to TTL auto-expiration. Useful for:
        - Different cleanup thresholds than TTL
        - Manual cleanup trigger
        - Cleanup of jobs that might have missed TTL

        Args:
            progress_hours: Delete progress data older than N hours (default 2h)
            result_hours: Delete result data older than N hours (default 48h)

        Returns:
            Number of jobs cleaned up

        Examples:
            >>> # Cleanup progress >2h old, results >48h old
            >>> count = tracker.cleanup_old_jobs(progress_hours=2, result_hours=48)
            >>> print(f"Cleaned up {count} old jobs")
        """
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
