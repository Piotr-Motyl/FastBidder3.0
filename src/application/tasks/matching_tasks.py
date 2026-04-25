"""
Celery Task for Asynchronous Matching Process (Phase 2 - Detailed Contract)

Long-running task that processes matching in background.
Updates progress in Redis for frontend consumption.

Responsibility:
    - Execute matching workflow asynchronously
    - Update progress at granular intervals (every 100 records or 10%)
    - Handle errors with exponential backoff retry
    - Save partial results on failure
    - Log each step with timestamp and memory usage
    - Return detailed results including metrics

Architecture Notes:
    - Part of Application Layer (orchestration)
    - Delegates to Domain services for business logic
    - Uses Infrastructure services for I/O operations
    - Thin orchestrator - no business logic here
    - Progress tracking via self.update_state() (Phase 2 minimum)
    - Phase 3 will add dedicated ProgressTracker from Task 2.2.3

Phase 2 Extensions:
    - Support for matching_strategy (FIRST_MATCH, BEST_MATCH, ALL_MATCHES)
    - Support for report_format (SIMPLE, DETAILED, DEBUG)
    - Granular progress updates (every 100 records or 10%, whichever is more frequent)
    - Exponential backoff retry with max 900s delay
    - Partial results saved on failure
    - Detailed logging with memory tracking
    - Extended return dict with processing metrics
"""

import asyncio
import logging
import os
import time
from datetime import datetime

import psutil
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from .celery_app import celery_app
from src.application.models import MatchingStrategy, ReportFormat
from src.application.services.matching_service import ProcessMatchingService
from src.infrastructure.matching.concrete_parameter_extractor import (
    ConcreteParameterExtractor,
)
from src.domain.hvac.services.simple_matching_engine import SimpleMatchingEngine
from src.domain.hvac.matching_config import MatchingConfig
from src.infrastructure.file_storage.excel_reader import ExcelReaderService
from src.infrastructure.file_storage.excel_writer import ExcelWriterService
from src.infrastructure.file_storage.file_storage_service import FileStorageService
from src.infrastructure.persistence.redis.progress_tracker import (
    RedisProgressTracker,
)

# Configure logger for this module
logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="process_matching",
    max_retries=3,
    retry_backoff=True,  # Enable exponential backoff
    retry_backoff_max=900,  # Max 900 seconds (15 minutes) between retries
    time_limit=300,  # 5 minutes hard limit
    soft_time_limit=270,  # Warning 30 seconds before timeout
)
def process_matching_task(
    self: Task,
    working_file: dict,
    reference_file: dict,
    matching_threshold: float = 75.0,
    matching_strategy: str = "best_match",
    report_format: str = "simple",
) -> dict:
    """
    Process HVAC descriptions matching between working and reference files (Phase 2 Contract).

    Orchestrates the entire matching workflow in background with detailed progress tracking,
    error handling, and metrics collection. Supports configurable matching strategies
    and report formats.

    Architecture Pattern:
        - Thin orchestrator pattern
        - Delegates actual work to Domain and Infrastructure services
        - Updates progress for frontend consumption via self.update_state()
        - Handles errors with exponential backoff retry
        - Saves partial results on failure for user recovery

    Args:
        self: Celery task instance (bind=True enables access to self.request.id and self.update_state)
        working_file: Dict with file configuration from WorkingFileConfig:
            - file_id (str): UUID of working file
            - description_column (str): Column with descriptions (e.g., 'C')
            - description_range (dict): {"start": int, "end": int} (1-based Excel notation)
            - price_target_column (str): Column where prices will be written
            - matching_report_column (str, optional): Column for match report
        reference_file: Dict with file configuration from ReferenceFileConfig:
            - file_id (str): UUID of reference file
            - description_column (str): Column with descriptions (e.g., 'B')
            - description_range (dict): {"start": int, "end": int}
            - price_source_column (str): Column with prices to copy
        matching_threshold: Minimum similarity score to accept match (1.0-100.0)
        matching_strategy: Strategy for handling multiple matches:
            - "first_match": Return first match >= threshold (fastest)
            - "best_match": Return highest scoring match (default, balanced)
            - "all_matches": Return all matches >= threshold (comprehensive)
        report_format: Format of matching report in Excel:
            - "simple": Basic match info (score + matched description)
            - "detailed": Include parameter breakdown (DN, PN, material)
            - "debug": Full debug info (all scores, parameters, confidence)

    Returns:
        dict: Task result with detailed metrics (Phase 4: includes AI matching info):
            {
                "status": str,  # "completed" or "failed"
                "job_id": str,  # Celery task ID
                "matches_count": int,  # Number of successful matches
                "processing_time": float,  # Seconds taken
                "result_file_id": str,  # UUID of generated result file
                "matching_strategy_used": str,  # Strategy used for this run
                "report_format_used": str,  # Report format used
                "partial_results": bool,  # True if task failed but partial results saved
                "rows_processed": int,  # Total rows processed
                "rows_matched": int,  # Rows that found a match
                "using_ai": bool,  # Phase 4: True if AI matching (HybridMatchingEngine) was used
                "ai_model": str | None,  # Phase 4: AI model name if AI enabled, None otherwise
                "error": str  # Error message if status="failed" (optional)
            }

    Raises:
        FileNotFoundException: If files not found in storage (will retry with backoff)
        ExcelParsingError: If Excel files cannot be parsed (will retry with backoff)
        MatchingEngineError: If matching process fails (will retry with backoff)
        SoftTimeLimitExceeded: If task runs longer than soft_time_limit (270s)
        TimeLimitExceeded: If task runs longer than time_limit (300s)

    Progress Updates (Phase 2 - Granular):
        Progress is updated at these stages with granular intervals:
        - 0%: Task started (START)
        - 10%: Files loaded from storage (FILES_LOADED)
        - 30%: Descriptions extracted from Excel (DESCRIPTIONS_EXTRACTED)
        - 50%: Parameters extracted (DN, PN, etc.) (PARAMETERS_EXTRACTED)
        - 50-90%: Matching in progress (MATCHING)
            * Updated every 100 records OR 10% of total (whichever is more frequent)
            * Calculation: update_interval = min(100, max(1, total_records // 10))
        - 90%: Results being saved to Excel (SAVING_RESULTS)
        - 100%: Task completed successfully (COMPLETE)

        Update format via self.update_state():
            state='PROCESSING',
            meta={
                'progress': int,  # 0-100 percentage
                'message': str,  # Human-readable status message
                'current_item': int,  # Current record being processed
                'total_items': int,  # Total records to process
                'stage': str  # Current stage name (START, FILES_LOADED, etc.)
            }

    Error Handling (Phase 2):
        - Retry with exponential backoff (60s, 120s, 240s, max 900s)
        - Save partial results on failure:
            * Write Excel with prices for successfully matched rows
            * Leave unmatched rows empty (user can complete manually)
            * Set partial_results=True in return dict
        - Cleanup memory (clear large variables in finally block)
        - Do NOT delete files (cleanup handled by separate job)

    Logging (Phase 2):
        Each step logged with:
        - Timestamp (ISO format)
        - Memory usage (MB) via psutil.Process().memory_info().rss
        - Stage name
        - Records processed count
        - Example: "2025-01-11T10:30:45.123 | 512MB | MATCHING | Processed 450/1000"

    Process Flow with Column Mappings:
        1. START (0%): Initialize services and get job_id from self.request.id
        2. FILES_LOADED (10%): Load Excel files using file_ids
        3. DESCRIPTIONS_EXTRACTED (30%): Extract descriptions from specified columns/ranges:
           - WF: working_file['description_column'][start:end]
           - REF: reference_file['description_column'][start:end]
        4. PARAMETERS_EXTRACTED (50%): Extract HVAC parameters (DN, PN) from descriptions
        5. MATCHING (50-90%): For each WF description, find match in REF:
           - Apply matching_strategy (FIRST_MATCH, BEST_MATCH, ALL_MATCHES)
           - Update progress every 100 records or 10% (whichever more frequent)
           - If match score >= threshold:
             * Copy price from REF[price_source_column] to WF[price_target_column]
             * Write report to WF[matching_report_column] using report_format
        6. SAVING_RESULTS (90%): Save modified WF as result file
        7. COMPLETE (100%): Return success result with metrics

    CONTRACT ONLY - Implementation in Phase 3.
    This is a detailed contract showing expected behavior with Phase 2 extensions.

    Examples:
        >>> # Trigger task with default strategy and format
        >>> result = process_matching_task.delay(
        ...     working_file={"file_id": "abc-123", "description_column": "C", ...},
        ...     reference_file={"file_id": "def-456", "description_column": "B", ...},
        ...     matching_threshold=80.0
        ... )
        >>> result.get(timeout=300)
        {
            'status': 'completed',
            'job_id': '3fa85f64-5717-4562-b3fc-2c963f66afa6',
            'matches_count': 450,
            'processing_time': 45.23,
            'result_file_id': 'result-uuid',
            'matching_strategy_used': 'best_match',
            'report_format_used': 'simple',
            'partial_results': False,
            'rows_processed': 500,
            'rows_matched': 450
        }

        >>> # Trigger with specific strategy and format
        >>> result = process_matching_task.delay(
        ...     working_file={...},
        ...     reference_file={...},
        ...     matching_threshold=75.0,
        ...     matching_strategy="all_matches",
        ...     report_format="detailed"
        ... )
    """
    start_time = time.time()
    job_id = self.request.id
    sys_process = psutil.Process(os.getpid())
    ai_event_loop = None
    using_ai = False
    ai_model = None

    def log_mem(stage: str, message: str) -> None:
        memory_mb = sys_process.memory_info().rss / 1024 / 1024
        logger.info(f"{datetime.now().isoformat()} | {memory_mb:.1f}MB | {stage} | {message}")

    progress_tracker = RedisProgressTracker()

    def update_progress(pct: int, msg: str, stage: str, current: int = 0, total: int = 0) -> None:
        self.update_state(
            state="PROCESSING",
            meta={"progress": pct, "message": msg, "current_item": current,
                  "total_items": total, "stage": stage},
        )
        try:
            progress_tracker.update_progress(
                job_id=job_id, progress=pct, message=msg,
                current_item=current, total_items=total, stage=stage,
                eta_seconds=0, memory_mb=sys_process.memory_info().rss / 1024 / 1024,
                errors=None,
            )
        except Exception as e:
            logger.warning(f"Failed to update Redis progress: {e}")
        log_mem(stage, f"{msg} ({current}/{total})" if total > 0 else msg)

    try:
        progress_tracker.start_job(job_id=job_id, message="Job started", total_items=0)
        update_progress(0, "Task started", "START")

        # ===== Service and engine initialization =====
        excel_reader = ExcelReaderService()
        excel_writer = ExcelWriterService()
        file_storage = FileStorageService()
        parameter_extractor = ConcreteParameterExtractor()
        config = MatchingConfig.default()

        # Validate strategy/format enums (raises ValueError for unknown values)
        MatchingStrategy(matching_strategy)
        ReportFormat(report_format)

        use_ai_matching = os.getenv("USE_AI_MATCHING", "false").lower() == "true"

        if use_ai_matching:
            try:
                from src.infrastructure.ai.embeddings.embedding_service import EmbeddingServiceSingleton
                from src.infrastructure.ai.vector_store.chroma_client import ChromaClientSingleton
                from src.infrastructure.ai.retrieval.semantic_retriever import SemanticRetriever
                from src.infrastructure.matching.hybrid_matching_engine import HybridMatchingEngine

                embedding_service = EmbeddingServiceSingleton.get_instance()
                chroma_client = ChromaClientSingleton.get_instance()
                semantic_retriever = SemanticRetriever(embedding_service, chroma_client)
                simple_engine = SimpleMatchingEngine(parameter_extractor, config, embedding_service)
                matching_engine = HybridMatchingEngine(
                    semantic_retriever=semantic_retriever,
                    simple_matching_engine=simple_engine,
                    config=config,
                    reference_file_id=reference_file["file_id"],
                )
                using_ai = True
                ai_model = embedding_service.model_name
                ai_event_loop = asyncio.new_event_loop()
                logger.info("AI matching enabled: HybridMatchingEngine")
            except Exception as e:
                logger.warning(f"AI init failed, falling back to SimpleMatchingEngine: {e}")
                matching_engine = SimpleMatchingEngine(parameter_extractor, config)
        else:
            matching_engine = SimpleMatchingEngine(parameter_extractor, config)

        # ===== Delegate all matching business logic to ProcessMatchingService =====
        service = ProcessMatchingService(
            matching_engine=matching_engine,
            parameter_extractor=parameter_extractor,
            file_storage=file_storage,
            excel_reader=excel_reader,
            excel_writer=excel_writer,
            using_ai=using_ai,
            ai_model=ai_model,
            ai_event_loop=ai_event_loop,
        )

        result = service.process(
            job_id=job_id,
            working_file=working_file,
            reference_file=reference_file,
            matching_threshold=matching_threshold,
            matching_strategy=matching_strategy,
            report_format=report_format,
            progress_callback=update_progress,
        )

        processing_time = time.time() - start_time
        update_progress(100, "Matching completed successfully", "COMPLETE",
                        result.rows_processed, result.rows_processed)
        progress_tracker.complete_job(
            job_id,
            {"matches_count": result.matches_count,
             "rows_processed": result.rows_processed,
             "rows_matched": result.rows_matched},
        )

        return {
            "status": "completed",
            "job_id": job_id,
            "matches_count": result.matches_count,
            "processing_time": processing_time,
            "result_file_id": job_id,
            "matching_strategy_used": matching_strategy,
            "report_format_used": report_format,
            "partial_results": False,
            "rows_processed": result.rows_processed,
            "rows_matched": result.rows_matched,
            "using_ai": result.using_ai,
            "ai_model": result.ai_model,
        }

    except SoftTimeLimitExceeded:
        logger.warning(f"Job {job_id}: Soft time limit exceeded")
        raise

    except Exception as exc:
        log_mem("ERROR", f"Task failed: {exc}")
        try:
            progress_tracker.fail_job(job_id, str(exc))
        except Exception:
            pass
        raise self.retry(exc=exc)

    finally:
        if ai_event_loop is not None:
            try:
                ai_event_loop.close()
            except Exception as e:
                logger.error(f"Job {job_id}: Failed to close event loop: {e}")
        log_mem("CLEANUP", "Task finished")
