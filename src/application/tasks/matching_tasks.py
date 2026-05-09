"""
Celery task: process_matching.

Thin orchestrator that wires up Domain + Infrastructure services and delegates
the actual matching pipeline to ProcessMatchingService.
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

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="process_matching",
    max_retries=3,
    retry_backoff=True,
    retry_backoff_max=900,  # 15 min between retries
    time_limit=300,
    soft_time_limit=270,  # 30s warning before hard limit
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
    Run HVAC matching between working and reference files.

    Stages: START → FILES_LOADED → DESCRIPTIONS_EXTRACTED → PARAMETERS_EXTRACTED →
    MATCHING (50–90%) → SAVING_RESULTS → COMPLETE.

    Args:
        working_file: Dict with file_id, description_column, description_range,
            price_target_column, optional matching_report_column.
        reference_file: Dict with file_id, description_column, description_range,
            price_source_column.
        matching_threshold: Minimum similarity score (1.0–100.0) to accept match.
        matching_strategy: "first_match" | "best_match" | "all_matches".
        report_format: "simple" | "detailed" | "debug".

    Returns:
        Dict with status, job_id, matches_count, processing_time, result_file_id,
        rows_processed, rows_matched, using_ai, ai_model.

    Raises:
        SoftTimeLimitExceeded: at 270s — re-raised so Celery can fail the task cleanly.
        Other exceptions trigger exponential-backoff retry (max 3 retries, 900s cap).
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

        excel_reader = ExcelReaderService()
        excel_writer = ExcelWriterService()
        file_storage = FileStorageService()
        parameter_extractor = ConcreteParameterExtractor()
        config = MatchingConfig.default()

        # Validates strategy/format strings (raises ValueError on bad value).
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
