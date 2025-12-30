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

import logging
import os
import time
from datetime import datetime
from decimal import Decimal
from uuid import UUID

import psutil
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from .celery_app import celery_app
from src.application.commands.process_matching import ProcessMatchingCommand
from src.application.models import MatchingStrategy, ReportFormat
from src.domain.hvac.entities.hvac_description import HVACDescription
from src.domain.hvac.services.concrete_parameter_extractor import (
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
    # CONTRACT ONLY - Implementation in Phase 3
    #
    # Implementation structure (Phase 3):
    #
    # import os
    # import psutil
    # from datetime import datetime
    # from uuid import UUID
    # from celery.exceptions import SoftTimeLimitExceeded
    # from src.infrastructure.file_storage.excel_reader import ExcelReaderService
    # from src.infrastructure.file_storage.excel_writer import ExcelWriterService
    # from src.infrastructure.file_storage.file_storage_service import FileStorageService
    # from src.infrastructure.matching.matching_engine import ConcreteMatchingEngine
    # from src.domain.hvac.services.parameter_extractor import ParameterExtractor
    # from src.application.models import MatchingStrategy, ReportFormat
    #
    # # Initialize tracking variables
    # start_time = time.time()
    # job_id = self.request.id
    # process = psutil.Process(os.getpid())
    # partial_results = False
    # rows_processed = 0
    # rows_matched = 0
    #
    # # Helper function for logging with memory
    # def log_with_memory(stage: str, message: str):
    #     memory_mb = process.memory_info().rss / 1024 / 1024
    #     timestamp = datetime.now().isoformat()
    #     logger.info(f"{timestamp} | {memory_mb:.1f}MB | {stage} | {message}")
    #
    # # Helper function for progress updates
    # def update_progress(percentage: int, message: str, stage: str, current: int = 0, total: int = 0):
    #     self.update_state(
    #         state='PROCESSING',
    #         meta={
    #             'progress': percentage,
    #             'message': message,
    #             'current_item': current,
    #             'total_items': total,
    #             'stage': stage
    #         }
    #     )
    #     log_with_memory(stage, f"{message} ({current}/{total})" if total > 0 else message)
    #
    # try:
    #     # ===== STAGE 1: START (0%) =====
    #     update_progress(0, 'Task started', 'START')
    #
    #     # Initialize services
    #     excel_reader = ExcelReaderService()
    #     excel_writer = ExcelWriterService()
    #     file_storage = FileStorageService()
    #     matching_engine = ConcreteMatchingEngine()
    #     parameter_extractor = ParameterExtractor()
    #
    #     # Convert strategy and format strings to Enums
    #     strategy = MatchingStrategy(matching_strategy)
    #     format_type = ReportFormat(report_format)
    #
    #     # ===== STAGE 2: FILES_LOADED (10%) =====
    #     update_progress(10, 'Loading files from storage', 'FILES_LOADED')
    #
    #     # Get file paths from storage
    #     wf_path = file_storage.get_file_path(UUID(working_file['file_id']), 'working')
    #     ref_path = file_storage.get_file_path(UUID(reference_file['file_id']), 'reference')
    #
    #     # Load Excel files
    #     wf_data = excel_reader.read_file(wf_path)
    #     ref_data = excel_reader.read_file(ref_path)
    #
    #     # ===== STAGE 3: DESCRIPTIONS_EXTRACTED (30%) =====
    #     update_progress(30, 'Extracting descriptions from Excel', 'DESCRIPTIONS_EXTRACTED')
    #
    #     # Extract descriptions from specified columns/ranges
    #     wf_descriptions = excel_reader.extract_column_range(
    #         wf_data,
    #         working_file['description_column'],
    #         working_file['description_range']
    #     )
    #     ref_descriptions = excel_reader.extract_column_range(
    #         ref_data,
    #         reference_file['description_column'],
    #         reference_file['description_range']
    #     )
    #
    #     total_wf_rows = len(wf_descriptions)
    #     log_with_memory('DESCRIPTIONS_EXTRACTED', f"Extracted {total_wf_rows} WF descriptions, {len(ref_descriptions)} REF descriptions")
    #
    #     # ===== STAGE 4: PARAMETERS_EXTRACTED (50%) =====
    #     update_progress(50, 'Extracting HVAC parameters (DN, PN, etc.)', 'PARAMETERS_EXTRACTED')
    #
    #     # Extract parameters from all descriptions
    #     wf_entities = [parameter_extractor.extract(desc) for desc in wf_descriptions]
    #     ref_entities = [parameter_extractor.extract(desc) for desc in ref_descriptions]
    #
    #     # ===== STAGE 5: MATCHING (50-90%) =====
    #     # Calculate progress update interval: every 100 records OR 10% of total (whichever more frequent)
    #     update_interval = min(100, max(1, total_wf_rows // 10))
    #     log_with_memory('MATCHING', f"Starting matching with interval={update_interval} (every {update_interval} records or 10%)")
    #
    #     matches_count = 0
    #     rows_processed = 0
    #
    #     for wf_idx, wf_entity in enumerate(wf_entities):
    #         # Find match using matching engine
    #         match_result = matching_engine.match(
    #             wf_entity,
    #             ref_entities,
    #             threshold=matching_threshold,
    #             strategy=strategy
    #         )
    #
    #         rows_processed += 1
    #
    #         if match_result:
    #             # Write price to target column
    #             excel_writer.write_cell(
    #                 wf_data,
    #                 row=wf_idx + working_file['description_range']['start'],
    #                 column=working_file['price_target_column'],
    #                 value=match_result.price
    #             )
    #
    #             # Write match report if column specified
    #             if working_file.get('matching_report_column'):
    #                 report = match_result.generate_report(format_type)
    #                 excel_writer.write_cell(
    #                     wf_data,
    #                     row=wf_idx + working_file['description_range']['start'],
    #                     column=working_file['matching_report_column'],
    #                     value=report
    #                 )
    #
    #             matches_count += 1
    #             rows_matched += 1
    #
    #         # Update progress at calculated interval
    #         if (wf_idx + 1) % update_interval == 0 or wf_idx == total_wf_rows - 1:
    #             # Progress from 50% to 90% during matching
    #             progress_pct = 50 + int((wf_idx + 1) / total_wf_rows * 40)
    #             update_progress(
    #                 progress_pct,
    #                 f'Matching descriptions ({matches_count} matched)',
    #                 'MATCHING',
    #                 current=wf_idx + 1,
    #                 total=total_wf_rows
    #             )
    #
    #     # ===== STAGE 6: SAVING_RESULTS (90%) =====
    #     update_progress(90, 'Saving results to Excel file', 'SAVING_RESULTS')
    #
    #     # Save modified working file as result
    #     result_file_id = excel_writer.save(wf_data, job_id)
    #
    #     # ===== STAGE 7: COMPLETE (100%) =====
    #     processing_time = time.time() - start_time
    #     update_progress(100, 'Matching completed successfully', 'COMPLETE', total_wf_rows, total_wf_rows)
    #
    #     # Return success result with metrics
    #     return {
    #         "status": "completed",
    #         "job_id": job_id,
    #         "matches_count": matches_count,
    #         "processing_time": processing_time,
    #         "result_file_id": str(result_file_id),
    #         "matching_strategy_used": matching_strategy,
    #         "report_format_used": report_format,
    #         "partial_results": False,
    #         "rows_processed": rows_processed,
    #         "rows_matched": rows_matched
    #     }
    #
    # except SoftTimeLimitExceeded:
    #     # Soft time limit reached - save partial results and complete gracefully
    #     logger.warning(f"Job {job_id}: Soft time limit exceeded, saving partial results")
    #     partial_results = True
    #
    #     # Save what we have so far
    #     try:
    #         result_file_id = excel_writer.save(wf_data, job_id)
    #         processing_time = time.time() - start_time
    #
    #         return {
    #             "status": "completed",  # Mark as completed but with partial_results=True
    #             "job_id": job_id,
    #             "matches_count": matches_count,
    #             "processing_time": processing_time,
    #             "result_file_id": str(result_file_id),
    #             "matching_strategy_used": matching_strategy,
    #             "report_format_used": report_format,
    #             "partial_results": True,
    #             "rows_processed": rows_processed,
    #             "rows_matched": rows_matched,
    #             "error": "Soft time limit exceeded - partial results saved"
    #         }
    #     except Exception as save_exc:
    #         logger.error(f"Job {job_id}: Failed to save partial results: {save_exc}")
    #         raise
    #
    # except Exception as exc:
    #     # Log error with memory usage
    #     log_with_memory('ERROR', f"Task failed: {str(exc)}")
    #
    #     # Try to save partial results
    #     try:
    #         if rows_processed > 0:
    #             logger.info(f"Job {job_id}: Attempting to save partial results ({rows_processed} rows processed)")
    #             result_file_id = excel_writer.save(wf_data, job_id)
    #             partial_results = True
    #             logger.info(f"Job {job_id}: Partial results saved to {result_file_id}")
    #     except Exception as save_exc:
    #         logger.error(f"Job {job_id}: Failed to save partial results: {save_exc}")
    #         partial_results = False
    #
    #     # Retry with exponential backoff
    #     raise self.retry(exc=exc)
    #
    # finally:
    #     # Cleanup: clear large variables from memory
    #     # Do NOT delete files - that's handled by FileStorageService.cleanup_old_jobs()
    #     try:
    #         if 'wf_data' in locals():
    #             del wf_data
    #         if 'ref_data' in locals():
    #             del ref_data
    #         if 'wf_entities' in locals():
    #             del wf_entities
    #         if 'ref_entities' in locals():
    #             del ref_entities
    #
    #         log_with_memory('CLEANUP', 'Memory cleanup completed')
    #     except Exception as cleanup_exc:
    #         logger.error(f"Job {job_id}: Cleanup failed: {cleanup_exc}")

    # Implementation based on Phase 2 contract
    # Initialize tracking variables
    start_time = time.time()
    job_id = self.request.id
    process = psutil.Process(os.getpid())
    partial_results = False
    rows_processed = 0
    rows_matched = 0

    # Helper function for logging with memory
    def log_with_memory(stage: str, message: str):
        memory_mb = process.memory_info().rss / 1024 / 1024
        timestamp = datetime.now().isoformat()
        logger.info(f"{timestamp} | {memory_mb:.1f}MB | {stage} | {message}")

    # Initialize RedisProgressTracker BEFORE try block
    # This ensures it's available in update_progress() closure and except/finally blocks
    progress_tracker = RedisProgressTracker()

    # Helper function for progress updates (uses both Celery native + RedisProgressTracker)
    def update_progress(
        percentage: int, message: str, stage: str, current: int = 0, total: int = 0
    ):
        # Update Celery native state (for Flower monitoring)
        self.update_state(
            state="PROCESSING",
            meta={
                "progress": percentage,
                "message": message,
                "current_item": current,
                "total_items": total,
                "stage": stage,
            },
        )

        # Update RedisProgressTracker (for our API /jobs/{job_id}/status)
        try:
            progress_tracker.update_progress(
                job_id=job_id,
                progress=percentage,
                message=message,
                current_item=current,
                total_items=total,
                stage=stage,
                eta_seconds=0,  # TODO: calculate ETA based on processing rate
                memory_mb=process.memory_info().rss / 1024 / 1024,
                errors=None,
            )
        except Exception as e:
            # Log error but don't fail task if progress tracking fails
            logger.warning(f"Failed to update progress in Redis: {e}")

        log_with_memory(
            stage, f"{message} ({current}/{total})" if total > 0 else message
        )

    # Initialize variables before try block to avoid UnboundLocalError in except/finally
    file_storage = None
    excel_writer = None
    excel_reader = None
    wf_df = None
    ref_df = None
    wf_descriptions = None
    ref_descriptions = None
    matches_count = 0
    rows_processed = 0
    rows_matched = 0
    partial_results = False
    start_time = time.time()

    try:
        # ===== STAGE 0: Initialize job in Redis =====
        # Start job tracking in Redis so API can query status immediately
        progress_tracker.start_job(
            job_id=job_id,
            message="Job started - initializing services",
            total_items=0,  # Will be updated after loading files
        )
        logger.info(f"Job {job_id} started in Redis progress tracker")

        # ===== STAGE 1: START (0%) =====
        update_progress(0, "Task started", "START")

        # Initialize services
        excel_reader = ExcelReaderService()
        excel_writer = ExcelWriterService()
        file_storage = FileStorageService()
        parameter_extractor = ConcreteParameterExtractor()

        # Create matching config
        config = MatchingConfig.default()

        # ===== AI MATCHING SETUP (Phase 4) =====
        # Check if AI matching is enabled via environment variable
        use_ai_matching = os.getenv("USE_AI_MATCHING", "false").lower() == "true"
        using_ai = False
        ai_model = None

        if use_ai_matching:
            try:
                # Try to initialize HybridMatchingEngine with AI components
                logger.info("USE_AI_MATCHING=true detected - initializing AI matching pipeline")

                from src.infrastructure.ai.embeddings.embedding_service import (
                    EmbeddingService,
                )
                from src.infrastructure.ai.vector_store.chroma_client import (
                    ChromaClientSingleton,
                )
                from src.infrastructure.ai.retrieval.semantic_retriever import (
                    SemanticRetriever,
                )
                from src.infrastructure.matching.hybrid_matching_engine import (
                    HybridMatchingEngine,
                )

                # Initialize AI services for two-stage pipeline
                embedding_service = EmbeddingService()
                chroma_client = ChromaClientSingleton.get_instance()
                semantic_retriever = SemanticRetriever(embedding_service, chroma_client)

                # Create SimpleMatchingEngine with AI embeddings for Stage 2
                simple_engine = SimpleMatchingEngine(
                    parameter_extractor, config, embedding_service
                )

                # Create HybridMatchingEngine (two-stage pipeline: retrieval + scoring)
                matching_engine = HybridMatchingEngine(
                    semantic_retriever=semantic_retriever,
                    simple_matching_engine=simple_engine,
                    config=config,
                )

                using_ai = True
                ai_model = "paraphrase-multilingual-MiniLM-L12-v2"  # From EmbeddingService
                logger.info(
                    "AI matching enabled: Using HybridMatchingEngine (Stage 1: Retrieval, Stage 2: Scoring)"
                )

            except Exception as e:
                # Fallback to SimpleMatchingEngine if AI initialization fails
                logger.warning(
                    f"Failed to initialize AI matching, falling back to SimpleMatchingEngine: {e}"
                )
                matching_engine = SimpleMatchingEngine(parameter_extractor, config)
                using_ai = False
                ai_model = None
        else:
            # AI matching disabled - use SimpleMatchingEngine with placeholder semantic scores
            logger.info(
                "AI matching disabled (USE_AI_MATCHING=false): Using SimpleMatchingEngine with placeholder semantic scores"
            )
            matching_engine = SimpleMatchingEngine(parameter_extractor, config)
            using_ai = False
            ai_model = None

        # Convert strategy and format strings to Enums
        strategy = MatchingStrategy(matching_strategy)
        format_type = ReportFormat(report_format)

        # ===== STAGE 2: FILES_LOADED (10%) =====
        update_progress(10, "Loading files from storage", "FILES_LOADED")

        # Get file paths from UPLOAD storage (not job storage)
        # Files are in /tmp/fastbidder/uploads/{file_id}/ after upload
        wf_file_id = UUID(working_file["file_id"])
        ref_file_id = UUID(reference_file["file_id"])

        # Get upload directories
        wf_upload_dir = file_storage.get_uploaded_file_path(wf_file_id)
        ref_upload_dir = file_storage.get_uploaded_file_path(ref_file_id)

        # Find the uploaded files (original filename is preserved in upload storage)
        wf_files = list(wf_upload_dir.glob("*.xlsx"))
        ref_files = list(ref_upload_dir.glob("*.xlsx"))

        if not wf_files:
            raise FileNotFoundError(
                f"No working file found in upload directory: {wf_upload_dir}"
            )
        if not ref_files:
            raise FileNotFoundError(
                f"No reference file found in upload directory: {ref_upload_dir}"
            )

        wf_path = wf_files[0]
        ref_path = ref_files[0]

        # Load Excel DataFrames
        wf_df = excel_reader.read_excel_to_dataframe(wf_path)
        ref_df = excel_reader.read_excel_to_dataframe(ref_path)

        # ===== STAGE 3: DESCRIPTIONS_EXTRACTED (30%) =====
        update_progress(
            30, "Extracting descriptions from Excel", "DESCRIPTIONS_EXTRACTED"
        )

        # Extract descriptions from working file
        wf_col_idx = ProcessMatchingCommand.column_to_index(
            working_file["description_column"]
        )
        wf_range_start = working_file["description_range"]["start"] - 1  # 0-based
        wf_range_end = working_file["description_range"]["end"]
        wf_raw_texts = wf_df.iloc[wf_range_start:wf_range_end, wf_col_idx].tolist()

        # Extract descriptions from reference file
        ref_col_idx = ProcessMatchingCommand.column_to_index(
            reference_file["description_column"]
        )
        ref_range_start = reference_file["description_range"]["start"] - 1
        ref_range_end = reference_file["description_range"]["end"]
        ref_raw_texts = ref_df.iloc[ref_range_start:ref_range_end, ref_col_idx].tolist()

        # Extract prices from reference file
        ref_price_col_idx = ProcessMatchingCommand.column_to_index(
            reference_file["price_source_column"]
        )
        ref_prices = ref_df.iloc[
            ref_range_start:ref_range_end, ref_price_col_idx
        ].tolist()

        total_wf_rows = len(wf_raw_texts)
        log_with_memory(
            "DESCRIPTIONS_EXTRACTED",
            f"Extracted {total_wf_rows} WF descriptions, {len(ref_raw_texts)} REF descriptions",
        )

        # ===== STAGE 4: PARAMETERS_EXTRACTED (50%) =====
        update_progress(
            50, "Extracting HVAC parameters (DN, PN, etc.)", "PARAMETERS_EXTRACTED"
        )

        # Initialize output columns in DataFrame (before matching loop)
        # This ensures columns exist even if no matches are found
        target_col_idx = ProcessMatchingCommand.column_to_index(
            working_file["price_target_column"]
        )
        # Ensure DataFrame has enough columns
        while len(wf_df.columns) <= target_col_idx:
            wf_df[f"Column_{len(wf_df.columns)}"] = ""

        # Set column name for price target (column B)
        wf_df.columns.values[target_col_idx] = "Cena"

        # Add Match Score column (column C - after price)
        score_col_idx = target_col_idx + 1
        while len(wf_df.columns) <= score_col_idx:
            wf_df[f"Column_{len(wf_df.columns)}"] = ""
        wf_df.columns.values[score_col_idx] = "Match Score"

        # Add matching report column if specified (column D - after score)
        report_col_idx = None  # Initialize to prevent unbound variable warning
        if working_file.get("matching_report_column"):
            report_col_idx = ProcessMatchingCommand.column_to_index(
                working_file["matching_report_column"]
            )
            # Ensure DataFrame has enough columns
            while len(wf_df.columns) <= report_col_idx:
                wf_df[f"Column_{len(wf_df.columns)}"] = ""

            # Set column name for match report
            wf_df.columns.values[report_col_idx] = "Match Report"

        logger.info(
            f"Initialized output columns: Cena (col {target_col_idx}), "
            f"Match Score (col {score_col_idx}), "
            f"Match Report (col {report_col_idx if working_file.get('matching_report_column') else 'N/A'})"
        )

        # Create HVACDescription entities and extract parameters
        wf_descriptions = []
        for idx, text in enumerate(wf_raw_texts):
            desc = HVACDescription(raw_text=str(text) if text else "")
            desc.extract_parameters(parameter_extractor)
            wf_descriptions.append(desc)

        ref_descriptions = []
        for idx, (text, price) in enumerate(zip(ref_raw_texts, ref_prices)):
            desc = HVACDescription(raw_text=str(text) if text else "")
            desc.extract_parameters(parameter_extractor)
            # Set price directly as field (HVACDescription is a dataclass)
            if price and price != "":
                desc.matched_price = Decimal(str(price))
            ref_descriptions.append(desc)

        # ===== STAGE 5: MATCHING (50-90%) =====
        update_interval = min(100, max(1, total_wf_rows // 10))
        log_with_memory(
            "MATCHING",
            f"Starting matching with interval={update_interval} (threshold={matching_threshold})",
        )

        matches_count = 0
        rows_processed = 0

        # Import asyncio for HybridMatchingEngine async calls
        import asyncio

        for wf_idx, wf_desc in enumerate(wf_descriptions):
            # Find match using matching engine (handle both sync and async engines)
            if using_ai:
                # HybridMatchingEngine: async interface, reference_descriptions not used (uses vector DB)
                match_result = asyncio.run(
                    matching_engine.match(
                        working_description=wf_desc,
                        reference_descriptions=[],  # Not used in hybrid mode
                        threshold=matching_threshold,
                    )
                )
            else:
                # SimpleMatchingEngine: sync interface
                match_result = matching_engine.match_single(
                    wf_desc, ref_descriptions, matching_threshold
                )

            rows_processed += 1

            if match_result:
                # Apply match result to working description
                wf_desc.apply_match_result(match_result)

                # Find matched reference description by ID
                matched_ref_desc = next(
                    (
                        desc
                        for desc in ref_descriptions
                        if desc.id == match_result.matched_reference_id
                    ),
                    None,
                )

                if matched_ref_desc:
                    # Write price to target column (Excel 1-based row)
                    target_row = wf_range_start + wf_idx
                    target_col_idx = ProcessMatchingCommand.column_to_index(
                        working_file["price_target_column"]
                    )
                    # Get price from matched reference (convert Decimal to float for Excel)
                    price_value = (
                        float(matched_ref_desc.matched_price)
                        if matched_ref_desc.matched_price
                        else ""
                    )
                    wf_df.iloc[target_row, target_col_idx] = price_value

                    # Write match score to Match Score column (column C)
                    score_col_idx = target_col_idx + 1
                    wf_df.iloc[target_row, score_col_idx] = (
                        match_result.score.final_score
                    )

                    # Write match report if column specified
                    if working_file.get("matching_report_column"):
                        # Use score.final_score (not total_score)
                        report_text = f"Match: {matched_ref_desc.raw_text[:50]}... | Score: {match_result.score.final_score:.1f}%"
                        report_col_idx = ProcessMatchingCommand.column_to_index(
                            working_file["matching_report_column"]
                        )
                        wf_df.iloc[target_row, report_col_idx] = report_text

                    matches_count += 1
                    rows_matched += 1

            # Update progress at calculated interval
            if (wf_idx + 1) % update_interval == 0 or wf_idx == total_wf_rows - 1:
                progress_pct = 50 + int((wf_idx + 1) / total_wf_rows * 40)
                update_progress(
                    progress_pct,
                    f"Matching descriptions ({matches_count} matched)",
                    "MATCHING",
                    current=wf_idx + 1,
                    total=total_wf_rows,
                )

        # ===== STAGE 6: SAVING_RESULTS (90%) =====
        update_progress(90, "Saving results to Excel file", "SAVING_RESULTS")

        # Save modified working file as result
        result_path = file_storage.get_result_file_path(UUID(job_id))
        excel_writer.save_dataframe_to_excel(wf_df, result_path)

        # ===== STAGE 7: COMPLETE (100%) =====
        processing_time = time.time() - start_time
        update_progress(
            100,
            "Matching completed successfully",
            "COMPLETE",
            total_wf_rows,
            total_wf_rows,
        )

        # Update Redis progress tracker
        progress_tracker.complete_job(
            job_id,
            {
                "matches_count": matches_count,
                "rows_processed": rows_processed,
                "rows_matched": rows_matched,
            },
        )

        # Return success result with metrics (Phase 4: includes AI matching info)
        return {
            "status": "completed",
            "job_id": job_id,
            "matches_count": matches_count,
            "processing_time": processing_time,
            "result_file_id": job_id,  # Result file uses job_id
            "matching_strategy_used": matching_strategy,
            "report_format_used": report_format,
            "partial_results": False,
            "rows_processed": rows_processed,
            "rows_matched": rows_matched,
            "using_ai": using_ai,  # Phase 4: AI matching enabled
            "ai_model": ai_model,  # Phase 4: AI model name if AI enabled
        }

    except SoftTimeLimitExceeded:
        # Soft time limit reached - save partial results
        logger.warning(
            f"Job {job_id}: Soft time limit exceeded, saving partial results"
        )
        partial_results = True

        # Only save if services were initialized
        if file_storage is not None and excel_writer is not None and wf_df is not None:
            try:
                result_path = file_storage.get_result_file_path(UUID(job_id))
                excel_writer.save_dataframe_to_excel(wf_df, result_path)
                processing_time = time.time() - start_time

                return {
                    "status": "completed",
                    "job_id": job_id,
                    "matches_count": matches_count,
                    "processing_time": processing_time,
                    "result_file_id": job_id,
                    "matching_strategy_used": matching_strategy,
                    "report_format_used": report_format,
                    "partial_results": True,
                    "rows_processed": rows_processed,
                    "rows_matched": rows_matched,
                    "error": "Soft time limit exceeded - partial results saved",
                    "using_ai": using_ai,  # Phase 4: AI matching info
                    "ai_model": ai_model,  # Phase 4: AI model name
                }
            except Exception as save_exc:
                logger.error(
                    f"Job {job_id}: Failed to save partial results: {save_exc}"
                )
                raise
        else:
            # Services not initialized, cannot save
            logger.error(
                f"Job {job_id}: Cannot save partial results - services not initialized"
            )
            raise

    except Exception as exc:
        # Log error with memory usage
        log_with_memory("ERROR", f"Task failed: {str(exc)}")

        # Try to save partial results (only if services were initialized)
        try:
            if (
                rows_processed > 0
                and wf_df is not None
                and file_storage is not None
                and excel_writer is not None
            ):
                logger.info(
                    f"Job {job_id}: Attempting to save partial results ({rows_processed} rows processed)"
                )
                result_path = file_storage.get_result_file_path(UUID(job_id))
                excel_writer.save_dataframe_to_excel(wf_df, result_path)
                partial_results = True
                logger.info(f"Job {job_id}: Partial results saved")
        except Exception as save_exc:
            logger.error(f"Job {job_id}: Failed to save partial results: {save_exc}")
            partial_results = False

        # Update progress tracker with failure (only if tracker was initialized)
        try:
            if progress_tracker is not None:
                progress_tracker.fail_job(job_id, str(exc))
        except Exception:
            pass

        # Retry with exponential backoff
        raise self.retry(exc=exc)

    finally:
        # Cleanup: clear large variables from memory
        try:
            if wf_df is not None:
                del wf_df
            if ref_df is not None:
                del ref_df
            if wf_descriptions is not None:
                del wf_descriptions
            if ref_descriptions is not None:
                del ref_descriptions

            log_with_memory("CLEANUP", "Memory cleanup completed")
        except Exception as cleanup_exc:
            logger.error(f"Job {job_id}: Cleanup failed: {cleanup_exc}")
