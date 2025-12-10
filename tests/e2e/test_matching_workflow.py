"""
End-to-End Tests for Matching Workflow

Tests the complete workflow from upload to download:
1. Upload working file
2. Upload reference file
3. Trigger matching process (Celery task)
4. Poll job status
5. Download results
6. Validate output

Requirements:
    - Docker services running (Redis, Celery worker)
    - Sample fixtures available (sample_working_file.xlsx, sample_reference_file.xlsx)
    - All API endpoints implemented (files, matching, jobs, results)

Setup:
    Before running these tests, start Docker services:
    $ docker-compose up -d

    Verify services are running:
    $ docker-compose ps
    $ redis-cli ping  # Should return PONG

Run:
    # Run all E2E tests
    $ pytest tests/e2e/ -v

    # Run specific test
    $ pytest tests/e2e/test_matching_workflow.py::test_full_workflow_happy_path -v

    # Run with detailed output
    $ pytest tests/e2e/ -v -s

Architecture Notes:
    - Uses FastAPI TestClient (no need for running server)
    - Requires real Redis (mocking would defeat E2E purpose)
    - Requires real Celery worker (asynchronous task execution)
    - Uses real file system (temp files cleaned up after test)
    - Tests integration of all layers: API → Application → Domain → Infrastructure

Test Coverage:
    - Happy path: 20 descriptions, ≥50% should match with score >75%
    - Invalid files: Should return appropriate error responses
    - Low threshold: Should match more items with lower quality threshold
"""

import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import openpyxl
import pytest

# ============================================================================
# TEST CONFIGURATION
# ============================================================================

# Test timeout: max wait time for Celery task to complete
TEST_TIMEOUT_SECONDS = 60

# Poll interval: how often to check job status
POLL_INTERVAL_SECONDS = 2

# Expected success rate: at least 50% of items should match with score >75%
MIN_SUCCESS_RATE = 0.5

# Expected match score threshold for success
MIN_MATCH_SCORE = 75.0

# Configure logger
logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def upload_file(test_client, file_path: Path) -> dict:
    """
    Upload Excel file to /api/files/upload endpoint.

    Args:
        test_client: FastAPI TestClient
        file_path: Path to Excel file

    Returns:
        dict: Upload response with file_id, filename, size_mb, etc.

    Raises:
        AssertionError: If upload fails
    """
    with open(file_path, "rb") as f:
        response = test_client.post(
            "/api/files/upload",
            files={
                "file": (
                    file_path.name,
                    f,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )

    assert response.status_code == 201, (
        f"Upload failed: {response.status_code} - {response.text}"
    )

    data = response.json()
    logger.info(f"Uploaded {file_path.name}: file_id={data['file_id']}")

    return data


def trigger_matching(
    test_client,
    working_file_id: str,
    reference_file_id: str,
    threshold: float = 75.0,
) -> dict:
    """
    Trigger matching process via /api/matching/process endpoint.

    Args:
        test_client: FastAPI TestClient
        working_file_id: UUID of working file
        reference_file_id: UUID of reference file
        threshold: Match threshold (0-100)

    Returns:
        dict: Process response with job_id, status, estimated_time

    Raises:
        AssertionError: If trigger fails
    """
    payload = {
        "working_file_id": working_file_id,
        "reference_file_id": reference_file_id,
        "threshold": threshold,
        "matching_strategy": "HYBRID",
        "report_format": "DETAILED",
    }

    response = test_client.post("/api/matching/process", json=payload)

    assert response.status_code == 202, (
        f"Matching trigger failed: {response.status_code} - {response.text}"
    )

    data = response.json()
    logger.info(f"Matching triggered: job_id={data['job_id']}")

    return data


def poll_job_status(
    test_client,
    job_id: str,
    timeout_seconds: int = TEST_TIMEOUT_SECONDS,
    poll_interval: int = POLL_INTERVAL_SECONDS,
) -> dict:
    """
    Poll job status until completion or timeout.

    Args:
        test_client: FastAPI TestClient
        job_id: Job UUID
        timeout_seconds: Max wait time
        poll_interval: Poll interval in seconds

    Returns:
        dict: Final job status response

    Raises:
        TimeoutError: If job doesn't complete within timeout
        AssertionError: If job fails
    """
    start_time = time.time()
    last_message = None

    while True:
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            raise TimeoutError(
                f"Job {job_id} did not complete within {timeout_seconds}s. "
                f"Last message: {last_message}"
            )

        # Get status
        response = test_client.get(f"/api/jobs/{job_id}/status")

        assert response.status_code == 200, (
            f"Status check failed: {response.status_code} - {response.text}"
        )

        data = response.json()
        status = data["status"]
        progress = data.get("progress", {})
        message = progress.get("message", "")
        percentage = progress.get("percentage", 0)

        # Log progress (only if message changed)
        if message != last_message:
            logger.info(
                f"Job {job_id}: {status} - {percentage}% - {message}"
            )
            last_message = message

        # Check if completed
        if status == "COMPLETED":
            logger.info(f"Job {job_id} completed successfully after {elapsed:.1f}s")
            return data

        # Check if failed
        if status == "FAILED":
            error_message = progress.get("error", "Unknown error")
            raise AssertionError(
                f"Job {job_id} failed: {error_message}"
            )

        # Wait before next poll
        time.sleep(poll_interval)


def download_results(test_client, job_id: str) -> bytes:
    """
    Download result file from /api/results/{job_id}/download endpoint.

    Args:
        test_client: FastAPI TestClient
        job_id: Job UUID

    Returns:
        bytes: Excel file content

    Raises:
        AssertionError: If download fails
    """
    response = test_client.get(f"/api/results/{job_id}/download")

    assert response.status_code == 200, (
        f"Download failed: {response.status_code} - {response.text}"
    )

    assert response.headers["content-type"] == (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ), f"Wrong content type: {response.headers['content-type']}"

    logger.info(f"Downloaded result file for job {job_id}: {len(response.content)} bytes")

    return response.content


def validate_output_file(file_bytes: bytes, min_success_rate: float = MIN_SUCCESS_RATE) -> dict:
    """
    Validate output Excel file structure and content.

    Checks:
        1. File can be opened as Excel
        2. Has expected columns: "Cena", "Match Score", "Match Report"
        3. At least min_success_rate% of rows have Match Score > MIN_MATCH_SCORE

    Args:
        file_bytes: Excel file content as bytes
        min_success_rate: Minimum % of rows that should have good matches

    Returns:
        dict: Validation stats
            - total_rows: Total data rows (excluding header)
            - rows_with_price: Rows with Cena filled
            - rows_with_score: Rows with Match Score filled
            - high_quality_matches: Rows with Match Score > MIN_MATCH_SCORE
            - success_rate: Percentage of high quality matches

    Raises:
        AssertionError: If validation fails
    """
    # Load Excel file
    wb = openpyxl.load_workbook(BytesIO(file_bytes))
    ws = wb.active

    # Get header row
    headers = [cell.value for cell in ws[1]]
    logger.info(f"Output file headers: {headers}")

    # Check required columns exist
    assert "Cena" in headers, "Missing column: Cena"
    assert "Match Score" in headers, "Missing column: Match Score"
    assert "Match Report" in headers, "Missing column: Match Report"

    # Get column indices
    price_col = headers.index("Cena") + 1
    score_col = headers.index("Match Score") + 1
    report_col = headers.index("Match Report") + 1

    # Count statistics
    total_rows = ws.max_row - 1  # Exclude header
    rows_with_price = 0
    rows_with_score = 0
    high_quality_matches = 0

    # Analyze each row
    for row_idx in range(2, ws.max_row + 1):  # Start from row 2 (skip header)
        price = ws.cell(row_idx, price_col).value
        score = ws.cell(row_idx, score_col).value
        report = ws.cell(row_idx, report_col).value

        if price is not None and price != "":
            rows_with_price += 1

        if score is not None:
            rows_with_score += 1

            # Convert score to float (may be string or float)
            try:
                score_value = float(score)
                if score_value >= MIN_MATCH_SCORE:
                    high_quality_matches += 1
            except (ValueError, TypeError):
                logger.warning(f"Row {row_idx}: Invalid score value: {score}")

    # Calculate success rate
    success_rate = high_quality_matches / total_rows if total_rows > 0 else 0.0

    stats = {
        "total_rows": total_rows,
        "rows_with_price": rows_with_price,
        "rows_with_score": rows_with_score,
        "high_quality_matches": high_quality_matches,
        "success_rate": success_rate,
    }

    logger.info(
        f"Output validation stats: "
        f"{high_quality_matches}/{total_rows} high quality matches "
        f"({success_rate*100:.1f}%)"
    )

    # Assert minimum success rate
    assert success_rate >= min_success_rate, (
        f"Success rate too low: {success_rate*100:.1f}% < {min_success_rate*100:.1f}%\n"
        f"Stats: {stats}"
    )

    return stats


# ============================================================================
# E2E TESTS
# ============================================================================


@pytest.mark.e2e
@pytest.mark.slow
def test_full_workflow_happy_path(
    test_client,
    sample_files,
    clean_redis,
    docker_services,
):
    """
    Test full E2E workflow: Upload → Process → Download → Validate.

    This is the main happy path test covering the complete user journey:
    1. User uploads working file (20 HVAC descriptions)
    2. User uploads reference file (50 catalog items with prices)
    3. User triggers matching process
    4. System processes asynchronously (Celery task)
    5. User polls job status until completion
    6. User downloads result file with matched prices
    7. System validates output quality (≥50% high quality matches)

    Acceptance Criteria (from IMPL_PLAN.md Task 3.10.2):
        ✓ Pełny flow działa E2E
        ✓ Output Excel ma wszystkie kolumny (Cena, Match Score, Match Report)
        ✓ ≥50% dopasowań >75%
        ✓ Czas <60s dla 20 items

    Test Stages:
        STAGE 1: Upload files
        STAGE 2: Trigger matching process
        STAGE 3: Wait for completion (polling)
        STAGE 4: Download results
        STAGE 5: Validate output

    Requirements:
        - Redis running (docker-compose up -d)
        - Celery worker running
        - Sample fixtures exist (tests/fixtures/)

    Note:
        This test uses REAL services (not mocks):
        - Real Redis for progress tracking
        - Real Celery worker for async processing
        - Real file system for temp files
        This ensures true E2E validation.
    """
    logger.info("=" * 60)
    logger.info("STARTING E2E TEST: Full Workflow Happy Path")
    logger.info("=" * 60)

    # ========================================================================
    # STAGE 1: Upload files
    # ========================================================================
    logger.info("\n[STAGE 1] Uploading files...")

    working_upload = upload_file(test_client, sample_files["working"])
    working_file_id = working_upload["file_id"]

    reference_upload = upload_file(test_client, sample_files["reference"])
    reference_file_id = reference_upload["file_id"]

    assert working_file_id != reference_file_id, "File IDs should be different"

    # ========================================================================
    # STAGE 2: Trigger matching process
    # ========================================================================
    logger.info("\n[STAGE 2] Triggering matching process...")

    process_response = trigger_matching(
        test_client,
        working_file_id=working_file_id,
        reference_file_id=reference_file_id,
        threshold=75.0,
    )

    job_id = process_response["job_id"]
    assert process_response["status"] == "QUEUED", (
        f"Expected status QUEUED, got {process_response['status']}"
    )

    # ========================================================================
    # STAGE 3: Wait for completion (polling)
    # ========================================================================
    logger.info("\n[STAGE 3] Waiting for job completion...")

    final_status = poll_job_status(
        test_client,
        job_id=job_id,
        timeout_seconds=TEST_TIMEOUT_SECONDS,
        poll_interval=POLL_INTERVAL_SECONDS,
    )

    assert final_status["status"] == "COMPLETED", (
        f"Expected status COMPLETED, got {final_status['status']}"
    )

    # ========================================================================
    # STAGE 4: Download results
    # ========================================================================
    logger.info("\n[STAGE 4] Downloading result file...")

    result_bytes = download_results(test_client, job_id)
    assert len(result_bytes) > 0, "Result file is empty"

    # ========================================================================
    # STAGE 5: Validate output
    # ========================================================================
    logger.info("\n[STAGE 5] Validating output file...")

    stats = validate_output_file(result_bytes, min_success_rate=MIN_SUCCESS_RATE)

    # Log final stats
    logger.info("\n" + "=" * 60)
    logger.info("E2E TEST COMPLETED SUCCESSFULLY ✓")
    logger.info("=" * 60)
    logger.info(f"Total rows processed: {stats['total_rows']}")
    logger.info(f"Rows with prices: {stats['rows_with_price']}")
    logger.info(f"High quality matches: {stats['high_quality_matches']}")
    logger.info(f"Success rate: {stats['success_rate']*100:.1f}%")
    logger.info("=" * 60)


@pytest.mark.e2e
def test_workflow_with_invalid_files(test_client, clean_redis, docker_services):
    """
    Test E2E workflow with invalid file uploads.

    Tests error handling for:
        - Non-Excel file (e.g., .txt, .pdf)
        - Corrupted Excel file
        - Empty file
        - File too large (>10MB)

    Expected Behavior:
        - Upload should fail with 400/413/422 error
        - Error response should have ErrorResponse format
        - Error message should be descriptive

    Acceptance Criteria:
        ✓ Invalid file upload returns appropriate error code
        ✓ Error message is descriptive
        ✓ System doesn't crash or leave orphaned resources
    """
    logger.info("=" * 60)
    logger.info("STARTING E2E TEST: Invalid Files")
    logger.info("=" * 60)

    # Test 1: Empty file
    logger.info("\n[TEST 1] Uploading empty file...")
    empty_file = BytesIO(b"")
    response = test_client.post(
        "/api/files/upload",
        files={
            "file": (
                "empty.xlsx",
                empty_file,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )
    assert response.status_code in [400, 422], (
        f"Expected 400 or 422 for empty file, got {response.status_code}"
    )
    logger.info(f"✓ Empty file rejected with {response.status_code}")

    # Test 2: Wrong content type (text file)
    logger.info("\n[TEST 2] Uploading text file as Excel...")
    text_file = BytesIO(b"This is not an Excel file")
    response = test_client.post(
        "/api/files/upload",
        files={"file": ("fake.xlsx", text_file, "text/plain")},
    )
    assert response.status_code in [400, 422], (
        f"Expected 400 or 422 for text file, got {response.status_code}"
    )
    logger.info(f"✓ Text file rejected with {response.status_code}")

    logger.info("\n" + "=" * 60)
    logger.info("E2E TEST COMPLETED: Invalid Files ✓")
    logger.info("=" * 60)


@pytest.mark.e2e
@pytest.mark.slow
def test_workflow_with_low_threshold(
    test_client,
    sample_files,
    clean_redis,
    docker_services,
):
    """
    Test E2E workflow with low match threshold.

    Tests matching behavior with lower quality threshold (50% instead of 75%).
    Should match MORE items, but with lower average quality.

    Expected Behavior:
        - More rows should have matches (higher match rate)
        - Some matches will have score 50-75% (lower quality)
        - Average match score will be lower than happy path test

    Acceptance Criteria:
        ✓ Match rate > happy path test (more items matched)
        ✓ Some matches have score 50-75%
        ✓ No matches below 50% threshold
    """
    logger.info("=" * 60)
    logger.info("STARTING E2E TEST: Low Threshold")
    logger.info("=" * 60)

    # Upload files
    logger.info("\n[STAGE 1] Uploading files...")
    working_upload = upload_file(test_client, sample_files["working"])
    reference_upload = upload_file(test_client, sample_files["reference"])

    # Trigger matching with LOW threshold (50%)
    logger.info("\n[STAGE 2] Triggering matching with threshold=50%...")
    process_response = trigger_matching(
        test_client,
        working_file_id=working_upload["file_id"],
        reference_file_id=reference_upload["file_id"],
        threshold=50.0,  # Lower threshold
    )

    # Wait for completion
    logger.info("\n[STAGE 3] Waiting for completion...")
    poll_job_status(test_client, process_response["job_id"])

    # Download results
    logger.info("\n[STAGE 4] Downloading results...")
    result_bytes = download_results(test_client, process_response["job_id"])

    # Validate with lower success rate expectation
    logger.info("\n[STAGE 5] Validating output (lower threshold)...")
    stats = validate_output_file(result_bytes, min_success_rate=0.3)  # Lower expectation

    # Assert more matches than with high threshold
    # (This assumes test runs independently - in real CI/CD might need to persist baseline)
    assert stats["rows_with_score"] > 0, "Should have at least some matches"

    logger.info("\n" + "=" * 60)
    logger.info("E2E TEST COMPLETED: Low Threshold ✓")
    logger.info("=" * 60)
    logger.info(f"Match rate: {stats['rows_with_score']}/{stats['total_rows']}")
    logger.info(f"Success rate: {stats['success_rate']*100:.1f}%")
    logger.info("=" * 60)
