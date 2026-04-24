"""
Unit tests for ProcessMatchingService.

Verifies matching pipeline logic without real Celery worker, Redis, or Excel files.
All infrastructure dependencies are mocked.
"""
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import polars as pl
import pytest

from src.application.services.matching_service import MatchingResult, ProcessMatchingService
from src.domain.hvac.value_objects.match_result import MatchResult
from src.domain.hvac.value_objects.match_score import MatchScore


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_service(**overrides):
    """Create ProcessMatchingService with all deps mocked, apply overrides."""
    defaults = dict(
        matching_engine=MagicMock(),
        parameter_extractor=MagicMock(),
        file_storage=MagicMock(),
        excel_reader=MagicMock(),
        excel_writer=MagicMock(),
        using_ai=False,
        ai_model=None,
        ai_event_loop=None,
    )
    defaults.update(overrides)
    return ProcessMatchingService(**defaults)


def _make_wf_df():
    """Three-column working DataFrame: A, B, C with 3 data rows."""
    return pl.DataFrame({"A": ["desc1", "desc2", "desc3"], "B": [None, None, None], "C": [None, None, None]})


def _make_ref_df():
    """Two-column reference DataFrame: A (description), B (price)."""
    return pl.DataFrame({"A": ["ref1", "ref2"], "B": [100.0, 200.0]})


def _working_file(file_id: str):
    return {
        "file_id": file_id,
        "description_column": "A",
        "description_range": {"start": 2, "end": 4},   # inclusive: rows 2,3,4 → df[0:3] → 3 rows
        "price_target_column": "B",
        "matching_report_column": "C",
    }


def _reference_file(file_id: str):
    return {
        "file_id": file_id,
        "description_column": "A",
        "description_range": {"start": 2, "end": 3},   # inclusive: rows 2,3 → df[0:2] → 2 rows
        "price_source_column": "B",
    }


def _mock_match_result(score: float, ref_id: str):
    return MatchResult(
        matched_reference_id=ref_id,
        score=MatchScore.create(score, score, 75.0),
        confidence=0.9,
        message="Test match",
        breakdown={},
    )


def _setup_file_storage(svc, wf_id, ref_id, wf_df, ref_df):
    """Wire file_storage and excel_reader mocks so process() can load files."""
    wf_path = Path(f"/tmp/wf_{wf_id}.xlsx")
    ref_path = Path(f"/tmp/ref_{ref_id}.xlsx")

    wf_dir = MagicMock()
    wf_dir.glob.return_value = [wf_path]
    ref_dir = MagicMock()
    ref_dir.glob.return_value = [ref_path]

    svc._file_storage.get_uploaded_file_path.side_effect = (
        lambda fid: wf_dir if str(fid) == wf_id else ref_dir
    )
    svc._excel_reader.read_excel_to_dataframe.side_effect = (
        lambda p: wf_df if p == wf_path else ref_df
    )
    svc._file_storage.get_result_file_path.return_value = Path("/tmp/result.xlsx")


# ---------------------------------------------------------------------------
# MatchingResult dataclass
# ---------------------------------------------------------------------------


def test_matching_result_fields():
    """MatchingResult stores all expected fields."""
    r = MatchingResult(matches_count=5, rows_processed=10, rows_matched=5,
                       using_ai=False, ai_model=None)
    assert r.matches_count == 5
    assert r.rows_processed == 10
    assert r.rows_matched == 5
    assert r.using_ai is False
    assert r.ai_model is None


# ---------------------------------------------------------------------------
# process() — happy path (no matches)
# ---------------------------------------------------------------------------


def test_process_returns_matching_result():
    """process() returns a MatchingResult with correct type."""
    wf_id, ref_id = str(uuid4()), str(uuid4())
    svc = _make_service()
    _setup_file_storage(svc, wf_id, ref_id, _make_wf_df(), _make_ref_df())
    svc._matching_engine.match_single.return_value = None  # no matches

    result = svc.process(
        job_id=str(uuid4()),
        working_file=_working_file(wf_id),
        reference_file=_reference_file(ref_id),
        matching_threshold=75.0,
        matching_strategy="best_match",
        report_format="simple",
    )

    assert isinstance(result, MatchingResult)
    assert result.matches_count == 0
    assert result.rows_processed == 3   # range start=2,end=4 inclusive → df[0:3] → 3 rows


def test_process_calls_excel_writer():
    """process() saves the result DataFrame via excel_writer."""
    wf_id, ref_id = str(uuid4()), str(uuid4())
    svc = _make_service()
    _setup_file_storage(svc, wf_id, ref_id, _make_wf_df(), _make_ref_df())
    svc._matching_engine.match_single.return_value = None

    svc.process(str(uuid4()), _working_file(wf_id), _reference_file(ref_id),
                 75.0, "best_match", "simple")

    assert svc._excel_writer.save_dataframe_to_excel.called


def test_process_progress_callback_is_called():
    """process() fires the progress callback at key stages."""
    wf_id, ref_id = str(uuid4()), str(uuid4())
    svc = _make_service()
    _setup_file_storage(svc, wf_id, ref_id, _make_wf_df(), _make_ref_df())
    svc._matching_engine.match_single.return_value = None

    calls = []
    svc.process(str(uuid4()), _working_file(wf_id), _reference_file(ref_id),
                 75.0, "best_match", "simple",
                 progress_callback=lambda pct, msg, stage, *a, **kw: calls.append(stage))

    stages = calls
    assert "FILES_LOADED" in stages
    assert "DESCRIPTIONS_EXTRACTED" in stages
    assert "PARAMETERS_EXTRACTED" in stages
    assert "SAVING_RESULTS" in stages


def test_process_no_progress_callback_does_not_crash():
    """process() works when progress_callback is None."""
    wf_id, ref_id = str(uuid4()), str(uuid4())
    svc = _make_service()
    _setup_file_storage(svc, wf_id, ref_id, _make_wf_df(), _make_ref_df())
    svc._matching_engine.match_single.return_value = None

    result = svc.process(str(uuid4()), _working_file(wf_id), _reference_file(ref_id),
                          75.0, "best_match", "simple", progress_callback=None)
    assert result.rows_processed == 3


# ---------------------------------------------------------------------------
# process() — match found (non-AI)
# ---------------------------------------------------------------------------


def test_process_counts_matches():
    """process() increments matches_count when a match is found."""
    wf_id, ref_id = str(uuid4()), str(uuid4())
    svc = _make_service()
    wf_df = _make_wf_df()
    ref_df = _make_ref_df()
    _setup_file_storage(svc, wf_id, ref_id, wf_df, ref_df)

    # ref descriptions will have id == "ref-id-1"
    def fake_match_single(wf_desc, ref_descs, threshold):
        mr = _mock_match_result(85.0, ref_descs[0].id if ref_descs else "no-id")
        return mr

    svc._matching_engine.match_single.side_effect = fake_match_single

    result = svc.process(str(uuid4()), _working_file(wf_id), _reference_file(ref_id),
                          75.0, "best_match", "simple")

    assert result.matches_count > 0
    assert result.rows_matched > 0


def test_process_output_columns_renamed():
    """Result DataFrame has human-readable column names: Cena, Match Score, Match Report."""
    wf_id, ref_id = str(uuid4()), str(uuid4())
    svc = _make_service()
    _setup_file_storage(svc, wf_id, ref_id, _make_wf_df(), _make_ref_df())
    svc._matching_engine.match_single.return_value = None

    saved_dfs = []
    svc._excel_writer.save_dataframe_to_excel.side_effect = (
        lambda df, path: saved_dfs.append(df)
    )

    svc.process(str(uuid4()), _working_file(wf_id), _reference_file(ref_id),
                 75.0, "best_match", "simple")

    assert saved_dfs, "excel_writer was not called"
    cols = saved_dfs[0].columns
    assert "Cena" in cols
    assert "Match Score" in cols
    assert "Match Report" in cols


# ---------------------------------------------------------------------------
# process() — error paths
# ---------------------------------------------------------------------------


def test_process_raises_when_working_file_missing():
    """process() raises FileNotFoundError when working file upload dir is empty."""
    wf_id, ref_id = str(uuid4()), str(uuid4())
    svc = _make_service()

    empty_dir = MagicMock()
    empty_dir.glob.return_value = []
    svc._file_storage.get_uploaded_file_path.return_value = empty_dir

    with pytest.raises(FileNotFoundError):
        svc.process(str(uuid4()), _working_file(wf_id), _reference_file(ref_id),
                     75.0, "best_match", "simple")


# ---------------------------------------------------------------------------
# _get_price_for_match
# ---------------------------------------------------------------------------


def test_get_price_for_match_valid_id():
    """_get_price_for_match returns HVACDescription with price for valid ID."""
    svc = _make_service()
    ref_prices = [100.0, 200.0]
    ref_texts = ["item1", "item2"]
    # source_row_number=2, ref_range_start_excel=2 → df_index=0
    mr = _mock_match_result(85.0, f"{uuid4()}_2")

    desc = svc._get_price_for_match(mr, ref_prices, ref_texts, ref_range_start_excel=2)

    assert desc is not None
    assert desc.matched_price == Decimal("100.0")
    assert desc.raw_text == "item1"


def test_get_price_for_match_invalid_id_format():
    """_get_price_for_match returns None for malformed ID."""
    svc = _make_service()
    mr = _mock_match_result(85.0, "no-underscore-here")

    desc = svc._get_price_for_match(mr, [100.0], ["item1"], ref_range_start_excel=2)

    assert desc is None


def test_get_price_for_match_out_of_range():
    """_get_price_for_match returns None when df_index is out of range."""
    svc = _make_service()
    ref_prices = [100.0]
    # source_row_number=5, ref_range_start_excel=2 → df_index=3 → out of range
    mr = _mock_match_result(85.0, f"{uuid4()}_5")

    desc = svc._get_price_for_match(mr, ref_prices, ["item1"], ref_range_start_excel=2)

    assert desc is None


def test_get_price_for_match_no_price():
    """_get_price_for_match returns HVACDescription with no price when price is empty."""
    svc = _make_service()
    ref_prices = [None]
    ref_texts = ["item1"]
    mr = _mock_match_result(85.0, f"{uuid4()}_2")

    desc = svc._get_price_for_match(mr, ref_prices, ref_texts, ref_range_start_excel=2)

    assert desc is not None
    assert desc.matched_price is None
