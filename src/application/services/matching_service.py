"""
ProcessMatchingService — core matching business logic.

Extracted from the Celery task so it can be unit-tested without a running worker.
The task becomes a thin orchestrator that delegates all processing here.
"""
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Optional
from uuid import UUID

import polars as pl

from src.domain.hvac.entities.hvac_description import HVACDescription
from src.shared.utils.excel import excel_column_to_index

logger = logging.getLogger(__name__)


@dataclass
class MatchingResult:
    """DTO returned by ProcessMatchingService.process()."""

    matches_count: int
    rows_processed: int
    rows_matched: int
    using_ai: bool
    ai_model: Optional[str]


class ProcessMatchingService:
    """
    Orchestrates the HVAC matching pipeline for one job.

    Handles stages 2–7 of the matching workflow:
    file loading → description extraction → entity creation →
    matching loop → result writing → saving.

    All infrastructure dependencies are injected, making this class
    fully unit-testable with mocks.
    """

    def __init__(
        self,
        matching_engine,
        parameter_extractor,
        file_storage,
        excel_reader,
        excel_writer,
        using_ai: bool = False,
        ai_model: Optional[str] = None,
        ai_event_loop=None,
    ) -> None:
        self._matching_engine = matching_engine
        self._parameter_extractor = parameter_extractor
        self._file_storage = file_storage
        self._excel_reader = excel_reader
        self._excel_writer = excel_writer
        self._using_ai = using_ai
        self._ai_model = ai_model
        self._ai_event_loop = ai_event_loop

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        job_id: str,
        working_file: dict,
        reference_file: dict,
        matching_threshold: float,
        matching_strategy: str,
        report_format: str,
        progress_callback: Optional[Callable] = None,
    ) -> MatchingResult:
        """
        Run the full matching pipeline for one job and return metrics.

        Args:
            job_id: Celery task ID (used for result file path).
            working_file: Dict from WorkingFileConfig (file_id, column mappings, ranges).
            reference_file: Dict from ReferenceFileConfig.
            matching_threshold: Minimum score to accept a match (1–100).
            matching_strategy: "best_match" | "first_match" | "all_matches".
            report_format: "simple" | "detailed" | "debug".
            progress_callback: Optional callable(pct, message, stage, current, total).

        Returns:
            MatchingResult with counts and AI metadata.
        """
        progress = progress_callback or (lambda *a, **kw: None)

        # ===== STAGE 2: FILES_LOADED (10%) =====
        progress(10, "Loading files from storage", "FILES_LOADED")

        wf_file_id = UUID(working_file["file_id"])
        ref_file_id = UUID(reference_file["file_id"])

        wf_upload_dir = self._file_storage.get_uploaded_file_path(wf_file_id)
        ref_upload_dir = self._file_storage.get_uploaded_file_path(ref_file_id)

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

        wf_df: pl.DataFrame = self._excel_reader.read_excel_to_dataframe(wf_files[0])
        ref_df: pl.DataFrame = self._excel_reader.read_excel_to_dataframe(ref_files[0])

        # ===== STAGE 3: DESCRIPTIONS_EXTRACTED (30%) =====
        progress(30, "Extracting descriptions from Excel", "DESCRIPTIONS_EXTRACTED")

        # start is 1-based Excel row; with has_header=True, Excel row 2 = DataFrame row 0
        wf_col_idx = excel_column_to_index(working_file["description_column"])
        wf_range_start = working_file["description_range"]["start"] - 2
        wf_range_end = working_file["description_range"]["end"] - 1
        wf_col_name = wf_df.columns[wf_col_idx]
        wf_raw_texts = wf_df[wf_range_start:wf_range_end][wf_col_name].to_list()

        ref_col_idx = excel_column_to_index(reference_file["description_column"])
        ref_range_start = reference_file["description_range"]["start"] - 2
        ref_range_end = reference_file["description_range"]["end"] - 1
        ref_col_name = ref_df.columns[ref_col_idx]
        ref_raw_texts = ref_df[ref_range_start:ref_range_end][ref_col_name].to_list()

        ref_price_col_idx = excel_column_to_index(reference_file["price_source_column"])
        ref_price_col_name = ref_df.columns[ref_price_col_idx]
        ref_prices = ref_df[ref_range_start:ref_range_end][ref_price_col_name].to_list()

        total_wf_rows = len(wf_raw_texts)
        logger.info(
            f"Extracted {total_wf_rows} WF descriptions, {len(ref_raw_texts)} REF descriptions"
        )

        # ===== STAGE 4: PARAMETERS_EXTRACTED (50%) =====
        progress(50, "Extracting HVAC parameters (DN, PN, etc.)", "PARAMETERS_EXTRACTED")

        n_rows = len(wf_df)
        target_col_idx = excel_column_to_index(working_file["price_target_column"])
        price_results: list = [None] * n_rows
        score_results: list = [None] * n_rows

        report_col_idx: Optional[int] = None
        report_results: Optional[list] = None
        if working_file.get("matching_report_column"):
            report_col_idx = excel_column_to_index(working_file["matching_report_column"])
            report_results = [None] * n_rows

        wf_descriptions: list[Optional[HVACDescription]] = []
        for text in wf_raw_texts:
            text_str = str(text).strip() if text is not None else ""
            if len(text_str) < 3:
                wf_descriptions.append(None)
                continue
            desc = HVACDescription(raw_text=text_str)
            desc.extract_parameters(self._parameter_extractor)
            wf_descriptions.append(desc)

        ref_descriptions = []
        for text, price in zip(ref_raw_texts, ref_prices):
            text_str = str(text).strip() if text is not None else ""
            if len(text_str) < 3:
                continue
            desc = HVACDescription(raw_text=text_str)
            desc.extract_parameters(self._parameter_extractor)
            if price and price != "":
                desc.matched_price = Decimal(str(price))
            ref_descriptions.append(desc)

        # ===== STAGE 5: MATCHING (50–90%) =====
        update_interval = min(100, max(1, total_wf_rows // 10))
        matches_count = 0
        rows_processed = 0
        rows_matched = 0

        for wf_idx, wf_desc in enumerate(wf_descriptions):
            if wf_desc is None:
                rows_processed += 1
                continue
            if self._using_ai:
                assert self._ai_event_loop is not None
                match_result = self._ai_event_loop.run_until_complete(
                    self._matching_engine.match(
                        working_description=wf_desc,
                        reference_descriptions=[],
                        threshold=matching_threshold,
                    )
                )
            else:
                match_result = self._matching_engine.match_single(
                    wf_desc, ref_descriptions, matching_threshold
                )

            rows_processed += 1

            if match_result:
                wf_desc.apply_match_result(match_result)

                matched_ref_desc = None
                if self._using_ai:
                    matched_ref_desc = self._get_price_for_match(
                        match_result, ref_prices, ref_raw_texts,
                        reference_file["description_range"]["start"],
                    )
                else:
                    matched_ref_desc = next(
                        (d for d in ref_descriptions if d.id == match_result.matched_reference_id),
                        None,
                    )

                if matched_ref_desc:
                    target_row = wf_range_start + wf_idx
                    price_value = (
                        float(matched_ref_desc.matched_price)
                        if matched_ref_desc.matched_price
                        else None
                    )
                    price_results[target_row] = price_value
                    score_results[target_row] = float(match_result.score.final_score)

                    if report_results is not None:
                        report_results[target_row] = (
                            f"Match: {matched_ref_desc.raw_text[:50]}... "
                            f"| Score: {match_result.score.final_score:.1f}%"
                        )

                    matches_count += 1
                    rows_matched += 1

            if (wf_idx + 1) % update_interval == 0 or wf_idx == total_wf_rows - 1:
                pct = 50 + int((wf_idx + 1) / total_wf_rows * 40)
                progress(
                    pct,
                    f"Matching descriptions ({matches_count} matched)",
                    "MATCHING",
                    wf_idx + 1,
                    total_wf_rows,
                )

        # ===== Write results into DataFrame =====
        while len(wf_df.columns) <= target_col_idx:
            wf_df = wf_df.with_columns(pl.lit(None).alias(f"_col_{len(wf_df.columns)}"))

        score_col_idx = target_col_idx + 1
        if report_col_idx is not None and score_col_idx == report_col_idx:
            score_col_idx = report_col_idx + 1

        if report_results is not None and report_col_idx is not None:
            while len(wf_df.columns) <= report_col_idx:
                wf_df = wf_df.with_columns(pl.lit(None).alias(f"_col_{len(wf_df.columns)}"))
        while len(wf_df.columns) <= score_col_idx:
            wf_df = wf_df.with_columns(pl.lit(None).alias(f"_col_{len(wf_df.columns)}"))

        price_col_name = wf_df.columns[target_col_idx]
        score_col_name = wf_df.columns[score_col_idx]

        update_cols = [
            pl.Series(price_col_name, price_results),
            pl.Series(score_col_name, score_results),
        ]
        if report_results is not None and report_col_idx is not None:
            report_col_name = wf_df.columns[report_col_idx]
            update_cols.append(pl.Series(report_col_name, report_results))

        wf_df = wf_df.with_columns(update_cols)

        rename_map = {price_col_name: "Cena", score_col_name: "Match Score"}
        if report_results is not None and report_col_idx is not None:
            rename_map[report_col_name] = "Match Report"
        wf_df = wf_df.rename(rename_map)

        # ===== STAGE 6: SAVING_RESULTS (90%) =====
        progress(90, "Saving results to Excel file", "SAVING_RESULTS")
        result_path = self._file_storage.get_result_file_path(UUID(job_id))
        self._excel_writer.save_dataframe_to_excel(wf_df, result_path)

        return MatchingResult(
            matches_count=matches_count,
            rows_processed=rows_processed,
            rows_matched=rows_matched,
            using_ai=self._using_ai,
            ai_model=self._ai_model,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_price_for_match(
        self,
        match_result,
        ref_prices: list,
        ref_raw_texts: list,
        ref_range_start_excel: int,
    ) -> Optional[HVACDescription]:
        """
        Parse ChromaDB ID format "{file_id}_{row_number}" to retrieve price.

        The last segment after "_" is the 1-based Excel source_row_number stored
        at index time. We convert it to a 0-based index into ref_prices by
        subtracting the range start Excel row number.
        """
        try:
            id_parts = match_result.matched_reference_id.split("_")
            if len(id_parts) < 2:
                logger.warning(
                    f"AI matching: Invalid matched_reference_id format: "
                    f"{match_result.matched_reference_id}"
                )
                return None

            source_row_number = int(id_parts[-1])
            df_index = source_row_number - ref_range_start_excel

            if not (0 <= df_index < len(ref_prices)):
                logger.warning(
                    f"AI matching: Row index {df_index} out of range "
                    f"(total {len(ref_prices)} prices)"
                )
                return None

            price_value = ref_prices[df_index]
            ref_text = ref_raw_texts[df_index] if df_index < len(ref_raw_texts) else ""

            desc = HVACDescription(raw_text=str(ref_text) if ref_text else "")
            if price_value and price_value != "":
                desc.matched_price = Decimal(str(price_value))

            logger.debug(
                f"AI matching: Retrieved price for row {source_row_number}: "
                f"{price_value if price_value else 'N/A'}"
            )
            return desc

        except Exception as e:
            logger.error(
                f"AI matching: Failed to retrieve price from DataFrame: {e}",
                exc_info=True,
            )
            return None
