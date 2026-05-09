"""
Excel Reader Service — reads .xlsx files via Polars.

Used by ProcessMatchingService to load working/reference files. Caches loaded
DataFrames per (file_path, sheet_name) for the lifetime of this instance.
"""

from pathlib import Path
from typing import Optional
from uuid import UUID

import polars as pl

from src.domain.hvac.entities.hvac_description import HVACDescription
from src.domain.shared.exceptions import (
    FileSizeExceededError,
    ExcelParsingError,
    ColumnNotFoundError,
)


class ExcelReaderService:
    """
    Reads Excel descriptions and exposes raw Polars DataFrames.

    Polars is used (not Pandas) for ~10x speedup on large files. Encoding
    fallback (UTF-8 → CP1250) is handled internally by Polars.
    """

    MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB; aligned with FileStorageService

    def __init__(self) -> None:
        # cache key: (str(file_path), sheet_name or "")
        self._dataframe_cache: dict[tuple[str, str], pl.DataFrame] = {}

    @staticmethod
    def _column_letter_to_index(column: str) -> int:
        """Convert Excel column letter to 0-based DataFrame index. A=0, B=1, ..., AA=26, ZZ=701."""
        if not column or not column.isalpha():
            raise ValueError(f"Column must contain only letters, got: '{column}'")

        index = 0
        for char in column.upper():
            index = index * 26 + (ord(char) - ord('A') + 1)
        return index - 1

    def read_descriptions(
        self,
        file_path: Path,
        description_column: str,
        start_row: int = 2,  # 1-based Excel notation
        end_row: Optional[int] = None,
        sheet_name: Optional[str] = None,
        file_id: Optional[UUID] = None,
    ) -> list[HVACDescription]:
        """
        Read HVAC descriptions from a column range and return entities.

        Empty rows (None or whitespace-only) are filtered out.
        `source_row_number` on each entity is the original 1-based Excel row.

        Raises:
            FileNotFoundError, FileSizeExceededError, ExcelParsingError,
            ColumnNotFoundError, ValueError (start_row > end_row).
        """
        self._validate_file_size(file_path)
        dataframe = self._load_excel_dataframe(file_path, sheet_name)
        self._validate_column_exists(dataframe, description_column)
        text_with_rows = self._extract_column_range(
            dataframe=dataframe,
            column=description_column,
            start_row=start_row,
            end_row=end_row,
        )
        return self._create_hvac_descriptions(text_with_rows, file_id)

    def _validate_file_size(self, file_path: Path) -> None:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size_bytes = file_path.stat().st_size
        if file_size_bytes > self.MAX_FILE_SIZE_BYTES:
            raise FileSizeExceededError(
                message="Excel file exceeds maximum allowed size",
                file_size_bytes=file_size_bytes,
                max_size_bytes=self.MAX_FILE_SIZE_BYTES,
            )

    def _load_excel_dataframe(
        self, file_path: Path, sheet_name: Optional[str] = None
    ) -> pl.DataFrame:
        """Load Excel into Polars DataFrame with per-instance caching. Falls back to openpyxl engine."""
        cache_key = (str(file_path), sheet_name or "")
        if cache_key in self._dataframe_cache:
            return self._dataframe_cache[cache_key]

        df: Optional[pl.DataFrame] = None
        last_error: Optional[Exception] = None

        try:
            df = pl.read_excel(source=file_path, sheet_name=sheet_name)
        except Exception as e:
            last_error = e
            try:
                df = pl.read_excel(
                    source=file_path,
                    sheet_name=sheet_name,
                    engine="openpyxl",
                )
            except Exception as fallback_error:
                last_error = fallback_error

        if df is None:
            raise ExcelParsingError(
                message="Cannot parse Excel file (tried default and openpyxl engines)",
                file_path=str(file_path),
                original_error=last_error,
            )

        self._dataframe_cache[cache_key] = df
        return df

    def _validate_column_exists(
        self, dataframe: pl.DataFrame, column: str
    ) -> None:
        column_index = self._column_letter_to_index(column)
        num_columns = len(dataframe.columns)

        if column_index >= num_columns:
            raise ColumnNotFoundError(
                message=(
                    f"Column '{column}' (index {column_index}) not found. "
                    f"File has {num_columns} columns: {', '.join(dataframe.columns)}"
                ),
                column=column,
                available_columns=list(dataframe.columns),
            )

    def _extract_column_range(
        self,
        dataframe: pl.DataFrame,
        column: str,
        start_row: int,
        end_row: Optional[int],
    ) -> list[tuple[str, int]]:
        """Return [(text, excel_row_number)] for non-empty cells in the column range (1-based, inclusive)."""
        if end_row is not None and start_row > end_row:
            raise ValueError(
                f"start_row ({start_row}) must be <= end_row ({end_row})"
            )

        column_index = self._column_letter_to_index(column)
        column_name = dataframe.columns[column_index]

        # Excel row N → DataFrame index N-1; slice end is exclusive, so end_row is fine as-is.
        start_index = start_row - 1
        end_index = end_row if end_row is not None else len(dataframe)

        sliced_df = dataframe[start_index:end_index]
        column_data = sliced_df[column_name].to_list()

        results: list[tuple[str, int]] = []
        for df_idx, value in enumerate(column_data):
            excel_row = start_row + df_idx
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            results.append((text, excel_row))

        return results

    def _create_hvac_descriptions(
        self,
        text_with_rows: list[tuple[str, int]],
        file_id: Optional[UUID],
    ) -> list[HVACDescription]:
        return [
            HVACDescription.from_excel_row(
                raw_text=text,
                source_row_number=row_number,
                file_id=file_id,
            )
            for text, row_number in text_with_rows
        ]

    def read_excel_to_dataframe(
        self, file_path: Path, sheet_name: Optional[str] = None
    ):
        """Return raw Polars DataFrame (used by Celery task that needs full DF access)."""
        self._validate_file_size(file_path)
        return self._load_excel_dataframe(file_path, sheet_name)

    def _clear_cache(self) -> None:
        """Clear the in-memory DataFrame cache."""
        self._dataframe_cache.clear()
