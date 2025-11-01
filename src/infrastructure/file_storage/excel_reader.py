"""
Excel Reader Service

Reads Excel files using Polars library (10x faster than Pandas).
Extracts HVAC descriptions from specified columns and rows.

Responsibility:
    - Parse Excel files (.xlsx, .xls) using Polars
    - Extract description text from configured columns
    - Handle various Excel structures (different column positions)
    - Convert to HVACDescription entities
    - Validate Excel structure (required columns present)

Architecture Notes:
    - Infrastructure Layer (depends on Polars library)
    - Used by Application Layer (ProcessMatchingUseCase)
    - Phase 1: Simple column extraction
    - Phase 2: Advanced column detection, multi-sheet support
"""

from pathlib import Path
from typing import Optional

import polars as pl

from src.domain.hvac import HVACDescription


class ExcelReaderService:
    """
    Service for reading and parsing Excel files with Polars.

    FastBidder uses Polars instead of Pandas for:
    - 10x faster performance on large files
    - Lower memory consumption
    - Better handling of messy Excel data

    Usage Patterns:
    1. Working File (WF): Read descriptions to be priced
    2. Reference File (REF): Read catalog with descriptions + prices

    Excel Structure (Phase 1 assumptions):
        - Descriptions in specific column (configurable, e.g., column B)
        - Optional: Header row (row 1)
        - Data starts from row 2 (or configurable start_row)
        - Max 100 descriptions (Phase 1 limit)

    Phase 1 Scope:
        - Simple column extraction (column letter or index)
        - Single sheet only (first sheet)
        - No advanced detection (manual column specification)

    Examples:
        >>> reader = ExcelReaderService()
        >>>
        >>> # Read working file descriptions (column B, rows 2-101)
        >>> descriptions = await reader.read_descriptions(
        ...     file_path=Path("/tmp/job-123/working.xlsx"),
        ...     description_column="B",
        ...     start_row=2,
        ...     max_rows=100
        ... )
        >>> print(f"Read {len(descriptions)} descriptions")
        >>> for desc in descriptions:
        ...     print(desc.raw_text)
    """

    def __init__(self) -> None:
        """
        Initialize Excel reader service.

        Phase 1: No configuration needed (uses Polars defaults).
        Phase 2: Add caching, column detection strategies.
        """
        pass

    async def read_descriptions(
        self,
        file_path: Path,
        description_column: str,  # e.g., "B" or "C" or column index
        start_row: int = 2,
        max_rows: Optional[int] = None,
    ) -> list[HVACDescription]:
        """
        Read HVAC descriptions from Excel file.

        Extracts description text from specified column and converts
        to HVACDescription entities.

        Args:
            file_path: Path to Excel file (.xlsx or .xls)
            description_column: Column containing descriptions (letter or index)
                - Letter: "A", "B", "C", etc.
                - Index: 0, 1, 2, etc. (0-based)
            start_row: First data row (1-based, default 2 to skip header)
            max_rows: Maximum rows to read (default None = all rows)
                - Phase 1: Limit to 100 for testing

        Returns:
            List of HVACDescription entities (one per row)
            Empty list if no valid descriptions found

        Raises:
            FileNotFoundError: If Excel file doesn't exist
            ValueError: If column not found or Excel structure invalid
            PolarsError: If file cannot be parsed

        Examples:
            >>> reader = ExcelReaderService()
            >>>
            >>> # Read from column B, skip header row
            >>> descriptions = await reader.read_descriptions(
            ...     file_path=Path("working_file.xlsx"),
            ...     description_column="B",
            ...     start_row=2,
            ...     max_rows=100
            ... )
            >>>
            >>> # First description
            >>> print(descriptions[0].raw_text)
            'Zawór kulowy DN50 PN16'
            >>> print(descriptions[0].has_parameters())
            False  # Parameters not extracted yet

        Implementation Notes:
            - Use Polars read_excel() with engine="xlsx2csv" or "calamine"
            - Convert column letter to index if needed (A=0, B=1, etc.)
            - Skip empty rows automatically
            - Trim whitespace from descriptions (validator handles this)
            - Use HVACDescription.from_excel_row() factory method
        """
        raise NotImplementedError(
            "read_descriptions() to be implemented in Phase 3. "
            "Will use Polars to read Excel, extract column, create HVACDescription entities."
        )

    async def detect_description_column(self, file_path: Path) -> Optional[str]:
        """
        Auto-detect which column contains descriptions (future feature).

        Phase 1: Not implemented (manual column specification)
        Phase 2: Heuristic detection based on:
            - Column header keywords ("opis", "description", "nazwa")
            - Non-empty cell count
            - Average text length

        Args:
            file_path: Path to Excel file

        Returns:
            Detected column letter (e.g., "B") or None if cannot detect

        Examples:
            >>> reader = ExcelReaderService()
            >>> column = await reader.detect_description_column(
            ...     Path("file.xlsx")
            ... )
            >>> print(f"Detected column: {column}")
        """
        raise NotImplementedError(
            "detect_description_column() to be implemented in Phase 2. "
            "Phase 1 uses manual column specification."
        )

    async def validate_excel_structure(
        self, file_path: Path, required_columns: list[str]
    ) -> bool:
        """
        Validate that Excel file has required structure (basic check).

        Phase 1: Basic validation (file readable, has sheets, has rows)
        Phase 2: Advanced validation (column headers, data types)

        Args:
            file_path: Path to Excel file
            required_columns: List of required column letters/indices

        Returns:
            True if structure is valid, False otherwise

        Raises:
            FileNotFoundError: If file doesn't exist

        Examples:
            >>> reader = ExcelReaderService()
            >>> is_valid = await reader.validate_excel_structure(
            ...     Path("file.xlsx"),
            ...     required_columns=["B", "E"]  # Description and Price columns
            ... )
            >>> if is_valid:
            ...     print("Excel structure OK")
        """
        raise NotImplementedError(
            "validate_excel_structure() to be implemented in Phase 3. "
            "Will check if Excel is readable and has required columns."
        )
