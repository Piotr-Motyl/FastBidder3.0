"""
Excel Writer Service

Writes results to Excel files using Polars library.
Generates final output with matched prices and match reports.

Responsibility:
    - Create result Excel files with matched prices
    - Add match report columns (score, confidence, justification)
    - Preserve original Excel formatting (where possible)
    - Use Polars for fast writing

Architecture Notes:
    - Infrastructure Layer (depends on Polars library)
    - Used by Application Layer after matching completes
    - Phase 1: Simple write with new columns
    - Phase 2: Preserve Excel formatting, styling
"""

from pathlib import Path
from typing import Optional

import polars as pl

from src.domain.hvac import HVACDescription


class ExcelWriterService:
    """
    Service for writing match results to Excel files.

    Takes matched HVACDescription entities and generates output Excel
    with original data plus new columns:
    - Matched price (from reference catalog)
    - Match score (0-100)
    - Match confidence (0-1)
    - Match justification (why this match)

    Output Excel Structure:
        Original columns (preserved) + New columns:
        - "Cena" (Price) - matched price from reference
        - "Match Score" - final score (0-100)
        - "Confidence" - confidence level (0-1)
        - "Match Report" - human-readable justification

    Phase 1 Scope:
        - Simple write with Polars write_excel()
        - No formatting preservation (plain Excel)
        - All data in one sheet

    Examples:
        >>> writer = ExcelWriterService()
        >>>
        >>> # Write results
        >>> await writer.write_results(
        ...     descriptions=matched_descriptions,
        ...     output_path=Path("/tmp/job-123/result.xlsx"),
        ...     original_path=Path("/tmp/job-123/working.xlsx")
        ... )
    """

    def __init__(self) -> None:
        """
        Initialize Excel writer service.

        Phase 1: No configuration needed.
        Phase 2: Add formatting templates, style preservation.
        """
        pass

    async def write_results(
        self,
        descriptions: list[HVACDescription],
        output_path: Path,
        original_path: Optional[Path] = None,
    ) -> Path:
        """
        Write matched descriptions to Excel file with prices and match reports.

        Generates output Excel with:
        - Original data (if original_path provided)
        - Matched prices (from reference catalog lookup)
        - Match scores and confidence
        - Human-readable match justification

        Args:
            descriptions: List of matched HVACDescription entities
            output_path: Path where to save result Excel
            original_path: Optional path to original file (to preserve structure)

        Returns:
            Path to created Excel file

        Raises:
            OSError: If file cannot be written
            ValueError: If descriptions list is empty

        Examples:
            >>> writer = ExcelWriterService()
            >>>
            >>> # Matched descriptions with scores
            >>> matched = [
            ...     HVACDescription(
            ...         raw_text="Zawór DN50",
            ...         extracted_parameters={'dn': 50},
            ...         match_score=95.2,
            ...         matched_reference_id=UUID("...")
            ...     ),
            ...     # ... more
            ... ]
            >>>
            >>> # Write to Excel
            >>> result_path = await writer.write_results(
            ...     descriptions=matched,
            ...     output_path=Path("/tmp/result.xlsx"),
            ...     original_path=Path("/tmp/original.xlsx")
            ... )
            >>> print(f"Results saved to: {result_path}")

        Output Excel columns:
            Original columns (A, B, C, ...) + New columns:
            - Column X: "Cena" (matched price from reference)
            - Column Y: "Match Score" (0-100)
            - Column Z: "Confidence" (0-1)
            - Column AA: "Match Report" (justification)

        Implementation Notes:
            - Use Polars DataFrame for data manipulation
            - Read original file if provided (preserve other columns)
            - Add new columns with match data
            - Write using Polars write_excel()
            - Phase 2: Preserve cell formatting, colors, etc.
        """
        raise NotImplementedError(
            "write_results() to be implemented in Phase 3. "
            "Will create Polars DataFrame from descriptions and write to Excel."
        )

    async def write_unmatched_report(
        self, descriptions: list[HVACDescription], output_path: Path
    ) -> Path:
        """
        Write report of unmatched descriptions (for manual review).

        Creates Excel with descriptions that had no match above threshold.
        Used for quality assurance and manual pricing.

        Args:
            descriptions: List of unmatched HVACDescription entities
            output_path: Path where to save report Excel

        Returns:
            Path to created Excel file

        Examples:
            >>> writer = ExcelWriterService()
            >>>
            >>> # Descriptions with no match
            >>> unmatched = [
            ...     HVACDescription(
            ...         raw_text="Niestandardowy element",
            ...         extracted_parameters={},
            ...         match_score=None,
            ...         matched_reference_id=None
            ...     ),
            ...     # ... more
            ... ]
            >>>
            >>> report_path = await writer.write_unmatched_report(
            ...     descriptions=unmatched,
            ...     output_path=Path("/tmp/unmatched.xlsx")
            ... )

        Output Excel columns:
            - "Opis" (Description) - raw_text
            - "Extracted Parameters" - JSON of extracted_parameters
            - "Reason" - why no match (no params, no similar items, etc.)
        """
        raise NotImplementedError(
            "write_unmatched_report() to be implemented in Phase 3. "
            "Will create Excel report of items requiring manual review."
        )
