"""
Excel Writer Service

Writes results to Excel files using openpyxl library.
Generates final output with matched prices and match reports.

Responsibility:
    - Create result Excel files with matched prices
    - Add match report columns (score, confidence, justification)
    - Preserve original Excel formatting (styles, colors, formulas)
    - Apply conditional coloring based on match scores
    - Create backup of original file before modification
    - Auto-size columns for readability

Architecture Notes:
    - Infrastructure Layer (depends on openpyxl library)
    - Used by Application Layer after matching completes (matching_tasks)
    - Phase 1: Simple write contract
    - Phase 2: Full contract with formatting preservation, coloring, backup
"""

import shutil
from pathlib import Path
from typing import Optional

from openpyxl import load_workbook
from openpyxl.styles import PatternFill
from openpyxl.workbook import Workbook
from openpyxl.worksheet.worksheet import Worksheet

from src.application.models import ReportFormat
from src.domain.hvac.entities.hvac_description import HVACDescription


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

    # Color definitions for conditional formatting (RGB hex)
    COLOR_GREEN = "00FF00"  # Score > 90%
    COLOR_YELLOW = "FFFF00"  # Score 75-90%
    COLOR_RED = "FF0000"  # Score < 75%

    def __init__(self) -> None:
        """
        Initialize Excel writer service.

        Phase 1: No configuration needed.
        Phase 2: Color definitions for conditional formatting.

        Attributes:
            COLOR_GREEN: RGB hex for high score (>90%)
            COLOR_YELLOW: RGB hex for medium score (75-90%)
            COLOR_RED: RGB hex for low score (<75%)
        """
        pass

    @staticmethod
    def _column_letter_to_index(column: str) -> int:
        """
        Convert Excel column letter to 1-based index for openpyxl.

        Converts Excel column notation (A, B, AA, ZZ) to 1-based integer index
        for openpyxl column access (openpyxl uses 1-based indexing).

        Args:
            column: Excel column letter (e.g., "A", "B", "AA", "ZZ")

        Returns:
            1-based column index (A=1, B=2, Z=26, AA=27, etc.)

        Raises:
            ValueError: If column contains non-alphabetic characters

        Examples:
            >>> ExcelWriterService._column_letter_to_index("A")
            1
            >>> ExcelWriterService._column_letter_to_index("B")
            2
            >>> ExcelWriterService._column_letter_to_index("Z")
            26
            >>> ExcelWriterService._column_letter_to_index("AA")
            27

        Algorithm:
            Excel column letters work like base-26 number system:
            - A = 1, B = 2, ..., Z = 26
            - AA = 27, AB = 28, ..., AZ = 52
            - BA = 53, ..., ZZ = 702
            openpyxl uses 1-based indexing (different from Polars which is 0-based)
        """
        # Validate input
        if not column or not column.isalpha():
            raise ValueError(f"Column must contain only letters, got: '{column}'")

        # Convert Excel column letter to 1-based index (A=1, B=2, AA=27, etc.)
        index = 0
        for char in column.upper():
            index = index * 26 + (ord(char) - ord('A') + 1)

        return index

    def write_results_to_file(
        self,
        original_file_path: Path,
        descriptions: list[HVACDescription],
        price_column: str,
        report_column: Optional[str] = None,
        output_path: Optional[Path] = None,
        report_format: ReportFormat = ReportFormat.SIMPLE,
        sheet_name: Optional[str] = None,
    ) -> Path:
        """
        Write matching results (prices and reports) to Excel file (Phase 2 - Detailed Contract).

        Takes original working file, adds matched prices and optional reports,
        preserves all formatting, applies conditional coloring, and saves to output file.

        Process Flow:
            1. Create backup of original file
            2. Load workbook with openpyxl (preserves formatting)
            3. Write prices to price_column for each description
            4. Apply conditional coloring to price cells (based on match_score)
            5. Write match reports to report_column (if specified)
            6. Auto-size price_column and report_column
            7. Save workbook to output_path
            8. Return output_path

        Args:
            original_file_path: Path to original working file (to be modified)
            descriptions: List of HVACDescription entities with matched prices
                - Must have source_row_number for row mapping
                - matched_price: Decimal or None (if no match)
                - match_score: MatchScore or None (for coloring)
            price_column: Excel column letter where to write prices (e.g., "F")
                - Must be valid A-ZZ format
            report_column: Excel column letter where to write match reports (optional)
                - Default: None (no report column)
                - Example: "G", "H"
            output_path: Path where to save result file
                - Default: None (uses original_file_path parent / "result.xlsx")
            report_format: Format of match report (SIMPLE, DETAILED, DEBUG)
                - Default: ReportFormat.SIMPLE
                - Used only if report_column is specified
            sheet_name: Name of sheet to modify
                - Default: None (first sheet)

        Returns:
            Path to created result file

        Raises:
            FileNotFoundError: If original_file_path doesn't exist
            ValueError: If price_column or report_column has invalid format
            OSError: If file cannot be written or backup fails

        Conditional Coloring Rules (applied to price_column):
            - Green (#00FF00): match_score > 90%
            - Yellow (#FFFF00): match_score 75-90%
            - Red (#FF0000): match_score < 75%
            - No color: matched_price is None (no match)

        Examples:
            >>> writer = ExcelWriterService()
            >>> from pathlib import Path
            >>>
            >>> # Write results with prices only
            >>> output = await writer.write_results_to_file(
            ...     original_file_path=Path("/tmp/job-123/working_file.xlsx"),
            ...     descriptions=matched_descriptions,
            ...     price_column="F",
            ...     report_column=None
            ... )
            >>> print(output)
            Path('/tmp/job-123/result.xlsx')
            >>>
            >>> # Write results with prices and detailed reports
            >>> output = await writer.write_results_to_file(
            ...     original_file_path=Path("/tmp/job-123/working_file.xlsx"),
            ...     descriptions=matched_descriptions,
            ...     price_column="F",
            ...     report_column="G",
            ...     report_format=ReportFormat.DETAILED
            ... )

        Implementation Details (Phase 3):
            - Use openpyxl.load_workbook() to preserve formatting
            - Create backup: original_file.xlsx → original_file_backup.xlsx
            - Convert column letters to column index for openpyxl (A=1, B=2, AA=27, etc.)
            - For each description:
                * Get Excel row: desc.source_row_number (1-based)
                * If desc.matched_price is not None:
                    - Write price to ws[row][price_column]
                    - Apply color based on desc.match_score.final_score
                * If report_column and desc.match_score:
                    - Generate report using desc.get_match_report()
                    - Write to ws[row][report_column]
            - Auto-size only price_column and report_column (not all columns)
            - Save to output_path

        Performance Notes:
            - openpyxl is slower than Polars but preserves formatting
            - For 1000 rows: ~2-5 seconds (acceptable for Phase 2)
            - No parallel processing (Phase 3+)

        Phase 2 Notes:
            - Frozen panes: NOT modified (left as in original)
            - Formulas: NOT modified (openpyxl preserves them)
            - Other columns: NOT modified
            - Backup: Created in same directory with _backup suffix
            - Empty cells: If matched_price is None, cell stays empty (no text)
        """
        # Step 1: Create backup of original file
        self._create_backup(original_file_path)

        # Step 2: Load workbook with openpyxl (preserves formatting)
        workbook = self._load_workbook(original_file_path)

        # Step 3: Get worksheet (first sheet or specified sheet)
        worksheet = self._get_worksheet(workbook, sheet_name)

        # Step 4: Write prices to price_column
        self._write_prices_to_column(worksheet, descriptions, price_column)

        # Step 5: Apply conditional coloring to price cells
        self._apply_cell_coloring(worksheet, descriptions, price_column)

        # Step 6: Write match reports to report_column (if specified)
        if report_column:
            self._write_reports_to_column(
                worksheet, descriptions, report_column, report_format
            )

        # Step 7: Auto-size columns for readability
        columns_to_resize = [price_column]
        if report_column:
            columns_to_resize.append(report_column)
        self._autosize_columns(worksheet, columns_to_resize)

        # Step 8: Determine output path (default: parent / "result.xlsx")
        if output_path is None:
            output_path = original_file_path.parent / "result.xlsx"

        # Step 9: Save workbook to output_path
        result_path = self._save_workbook(workbook, output_path)

        return result_path

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
            >>> from src.domain.hvac.value_objects.extracted_parameters import ExtractedParameters
            >>> from src.domain.hvac.value_objects.match_score import MatchScore
            >>> from decimal import Decimal
            >>> matched = [
            ...     HVACDescription(
            ...         raw_text="Zawór DN50",
            ...         extracted_params=ExtractedParameters(dn=50, confidence_scores={"dn": 1.0}),
            ...         match_score=MatchScore.create(100.0, 92.0, 75.0),
            ...         matched_price=Decimal("250.00")
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

    def _create_backup(self, original_path: Path) -> Path:
        """
        Create backup of original file before modification.

        Creates a backup copy of the original file in the same directory
        with '_backup' suffix before the extension.

        Args:
            original_path: Path to original file

        Returns:
            Path to backup file

        Raises:
            FileNotFoundError: If original file doesn't exist
            OSError: If backup cannot be created (permissions, disk space)

        Examples:
            >>> writer = ExcelWriterService()
            >>> backup = writer._create_backup(Path("/tmp/working_file.xlsx"))
            >>> print(backup)
            Path('/tmp/working_file_backup.xlsx')

        Implementation (Phase 3):
            - Check if original_path exists
            - Create backup path: parent / (stem + "_backup" + suffix)
            - Copy file: shutil.copy2(original_path, backup_path)
            - Return backup_path
        """
        # Check if original file exists
        if not original_path.exists():
            raise FileNotFoundError(f"File not found: {original_path}")

        # Create backup path: parent directory / filename_backup.extension
        backup_path = original_path.parent / f"{original_path.stem}_backup{original_path.suffix}"

        # Copy file with metadata preservation (shutil.copy2 preserves timestamps)
        shutil.copy2(original_path, backup_path)

        return backup_path

    def _load_workbook(self, file_path: Path) -> Workbook:
        """
        Load Excel workbook with openpyxl (preserves formatting).

        Args:
            file_path: Path to Excel file

        Returns:
            openpyxl Workbook object

        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If openpyxl cannot load file

        Examples:
            >>> writer = ExcelWriterService()
            >>> wb = writer._load_workbook(Path("/tmp/file.xlsx"))
            >>> print(wb.sheetnames)
            ['Sheet1', 'Sheet2']

        Implementation (Phase 3):
            - Use openpyxl.load_workbook(file_path, data_only=False)
            - data_only=False preserves formulas (not just values)
            - Return workbook object
        """
        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load workbook with openpyxl
        # data_only=False preserves formulas (not just calculated values)
        workbook = load_workbook(filename=file_path, data_only=False)

        return workbook

    def _get_worksheet(
        self, workbook: Workbook, sheet_name: Optional[str] = None
    ) -> Worksheet:
        """
        Get worksheet from workbook.

        Args:
            workbook: openpyxl Workbook
            sheet_name: Name of sheet (None = first sheet)

        Returns:
            openpyxl Worksheet object

        Raises:
            ValueError: If sheet_name doesn't exist

        Examples:
            >>> writer = ExcelWriterService()
            >>> wb = writer._load_workbook(Path("file.xlsx"))
            >>> ws = writer._get_worksheet(wb, None)  # First sheet
            >>> ws = writer._get_worksheet(wb, "Sheet2")  # Named sheet

        Implementation (Phase 3):
            - If sheet_name is None: return wb.active
            - Else: return wb[sheet_name]
            - Handle KeyError if sheet doesn't exist
        """
        # If sheet_name not specified, use active sheet (first sheet)
        if sheet_name is None:
            return workbook.active

        # Get worksheet by name
        try:
            return workbook[sheet_name]
        except KeyError:
            available_sheets = ", ".join(workbook.sheetnames)
            raise ValueError(
                f"Sheet '{sheet_name}' not found. "
                f"Available sheets: {available_sheets}"
            )

    def _write_prices_to_column(
        self,
        worksheet: Worksheet,
        descriptions: list[HVACDescription],
        column: str,
    ) -> None:
        """
        Write matched prices to specified column.

        Writes price for each description to its source_row_number.
        If matched_price is None, leaves cell empty.

        Args:
            worksheet: openpyxl Worksheet to modify
            descriptions: List of HVACDescription with prices
            column: Excel column letter (e.g., "F")

        Examples:
            >>> writer = ExcelWriterService()
            >>> ws = worksheet
            >>> writer._write_prices_to_column(ws, descriptions, "F")
            >>> # Price written to F2, F3, F4, etc. (based on source_row_number)

        Implementation (Phase 3):
            - Convert column letter to column index for openpyxl (A=1, B=2, AA=27)
            - For each description:
                if desc.matched_price is not None:
                    row = desc.source_row_number (1-based)
                    cell = ws.cell(row=row, column=col_idx)
                    cell.value = float(desc.matched_price)
                    # Note: openpyxl uses 1-based for both row and column
        """
        # Convert column letter to 1-based index for openpyxl
        column_index = self._column_letter_to_index(column)

        # Write prices to column for each description
        for desc in descriptions:
            # Skip descriptions without matched price
            if desc.matched_price is None:
                continue

            # Get Excel row number (1-based, from source_row_number)
            row_number = desc.source_row_number

            # Get cell at (row, column) - both 1-based in openpyxl
            cell = worksheet.cell(row=row_number, column=column_index)

            # Write price as float (Decimal -> float for Excel)
            cell.value = float(desc.matched_price)

    def _write_reports_to_column(
        self,
        worksheet: Worksheet,
        descriptions: list[HVACDescription],
        column: str,
        report_format: ReportFormat,
    ) -> None:
        """
        Write match reports to specified column.

        Generates and writes match report for each description
        using HVACDescription.get_match_report().

        Args:
            worksheet: openpyxl Worksheet to modify
            descriptions: List of HVACDescription with match data
            column: Excel column letter (e.g., "G")
            report_format: Format of report (SIMPLE, DETAILED, DEBUG)

        Examples:
            >>> writer = ExcelWriterService()
            >>> ws = worksheet
            >>> writer._write_reports_to_column(
            ...     ws, descriptions, "G", ReportFormat.SIMPLE
            ... )
            >>> # Report written to G2, G3, G4, etc.

        Implementation (Phase 3):
            - Convert column letter to column index for openpyxl (A=1, B=2, AA=27)
            - For each description:
                if desc.match_score is not None:
                    report = desc.get_match_report()
                    row = desc.source_row_number
                    cell = ws.cell(row=row, column=col_idx)
                    cell.value = report
            - Note: report_format might be used in future for different report styles
        """
        # Convert column letter to 1-based index for openpyxl
        column_index = self._column_letter_to_index(column)

        # Write reports to column for each description
        for desc in descriptions:
            # Skip descriptions without match score (no match found)
            if desc.match_score is None:
                continue

            # Get match report from description
            report = desc.get_match_report()

            # Skip if report is None or empty
            if not report:
                continue

            # Get Excel row number (1-based)
            row_number = desc.source_row_number

            # Get cell at (row, column) - both 1-based in openpyxl
            cell = worksheet.cell(row=row_number, column=column_index)

            # Write report text
            cell.value = report

    def _apply_cell_coloring(
        self,
        worksheet: Worksheet,
        descriptions: list[HVACDescription],
        column: str,
    ) -> None:
        """
        Apply conditional coloring to price column cells based on match score.

        Colors cells according to score ranges:
        - Green: >90%
        - Yellow: 75-90%
        - Red: <75%
        - No color: matched_price is None

        Args:
            worksheet: openpyxl Worksheet to modify
            descriptions: List of HVACDescription with scores
            column: Excel column letter to color (price_column)

        Examples:
            >>> writer = ExcelWriterService()
            >>> ws = worksheet
            >>> writer._apply_cell_coloring(ws, descriptions, "F")
            >>> # Cells F2, F3, F4 colored based on match scores

        Implementation (Phase 3):
            - Convert column letter to column index for openpyxl (A=1, B=2, AA=27)
            - For each description:
                if desc.matched_price is not None and desc.match_score is not None:
                    score = desc.match_score.final_score
                    color = self._get_color_for_score(score)
                    if color:
                        row = desc.source_row_number
                        cell = ws.cell(row=row, column=col_idx)
                        fill = PatternFill(start_color=color, end_color=color, fill_type="solid")
                        cell.fill = fill
        """
        # Convert column letter to 1-based index for openpyxl
        column_index = self._column_letter_to_index(column)

        # Apply coloring to cells for each description
        for desc in descriptions:
            # Skip descriptions without matched price or match score
            if desc.matched_price is None or desc.match_score is None:
                continue

            # Get match score (0-100)
            score = desc.match_score.final_score

            # Get color based on score
            color = self._get_color_for_score(score)

            # Skip if no color (shouldn't happen, but defensive check)
            if not color:
                continue

            # Get Excel row number (1-based)
            row_number = desc.source_row_number

            # Get cell at (row, column) - both 1-based in openpyxl
            cell = worksheet.cell(row=row_number, column=column_index)

            # Create fill pattern with color
            fill = PatternFill(start_color=color, end_color=color, fill_type="solid")

            # Apply fill to cell
            cell.fill = fill

    def _get_color_for_score(self, score: float) -> Optional[str]:
        """
        Get RGB hex color for match score.

        Args:
            score: Match score (0-100)

        Returns:
            RGB hex color string (without #) or None

        Examples:
            >>> writer = ExcelWriterService()
            >>> writer._get_color_for_score(95.0)
            '00FF00'  # Green
            >>> writer._get_color_for_score(80.0)
            'FFFF00'  # Yellow
            >>> writer._get_color_for_score(70.0)
            'FF0000'  # Red

        Implementation (Phase 3):
            - If score > 90: return COLOR_GREEN
            - Elif score >= 75: return COLOR_YELLOW
            - Else: return COLOR_RED
        """
        if score > 90:
            return self.COLOR_GREEN
        elif score >= 75:
            return self.COLOR_YELLOW
        else:
            return self.COLOR_RED

    def _autosize_columns(
        self, worksheet: Worksheet, columns: list[str]
    ) -> None:
        """
        Auto-size specified columns based on content width.

        Adjusts column widths to fit content for readability.
        Only adjusts specified columns (not all columns).

        Args:
            worksheet: openpyxl Worksheet to modify
            columns: List of Excel column letters to auto-size (e.g., ["F", "G"])

        Examples:
            >>> writer = ExcelWriterService()
            >>> ws = worksheet
            >>> writer._autosize_columns(ws, ["F", "G"])
            >>> # Columns F and G widths adjusted

        Implementation (Phase 3):
            - For each column letter in columns:
                * Use column letter directly in openpyxl
                * Calculate max width:
                    max_width = 0
                    for cell in ws[column_letter]:
                        if cell.value:
                            max_width = max(max_width, len(str(cell.value)))
                * Set column width: ws.column_dimensions[column_letter].width = max_width + 2
        """
        # Auto-size each column in the list
        for column_letter in columns:
            # Calculate maximum content width for this column
            max_width = 0

            # Iterate through all cells in this column
            for cell in worksheet[column_letter]:
                # Skip empty cells
                if cell.value is None:
                    continue

                # Calculate length of cell value as string
                cell_length = len(str(cell.value))

                # Update max width
                max_width = max(max_width, cell_length)

            # Set column width (add padding of 2 for readability)
            # Minimum width of 8 to ensure columns are visible
            adjusted_width = max(max_width + 2, 8)
            worksheet.column_dimensions[column_letter].width = adjusted_width

    def _save_workbook(self, workbook: Workbook, output_path: Path) -> Path:
        """
        Save workbook to output path.

        Args:
            workbook: openpyxl Workbook to save
            output_path: Path where to save

        Returns:
            Path to saved file

        Raises:
            OSError: If file cannot be written

        Examples:
            >>> writer = ExcelWriterService()
            >>> wb = workbook
            >>> output = writer._save_workbook(wb, Path("/tmp/result.xlsx"))
            >>> print(output)
            Path('/tmp/result.xlsx')

        Implementation (Phase 3):
            - Create parent directory if doesn't exist: output_path.parent.mkdir(parents=True, exist_ok=True)
            - Save workbook: workbook.save(output_path)
            - Return output_path
        """
        # Create parent directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save workbook to file
        workbook.save(output_path)

        return output_path

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
            ...         extracted_params=None,
            ...         match_score=None,
            ...         matched_price=None
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
            - "Extracted Parameters" - JSON of extracted_params
            - "Reason" - why no match (no params, no similar items, etc.)
        """
        raise NotImplementedError(
            "write_unmatched_report() to be implemented in Phase 3. "
            "Will create Excel report of items requiring manual review."
        )
