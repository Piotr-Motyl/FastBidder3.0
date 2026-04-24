"""
Excel column utilities.

Pure utility functions for Excel column name ↔ index conversions.
Shared between Application layer (command validation) and
Infrastructure layer (Excel reading).
"""


def is_valid_excel_column(column: str) -> bool:
    """
    Validate Excel column format (A-ZZ range only).

    Args:
        column: Column string to validate (e.g., 'A', 'B', 'AA', 'ZZ')

    Returns:
        True if valid 1-2 uppercase letter format, False otherwise
    """
    if not column:
        return False
    if len(column) > 2:
        return False
    return column.isalpha() and column.isupper()


def excel_column_to_index(column: str) -> int:
    """
    Convert Excel column letter(s) to 0-based index.

    Examples:
        A → 0, B → 1, Z → 25, AA → 26, ZZ → 701

    Args:
        column: Excel column in uppercase format ('A', 'B', 'AA', etc.)

    Returns:
        0-based column index

    Raises:
        ValueError: If column format is invalid
    """
    if not is_valid_excel_column(column):
        raise ValueError(
            f"Invalid Excel column format: '{column}'. Must be A-ZZ range."
        )

    index = 0
    for i, char in enumerate(reversed(column)):
        char_value = ord(char) - ord("A") + 1
        index += char_value * (26**i)

    return index - 1  # Convert to 0-based
