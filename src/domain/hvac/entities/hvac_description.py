"""
HVACDescription Entity.

Core domain entity representing a single HVAC product description.
This entity has identity (UUID) and lifecycle (state transitions).

Unlike Value Objects, Entities are mutable and track their state over time.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from src.domain.hvac.value_objects.match_score import MatchScore
from src.domain.shared.exceptions import InvalidHVACDescriptionError


class HVACDescriptionState(str, Enum):
    """
    Lifecycle states of HVACDescription entity.

    State transitions represent the processing pipeline:
    CREATED -> PARAMETERS_EXTRACTED -> MATCHED -> PRICED

    States:
        CREATED: Initial state after creation from raw text
        PARAMETERS_EXTRACTED: Technical parameters (DN, PN, etc.) extracted
        MATCHED: Successfully matched with reference description
        PRICED: Price information merged from matching result
    """

    CREATED = "created"
    PARAMETERS_EXTRACTED = "parameters_extracted"
    MATCHED = "matched"
    PRICED = "priced"


@dataclass
class HVACDescription:
    """
    Mutable entity representing HVAC product description with lifecycle tracking.

    This is a core domain entity that encapsulates:
    - Raw textual description from Excel
    - Extracted technical parameters (DN, PN, material, etc.)
    - Matching results and scores
    - Price information from reference catalog
    - Processing state tracking

    The entity follows a state machine pattern:
    CREATED -> PARAMETERS_EXTRACTED -> MATCHED -> PRICED

    Attributes:
        id: Unique identifier (UUID4, auto-generated)
        raw_text: Original description text from Excel (min 3 characters)
        extracted_params: Dictionary of extracted technical parameters
        match_score: Hybrid matching score (None if not matched yet)
        source_row_number: Excel row number for tracking (optional)
        file_id: Source file identifier (optional)
        matched_price: Price from matched reference description (optional)
        state: Current processing state
        created_at: Entity creation timestamp
        updated_at: Last modification timestamp

    Examples:
        >>> desc = HVACDescription(
        ...     raw_text="Zawór kulowy DN50 PN16 mosiężny",
        ...     source_row_number=10,
        ...     file_id=UUID("...")
        ... )
        >>> desc.state
        <HVACDescriptionState.CREATED: 'created'>
        >>> desc.is_valid()
        True
        >>> desc.has_parameters()
        False
    """

    # Required fields
    raw_text: str

    # Optional tracking fields
    source_row_number: int | None = None
    file_id: UUID | None = None

    # Processing results (populated during pipeline)
    extracted_params: dict[str, Any] = field(default_factory=dict)
    match_score: MatchScore | None = None
    matched_price: Decimal | None = None

    # State tracking
    state: HVACDescriptionState = HVACDescriptionState.CREATED

    # Identity and timestamps (auto-generated)
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Minimum text length for valid description
    MIN_TEXT_LENGTH: int = 3

    def __post_init__(self) -> None:
        """
        Validate and normalize entity after initialization.

        Called automatically by dataclass after __init__.
        Performs validation and text normalization.

        Raises:
            InvalidHVACDescriptionError: If raw_text is invalid
        """
        self._validate_text(self.raw_text)
        self.raw_text = self._normalize_text(self.raw_text)

    @classmethod
    def from_excel_row(
        cls,
        raw_text: str,
        source_row_number: int,
        file_id: UUID | None = None,
    ) -> "HVACDescription":
        """
        Factory method to create HVACDescription from Excel row data.

        Creates a new HVACDescription entity with metadata from Excel source.
        This is the recommended way to create descriptions from Excel files
        as it ensures proper metadata tracking.

        Args:
            raw_text: Description text from Excel cell
            source_row_number: Excel row number (1-based notation)
            file_id: UUID of source Excel file (optional)

        Returns:
            New HVACDescription entity with CREATED state

        Raises:
            InvalidHVACDescriptionError: If raw_text is invalid
                - Empty or too short (< 3 characters)
                - Not a string

        Examples:
            >>> from uuid import UUID
            >>> desc = HVACDescription.from_excel_row(
            ...     raw_text="Zawór kulowy DN50 PN16 mosiężny",
            ...     source_row_number=5,
            ...     file_id=UUID("a3bb189e-8bf9-3888-9912-ace4e6543002")
            ... )
            >>> print(desc.raw_text)
            'Zawór kulowy DN50 PN16 mosiężny'
            >>> print(desc.source_row_number)
            5
            >>> print(desc.state)
            <HVACDescriptionState.CREATED: 'created'>

        Usage Pattern:
            This factory method is used by ExcelReaderService:
            >>> # In ExcelReaderService._create_hvac_descriptions()
            >>> descriptions = [
            ...     HVACDescription.from_excel_row(text, row_num, file_id)
            ...     for text, row_num in text_with_rows
            ... ]

        Architecture Note:
            Factory method pattern separates construction logic from entity.
            This allows Infrastructure Layer to create entities without knowing
            internal initialization details. Follows Clean Architecture principles.
        """
        return cls(
            raw_text=raw_text,
            source_row_number=source_row_number,
            file_id=file_id,
        )

    def _validate_text(self, text: str) -> None:
        """
        Validate raw_text meets business rules.

        Business rules:
        - Text must be non-empty string
        - Text must have at least MIN_TEXT_LENGTH characters (after strip)

        Args:
            text: Text to validate

        Raises:
            InvalidHVACDescriptionError: If validation fails
        """
        if not isinstance(text, str):
            raise InvalidHVACDescriptionError(
                f"raw_text must be string, got {type(text).__name__}"
            )

        if not text or len(text.strip()) < self.MIN_TEXT_LENGTH:
            raise InvalidHVACDescriptionError(
                f"raw_text must have at least {self.MIN_TEXT_LENGTH} characters, "
                f"got {len(text.strip())}"
            )

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text by removing extra whitespace.

        Normalization steps:
        1. Strip leading/trailing whitespace
        2. Replace multiple spaces with single space
        3. Replace tabs and newlines with spaces

        Args:
            text: Text to normalize

        Returns:
            Normalized text

        Examples:
            >>> self._normalize_text("  Zawór  DN50  ")
            'Zawór DN50'
            >>> self._normalize_text("Zawór\\n\\nDN50")
            'Zawór DN50'
        """
        # Replace tabs and newlines with spaces
        text = text.replace("\t", " ").replace("\n", " ").replace("\r", " ")

        # Replace multiple spaces with single space
        while "  " in text:
            text = text.replace("  ", " ")

        # Strip leading/trailing whitespace
        return text.strip()

    def is_valid(self) -> bool:
        """
        Check if description meets minimum validity requirements.

        A description is considered valid if:
        - raw_text is not empty and >= MIN_TEXT_LENGTH
        - No validation errors would be raised

        Returns:
            True if description is valid, False otherwise

        Examples:
            >>> desc = HVACDescription(raw_text="Zawór DN50")
            >>> desc.is_valid()
            True
            >>> desc.raw_text = ""
            >>> desc.is_valid()
            False
        """
        try:
            self._validate_text(self.raw_text)
            return True
        except InvalidHVACDescriptionError:
            return False

    def has_parameters(self) -> bool:
        """
        Check if technical parameters have been extracted.

        Returns:
            True if extracted_params contains at least one parameter

        Examples:
            >>> desc = HVACDescription(raw_text="Zawór DN50")
            >>> desc.has_parameters()
            False
            >>> desc.extracted_params = {"dn": DiameterNominal(50)}
            >>> desc.has_parameters()
            True
        """
        return bool(self.extracted_params)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize entity to dictionary for storage/transport.

        Converts entity to JSON-serializable dictionary.
        Nested objects (MatchScore, Value Objects) are also serialized.

        Returns:
            Dictionary representation of entity

        Examples:
            >>> desc = HVACDescription(raw_text="Zawór DN50")
            >>> data = desc.to_dict()
            >>> data['raw_text']
            'Zawór DN50'
            >>> data['state']
            'created'
        """
        return {
            "id": str(self.id),
            "raw_text": self.raw_text,
            "extracted_params": self._serialize_params(self.extracted_params),
            "match_score": self.match_score.to_dict() if self.match_score else None,
            "source_row_number": self.source_row_number,
            "file_id": str(self.file_id) if self.file_id else None,
            "matched_price": str(self.matched_price) if self.matched_price else None,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def _serialize_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Serialize extracted parameters to JSON-compatible format.

        Handles conversion of Value Objects (DN, PN) to strings.

        Args:
            params: Dictionary of parameters to serialize

        Returns:
            Serialized parameters dictionary
        """
        serialized = {}
        for key, value in params.items():
            # Check if value has to_string method (Value Objects)
            if hasattr(value, "to_string"):
                serialized[key] = value.to_string()
            # Check if value has value attribute (Value Objects alternative)
            elif hasattr(value, "value"):
                serialized[key] = value.value
            else:
                serialized[key] = value
        return serialized

    def merge_with_price(
        self, price: Decimal, matched_description: str, match_score: MatchScore
    ) -> None:
        """
        Merge price information from matching result.

        Updates entity with:
        - Matched price from reference catalog
        - Match score details
        - State transition to PRICED
        - Updated timestamp

        Args:
            price: Price from matched reference description
            matched_description: Text of matched reference description (for reporting)
            match_score: Matching score details

        Raises:
            InvalidHVACDescriptionError: If price is negative or match_score is invalid

        Examples:
            >>> desc = HVACDescription(raw_text="Zawór DN50")
            >>> desc.merge_with_price(
            ...     price=Decimal("250.00"),
            ...     matched_description="Zawór kulowy DN50 PN16",
            ...     match_score=MatchScore(...)
            ... )
            >>> desc.state
            <HVACDescriptionState.PRICED: 'priced'>
            >>> desc.matched_price
            Decimal('250.00')
        """
        if price < 0:
            raise InvalidHVACDescriptionError(f"Price cannot be negative, got {price}")

        if not isinstance(match_score, MatchScore):
            raise InvalidHVACDescriptionError(
                f"match_score must be MatchScore instance, got {type(match_score).__name__}"
            )

        self.matched_price = price
        self.match_score = match_score
        self.extracted_params["_matched_description"] = matched_description
        self.state = HVACDescriptionState.PRICED
        self.updated_at = datetime.now()

    def get_match_report(self) -> str | None:
        """
        Generate human-readable matching report.

        Creates formatted string with matching details for Excel export.
        Returns None if description hasn't been matched yet.

        Format: "Matched: <description> | Score: <score>% | DN: <dn> | PN: <pn>"

        Returns:
            Formatted matching report string or None if not matched

        Examples:
            >>> desc = HVACDescription(raw_text="Zawór DN50")
            >>> desc.get_match_report()
            None
            >>> desc.merge_with_price(Decimal("250"), "Zawór DN50 PN16", score)
            >>> desc.get_match_report()
            'Matched: Zawór DN50 PN16 | Score: 95.2% | Price: 250.00 PLN'
        """
        if not self.match_score or self.state not in (
            HVACDescriptionState.MATCHED,
            HVACDescriptionState.PRICED,
        ):
            return None

        matched_desc = self.extracted_params.get("_matched_description", "N/A")
        score_pct = self.match_score.final_score

        report_parts = [f"Matched: {matched_desc}", f"Score: {score_pct:.1f}%"]

        if self.matched_price:
            report_parts.append(f"Price: {self.matched_price} PLN")

        # Add key parameters if available
        if "dn" in self.extracted_params:
            dn = self.extracted_params["dn"]
            dn_str = dn.to_string() if hasattr(dn, "to_string") else str(dn)
            report_parts.append(f"DN: {dn_str}")

        if "pn" in self.extracted_params:
            pn = self.extracted_params["pn"]
            pn_str = pn.to_string() if hasattr(pn, "to_string") else str(pn)
            report_parts.append(f"PN: {pn_str}")

        return " | ".join(report_parts)

    def __repr__(self) -> str:
        """
        Developer-friendly representation.

        Returns:
            String representation for debugging
        """
        return (
            f"HVACDescription(id={self.id}, "
            f"raw_text='{self.raw_text[:50]}...', "
            f"state={self.state.value})"
        )

    def __str__(self) -> str:
        """
        User-friendly representation.

        Returns:
            Readable string representation
        """
        return f"{self.raw_text} [{self.state.value}]"
