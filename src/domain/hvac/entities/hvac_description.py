"""
HVACDescription Entity.

Core domain entity representing a single HVAC product description with identity and lifecycle.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from src.domain.hvac.value_objects.match_score import MatchScore
from src.domain.hvac.value_objects.extracted_parameters import ExtractedParameters
from src.domain.hvac.value_objects.match_result import MatchResult
from src.domain.shared.exceptions import InvalidHVACDescriptionError

# TYPE_CHECKING import to avoid circular dependency
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.hvac.services.parameter_extractor import ParameterExtractorProtocol


class HVACDescriptionState(str, Enum):
    """
    Lifecycle states of HVACDescription entity.

    CREATED -> PARAMETERS_EXTRACTED -> MATCHED -> PRICED
    """

    CREATED = "created"
    PARAMETERS_EXTRACTED = "parameters_extracted"
    MATCHED = "matched"
    PRICED = "priced"


@dataclass
class HVACDescription:
    """
    Mutable entity representing HVAC product description with lifecycle tracking.

    State machine: CREATED -> PARAMETERS_EXTRACTED -> MATCHED -> PRICED

    Attributes:
        id: Unique identifier (UUID4, auto-generated, domain entity ID)
        raw_text: Original description text from Excel (min 3 characters)
        extracted_params: Dictionary of extracted technical parameters
        match_score: Hybrid matching score (None if not matched yet)
        source_row_number: Excel row number for tracking (optional)
        file_id: Source file identifier (optional)
        chromadb_id: ChromaDB document ID format "{file_id}_{row_number}" (optional, for references)
        matched_price: Price from matched reference description (optional)
        state: Current processing state
        created_at: Entity creation timestamp
        updated_at: Last modification timestamp
    """

    # Required fields
    raw_text: str

    # Optional tracking fields
    source_row_number: int | None = None
    file_id: UUID | None = None
    chromadb_id: str | None = None  # ChromaDB document ID: "{file_id}_{row_number}"

    # Processing results (populated during pipeline)
    extracted_params: Optional[ExtractedParameters] = None
    match_score: MatchScore | None = None
    matched_price: Decimal | None = None
    matched_description: str | None = None  # Text of matched reference description

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

        Args:
            raw_text: Description text from Excel cell
            source_row_number: Excel row number (1-based notation)
            file_id: UUID of source Excel file (optional)

        Returns:
            New HVACDescription entity with CREATED state

        Raises:
            InvalidHVACDescriptionError: If raw_text is invalid
        """
        return cls(
            raw_text=raw_text,
            source_row_number=source_row_number,
            file_id=file_id,
        )

    def _validate_text(self, text: str) -> None:
        """
        Validate raw_text meets minimum requirements.

        Raises:
            InvalidHVACDescriptionError: If text is not a string or is too short
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
        """Normalize text by stripping whitespace and collapsing multiple spaces."""
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

        Returns:
            True if raw_text is valid and source_row_number (if provided) is > 0
        """
        try:
            self._validate_text(self.raw_text)
            # If source_row_number is provided, it must be > 0
            if self.source_row_number is not None and self.source_row_number <= 0:
                return False
            return True
        except InvalidHVACDescriptionError:
            return False

    def has_parameters(self) -> bool:
        """Return True if extracted_params is set and contains at least one parameter."""
        return self.extracted_params is not None and self.extracted_params.has_parameters()

    def has_critical_parameters(self) -> bool:
        """Return True if both DN and PN have been extracted."""
        if self.extracted_params is None:
            return False

        return (
            self.extracted_params.dn is not None
            and self.extracted_params.pn is not None
        )

    def extract_parameters(
        self, extractor: "ParameterExtractorProtocol"
    ) -> None:
        """
        Extract technical parameters from raw_text and transition state.

        Calls extractor, stores result, transitions to PARAMETERS_EXTRACTED.
        State transitions to PARAMETERS_EXTRACTED even if no params were found.

        Args:
            extractor: Implementation of ParameterExtractorProtocol

        Raises:
            InvalidHVACDescriptionError: If extractor is None
        """
        if extractor is None:
            raise InvalidHVACDescriptionError("extractor cannot be None")

        # Extract parameters using provided extractor
        self.extracted_params = extractor.extract_parameters(self.raw_text)

        # Transition state to PARAMETERS_EXTRACTED
        self.state = HVACDescriptionState.PARAMETERS_EXTRACTED
        self.updated_at = datetime.now()

    def apply_match_result(self, result: MatchResult) -> None:
        """
        Apply matching result and transition state to MATCHED.

        State must be PARAMETERS_EXTRACTED (enforced). Raises error otherwise.

        Args:
            result: MatchResult value object from matching engine

        Raises:
            InvalidHVACDescriptionError: If result is None, wrong type, or state
                is not PARAMETERS_EXTRACTED
        """
        if result is None:
            raise InvalidHVACDescriptionError("result cannot be None")

        if not isinstance(result, MatchResult):
            raise InvalidHVACDescriptionError(
                f"result must be MatchResult instance, got {type(result).__name__}"
            )

        # Enforce state machine: must be PARAMETERS_EXTRACTED before matching
        if self.state != HVACDescriptionState.PARAMETERS_EXTRACTED:
            raise InvalidHVACDescriptionError(
                f"Cannot apply match result in state {self.state.value}. "
                f"Must be in PARAMETERS_EXTRACTED state first."
            )

        # Apply match result
        self.match_score = result.score
        # Store matched_reference_id for tracking (separate from matched_description)
        # matched_reference_id is UUID from result.matched_item_id

        # Transition state to MATCHED
        self.state = HVACDescriptionState.MATCHED
        self.updated_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Serialize entity to JSON-serializable dictionary, including nested Value Objects."""
        return {
            "id": str(self.id),
            "raw_text": self.raw_text,
            "extracted_params": self.extracted_params.to_dict() if self.extracted_params else None,
            "match_score": self.match_score.to_dict() if self.match_score else None,
            "source_row_number": self.source_row_number,
            "file_id": str(self.file_id) if self.file_id else None,
            "chromadb_id": self.chromadb_id,
            "matched_price": str(self.matched_price) if self.matched_price else None,
            "matched_description": self.matched_description,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HVACDescription":
        """
        Deserialize entity from dictionary (e.g., from to_dict() or Redis).

        Reconstructs nested Value Objects, converts string UUIDs, ISO datetimes,
        state strings, and Decimal prices back to their proper types.

        Args:
            data: Dictionary with entity data

        Returns:
            Reconstructed HVACDescription instance

        Raises:
            KeyError: If 'raw_text' is missing
        """
        # Required field
        raw_text = data["raw_text"]

        # Reconstruct nested Value Objects
        extracted_params = None
        if data.get("extracted_params"):
            extracted_params = ExtractedParameters.from_dict(data["extracted_params"])

        match_score = None
        if data.get("match_score"):
            match_score = MatchScore(
                parameter_score=data["match_score"]["parameter_score"],
                semantic_score=data["match_score"]["semantic_score"],
                final_score=data["match_score"]["final_score"],
                threshold=data["match_score"]["threshold"],
            )

        # Convert string types back to proper types
        id_value = UUID(data["id"]) if data.get("id") else uuid4()
        file_id = UUID(data["file_id"]) if data.get("file_id") else None
        matched_price = (
            Decimal(data["matched_price"]) if data.get("matched_price") else None
        )
        state = HVACDescriptionState(data.get("state", "created"))
        created_at = (
            datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.now()
        )
        updated_at = (
            datetime.fromisoformat(data["updated_at"])
            if data.get("updated_at")
            else datetime.now()
        )

        # Create entity (bypassing __post_init__ validation by using __new__)
        # This is needed because to_dict() includes normalized text,
        # and we don't want to re-normalize or re-validate
        instance = cls.__new__(cls)
        instance.id = id_value
        instance.raw_text = raw_text
        instance.extracted_params = extracted_params
        instance.match_score = match_score
        instance.source_row_number = data.get("source_row_number")
        instance.file_id = file_id
        instance.chromadb_id = data.get("chromadb_id")
        instance.matched_price = matched_price
        instance.matched_description = data.get("matched_description")
        instance.state = state
        instance.created_at = created_at
        instance.updated_at = updated_at
        instance.MIN_TEXT_LENGTH = 3  # Set class constant

        return instance

    def merge_with_price(
        self, price: Decimal, matched_description: str, match_score: MatchScore
    ) -> None:
        """
        Merge price from catalog result and transition state to PRICED.

        Args:
            price: Price from matched reference description
            matched_description: Text of matched reference description (for reporting)
            match_score: Matching score details

        Raises:
            InvalidHVACDescriptionError: If price is negative or match_score is invalid
        """
        if price < 0:
            raise InvalidHVACDescriptionError(f"Price cannot be negative, got {price}")

        if not isinstance(match_score, MatchScore):
            raise InvalidHVACDescriptionError(
                f"match_score must be MatchScore instance, got {type(match_score).__name__}"
            )

        self.matched_price = price
        self.match_score = match_score
        self.matched_description = matched_description
        self.state = HVACDescriptionState.PRICED
        self.updated_at = datetime.now()

    def get_match_report(self) -> str | None:
        """
        Generate human-readable matching report for Excel export.

        Format: "Matched: <description> | Score: <score>% | DN: <dn> | PN: <pn>"

        Returns:
            Formatted report string or None if not yet matched
        """
        if not self.match_score or self.state not in (
            HVACDescriptionState.MATCHED,
            HVACDescriptionState.PRICED,
        ):
            return None

        matched_desc = self.matched_description or "N/A"
        score_pct = self.match_score.final_score

        report_parts = [f"Matched: {matched_desc}", f"Score: {score_pct:.1f}%"]

        if self.matched_price:
            report_parts.append(f"Price: {self.matched_price} PLN")

        # Add key parameters if available
        if self.extracted_params and self.extracted_params.dn is not None:
            report_parts.append(f"DN: {self.extracted_params.dn}")

        if self.extracted_params and self.extracted_params.pn is not None:
            report_parts.append(f"PN: {self.extracted_params.pn}")

        return " | ".join(report_parts)

    def __repr__(self) -> str:
        """Developer-friendly representation for debugging."""
        text_preview = self.raw_text[:50] + ("..." if len(self.raw_text) > 50 else "")
        return (
            f"HVACDescription(id={self.id}, "
            f"raw_text='{text_preview}', "
            f"state={self.state.value})"
        )

    def __str__(self) -> str:
        """User-friendly representation."""
        return f"{self.raw_text} [{self.state.value}]"
