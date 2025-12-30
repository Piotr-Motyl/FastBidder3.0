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
        - source_row_number (if provided) is > 0
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
            >>> desc = HVACDescription(raw_text="Zawór DN50", source_row_number=0)
            >>> desc.is_valid()
            False
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
        """
        Check if technical parameters have been extracted.

        Returns:
            True if extracted_params is set and contains at least one parameter

        Examples:
            >>> desc = HVACDescription(raw_text="Zawór DN50")
            >>> desc.has_parameters()
            False
            >>> desc.extracted_params = ExtractedParameters(dn=50, confidence_scores={"dn": 1.0})
            >>> desc.has_parameters()
            True
        """
        return self.extracted_params is not None and self.extracted_params.has_parameters()

    def has_critical_parameters(self) -> bool:
        """
        Check if critical technical parameters (DN and PN) have been extracted.

        Critical parameters are those essential for matching:
        - DN (Diameter Nominal)
        - PN (Pressure Nominal)

        Returns:
            True if both DN and PN are extracted and not None

        Business Logic:
            DN and PN are the most important parameters for HVAC equipment matching.
            Without them, matching quality is significantly degraded.
            This method is used for fast-fail logic in matching engine.

        Examples:
            >>> desc = HVACDescription(raw_text="Zawór DN50 PN16")
            >>> desc.has_critical_parameters()
            False  # Not extracted yet
            >>> desc.extracted_params = ExtractedParameters(
            ...     dn=50, pn=16, confidence_scores={"dn": 1.0, "pn": 1.0}
            ... )
            >>> desc.has_critical_parameters()
            True
            >>> desc.extracted_params = ExtractedParameters(
            ...     dn=50, confidence_scores={"dn": 1.0}
            ... )
            >>> desc.has_critical_parameters()
            False  # PN missing
        """
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
        Extract technical parameters from raw_text using provided extractor.

        This method orchestrates parameter extraction and state transition:
        1. Calls extractor.extract_parameters(raw_text)
        2. Stores extracted parameters in extracted_params field
        3. Transitions state from CREATED to PARAMETERS_EXTRACTED
        4. Updates updated_at timestamp

        Args:
            extractor: Implementation of ParameterExtractorProtocol

        Raises:
            InvalidHVACDescriptionError: If extractor is None or invalid type

        State Transitions:
            CREATED -> PARAMETERS_EXTRACTED (success)
            No state change if extraction returns empty parameters

        Examples:
            >>> from src.domain.hvac.services.concrete_parameter_extractor import ConcreteParameterExtractor
            >>> desc = HVACDescription(raw_text="Zawór kulowy DN50 PN16")
            >>> desc.state
            <HVACDescriptionState.CREATED: 'created'>
            >>> extractor = ConcreteParameterExtractor()
            >>> desc.extract_parameters(extractor)
            >>> desc.state
            <HVACDescriptionState.PARAMETERS_EXTRACTED: 'parameters_extracted'>
            >>> desc.extracted_params.dn
            50
            >>> desc.extracted_params.pn
            16

        Business Logic:
            - Can only be called once (idempotent - calling again will re-extract)
            - Extraction can return empty parameters (all None) - this is valid
            - State transitions to PARAMETERS_EXTRACTED even if no params found
            - This decouples extraction logic from entity (Dependency Inversion)

        Architecture Note:
            Uses Protocol (ParameterExtractorProtocol) for dependency inversion.
            Entity doesn't know about ConcreteParameterExtractor implementation.
            This allows testing with mock extractors.
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
        Apply matching result to description.

        Updates entity with matching result and transitions state to MATCHED:
        1. Extracts MatchScore from MatchResult
        2. Stores matched_reference_id for tracking
        3. Transitions state from PARAMETERS_EXTRACTED to MATCHED
        4. Updates updated_at timestamp

        Args:
            result: MatchResult value object from matching engine

        Raises:
            InvalidHVACDescriptionError: If result is None or invalid type
            InvalidHVACDescriptionError: If state is not PARAMETERS_EXTRACTED

        State Transitions:
            PARAMETERS_EXTRACTED -> MATCHED (success)
            Raises error if called from other states

        Examples:
            >>> desc = HVACDescription(raw_text="Zawór DN50")
            >>> desc.state = HVACDescriptionState.PARAMETERS_EXTRACTED
            >>> result = MatchResult(
            ...     matched_item_id=UUID("..."),
            ...     score=MatchScore.create(100.0, 90.0, 75.0),
            ...     confidence=0.95,
            ...     message="High confidence match",
            ...     breakdown={}
            ... )
            >>> desc.apply_match_result(result)
            >>> desc.state
            <HVACDescriptionState.MATCHED: 'matched'>
            >>> desc.match_score.final_score
            95.2

        Business Logic:
            - Must be called AFTER extract_parameters()
            - State must be PARAMETERS_EXTRACTED (enforced)
            - Stores only MatchScore (not full MatchResult)
            - matched_reference_id is stored separately for tracking

        Architecture Note:
            Enforces state machine transition rules.
            Cannot match without extracting parameters first.
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
        Deserialize entity from dictionary.

        Creates HVACDescription instance from dictionary representation
        (typically from to_dict() output or storage/API).

        Args:
            data: Dictionary with entity data

        Returns:
            Reconstructed HVACDescription instance

        Raises:
            InvalidHVACDescriptionError: If required fields are missing or invalid
            KeyError: If 'raw_text' is missing

        Examples:
            >>> data = {
            ...     "id": "550e8400-e29b-41d4-a716-446655440000",
            ...     "raw_text": "Zawór DN50",
            ...     "state": "created",
            ...     "created_at": "2024-01-15T10:30:00",
            ...     "updated_at": "2024-01-15T10:30:00",
            ...     "source_row_number": 5,
            ...     "file_id": None,
            ...     "extracted_params": None,
            ...     "match_score": None,
            ...     "matched_price": None,
            ...     "matched_description": None
            ... }
            >>> desc = HVACDescription.from_dict(data)
            >>> desc.raw_text
            'Zawór DN50'
            >>> desc.state
            <HVACDescriptionState.CREATED: 'created'>

        Business Logic:
            - Reconstructs nested Value Objects (ExtractedParameters, MatchScore)
            - Converts string UUIDs back to UUID objects
            - Converts ISO datetime strings back to datetime objects
            - Converts state string back to HVACDescriptionState enum
            - Converts price string back to Decimal

        Architecture Note:
            Enables roundtrip serialization/deserialization for persistence.
            Used by repository implementations to reconstruct entities from storage.
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
        self.matched_description = matched_description
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
        """
        Developer-friendly representation.

        Returns:
            String representation for debugging
        """
        text_preview = self.raw_text[:50] + ("..." if len(self.raw_text) > 50 else "")
        return (
            f"HVACDescription(id={self.id}, "
            f"raw_text='{text_preview}', "
            f"state={self.state.value})"
        )

    def __str__(self) -> str:
        """
        User-friendly representation.

        Returns:
            Readable string representation
        """
        return f"{self.raw_text} [{self.state.value}]"
