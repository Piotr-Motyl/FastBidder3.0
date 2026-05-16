"""
ParameterExtractorProtocol — Interface for extracting HVAC parameters from text.
Stateless service using regex + domain dictionaries. Returns ExtractedParameters VO.
"""

from typing import Protocol, Optional
from src.domain.hvac.value_objects.extracted_parameters import ExtractedParameters


class ParameterExtractorProtocol(Protocol):
    """
    Protocol for HVAC parameter extraction from text descriptions.

    Extracts DN, PN, valve type, material, drive type, voltage, and manufacturer
    using regex patterns and domain dictionaries. Returns canonical forms with
    confidence scores (1.0 = exact match, 0.9 = synonym match, 0.0 = not found).
    """

    def extract_parameters(self, text: str) -> ExtractedParameters:
        """
        Extract all parameters from text. Returns empty ExtractedParameters (not None) when nothing found.

        Extraction order: DN/PN first, then valve type, material, drive, voltage, manufacturer.
        Text is normalized (lowercase, stripped) before processing.
        """
        ...

    def extract_dn(self, text: str) -> tuple[Optional[int], float]:
        """
        Extract DN value. Supported formats: DN50, DN 50, DN-50, Ø50.
        Returns (value, confidence). Confidence 1.0 for standard notation, 0.8 for word pattern.
        """
        ...

    def extract_pn(self, text: str) -> tuple[Optional[int], float]:
        """
        Extract PN value. Supported formats: PN16, PN 16, PN-16, "16 bar", "ciśnienie 16".
        Returns (value, confidence). Confidence 1.0 for standard notation, 0.8 for word pattern.
        """
        ...

    def extract_valve_type(self, text: str) -> tuple[Optional[str], float]:
        """
        Extract valve type via dictionary matching with synonym resolution.
        Returns canonical form (e.g., "kurek kulowy" → "kulowy"). Longest match wins.
        """
        ...

    def extract_material(self, text: str) -> tuple[Optional[str], float]:
        """
        Extract material via dictionary matching with synonym resolution.
        Returns canonical form (e.g., "mosiężny" → "mosiądz").
        """
        ...

    def extract_drive_type(self, text: str) -> tuple[Optional[str], float]:
        """
        Extract drive type via dictionary matching with synonym resolution.
        Returns canonical form (e.g., "siłownik elektryczny" → "elektryczny").
        """
        ...

    def extract_voltage(self, text: str) -> tuple[Optional[str], float]:
        """Extract voltage string (e.g., "230V"). Only meaningful when drive_type is "elektryczny"."""
        ...

    def extract_manufacturer(self, text: str) -> tuple[Optional[str], float]:
        """
        Extract manufacturer via exact dictionary match (KSB, Danfoss, Belimo, Grundfos, etc.).
        Returns uppercase name. No synonym resolution for manufacturers.
        """
        ...
