"""
ConcreteParameterExtractor - Implementation of ParameterExtractorProtocol

Concrete implementation of HVAC parameter extraction using regex patterns
and domain dictionaries. Extracts technical parameters (DN, PN, valve type,
material, drive type, voltage, manufacturer) from text descriptions.

Architecture Notes:
- Part of Infrastructure Layer (implements Domain Protocol)
- Implements ParameterExtractorProtocol (dependency inversion)
- Stateless service (no internal state between calls)
- Uses domain patterns and constants (separation of concerns)
- Returns ExtractedParameters Value Object

Performance:
- Target: <5ms per extraction
- No external API calls (all local regex/dict matching)
"""

from typing import Optional

from src.domain.hvac.patterns import (
    extract_dn_from_text,
    extract_pn_from_text,
    extract_voltage_from_text,
    normalize_text,
)
from src.domain.hvac.constants import (
    VALVE_TYPES,
    VALVE_SYNONYMS,
    MATERIALS,
    MATERIAL_SYNONYMS,
    DRIVE_TYPES,
    DRIVE_SYNONYMS,
    MANUFACTURERS,
)
from src.domain.hvac.value_objects.extracted_parameters import ExtractedParameters


class ConcreteParameterExtractor:
    """
    Concrete implementation of ParameterExtractorProtocol.

    Extracts HVAC technical parameters from text descriptions using:
    - Regex patterns for DN, PN, voltage
    - Dictionary matching for valve type, material, drive type, manufacturer
    - Synonym resolution for canonical forms
    - Confidence scoring (0.0-1.0) for each extraction

    This is a stateless service - no internal state between calls.
    Each extraction is independent and doesn't affect others.

    Examples:
        >>> extractor = ConcreteParameterExtractor()
        >>> text = "Zawór kulowy DN50 PN16 mosiądz napęd elektryczny 230V KSB"
        >>> params = extractor.extract_parameters(text)
        >>> params.dn
        50
        >>> params.valve_type
        'kulowy'
        >>> params.get_average_confidence()
        0.95
    """

    def extract_parameters(self, text: str) -> ExtractedParameters:
        """
        Extract all HVAC parameters from text description.

        Main orchestration method that calls all specific extraction methods
        and combines results into a single ExtractedParameters object.

        Args:
            text: HVAC equipment description text

        Returns:
            ExtractedParameters with all extracted values and confidence scores

        Algorithm:
            1. Normalize text (lowercase, trim, remove double spaces)
            2. Extract DN, PN, valve_type, material, drive_type, voltage, manufacturer
            3. Build confidence_scores dict (only non-None values)
            4. Return ExtractedParameters object
        """
        # 1. Normalize text for consistent matching
        normalized = normalize_text(text)

        # 2. Extract all parameters
        dn, dn_conf = self.extract_dn(normalized)
        pn, pn_conf = self.extract_pn(normalized)
        valve_type, valve_conf = self.extract_valve_type(normalized)
        material, material_conf = self.extract_material(normalized)
        drive_type, drive_conf = self.extract_drive_type(normalized)
        voltage, voltage_conf = self.extract_voltage(normalized)
        manufacturer, manuf_conf = self.extract_manufacturer(normalized)

        # 3. Build confidence scores dict (only for non-None values)
        confidence_scores = {}
        if dn is not None:
            confidence_scores["dn"] = dn_conf
        if pn is not None:
            confidence_scores["pn"] = pn_conf
        if valve_type is not None:
            confidence_scores["valve_type"] = valve_conf
        if material is not None:
            confidence_scores["material"] = material_conf
        if drive_type is not None:
            confidence_scores["drive_type"] = drive_conf
        if voltage is not None:
            confidence_scores["voltage"] = voltage_conf
        if manufacturer is not None:
            confidence_scores["manufacturer"] = manuf_conf

        # 4. Return ExtractedParameters
        return ExtractedParameters(
            dn=dn,
            pn=pn,
            valve_type=valve_type,
            material=material,
            drive_type=drive_type,
            voltage=voltage,
            manufacturer=manufacturer,
            confidence_scores=confidence_scores,
        )

    def extract_dn(self, text: str) -> tuple[Optional[int], float]:
        """
        Extract DN (Diameter Nominal) value from text.

        Args:
            text: Text to search for DN value

        Returns:
            Tuple of (dn_value, confidence_score)
        """
        return extract_dn_from_text(text)

    def extract_pn(self, text: str) -> tuple[Optional[int], float]:
        """
        Extract PN (Pressure Nominal) value from text.

        Args:
            text: Text to search for PN value

        Returns:
            Tuple of (pn_value, confidence_score)
        """
        return extract_pn_from_text(text)

    def extract_valve_type(self, text: str) -> tuple[Optional[str], float]:
        """
        Extract valve type from text using dictionary matching.

        Args:
            text: Text to search for valve type (will be normalized)

        Returns:
            Tuple of (valve_type, confidence_score)
        """
        normalized_text = normalize_text(text)
        sorted_valve_types = sorted(VALVE_TYPES, key=len, reverse=True)

        for valve_type in sorted_valve_types:
            if valve_type in normalized_text:
                if valve_type in VALVE_SYNONYMS:
                    canonical = VALVE_SYNONYMS[valve_type]
                    return canonical, 0.9
                else:
                    return valve_type, 1.0

        return None, 0.0

    def extract_material(self, text: str) -> tuple[Optional[str], float]:
        """
        Extract material from text using dictionary matching.

        Args:
            text: Text to search for material (will be normalized)

        Returns:
            Tuple of (material, confidence_score)
        """
        normalized_text = normalize_text(text)
        sorted_materials = sorted(MATERIALS, key=len, reverse=True)

        for material in sorted_materials:
            if material.lower() in normalized_text:
                if material in MATERIAL_SYNONYMS:
                    canonical = MATERIAL_SYNONYMS[material]
                    return canonical, 0.9
                else:
                    return material, 1.0

        return None, 0.0

    def extract_drive_type(self, text: str) -> tuple[Optional[str], float]:
        """
        Extract drive/actuation type from text using dictionary matching.

        Args:
            text: Text to search for drive type (will be normalized)

        Returns:
            Tuple of (drive_type, confidence_score)
        """
        normalized_text = normalize_text(text)
        sorted_drive_types = sorted(DRIVE_TYPES, key=len, reverse=True)

        for drive_type in sorted_drive_types:
            if drive_type in normalized_text:
                if drive_type in DRIVE_SYNONYMS:
                    canonical = DRIVE_SYNONYMS[drive_type]
                    return canonical, 0.9
                else:
                    return drive_type, 1.0

        return None, 0.0

    def extract_voltage(self, text: str) -> tuple[Optional[str], float]:
        """
        Extract voltage from text.

        Args:
            text: Text to search for voltage

        Returns:
            Tuple of (voltage, confidence_score)
        """
        return extract_voltage_from_text(text)

    def extract_manufacturer(self, text: str) -> tuple[Optional[str], float]:
        """
        Extract manufacturer name from text using dictionary matching.

        Args:
            text: Text to search for manufacturer (will be normalized)

        Returns:
            Tuple of (manufacturer, confidence_score)
        """
        normalized_text = normalize_text(text)
        sorted_manufacturers = sorted(MANUFACTURERS, key=len, reverse=True)

        for manufacturer in sorted_manufacturers:
            if manufacturer.lower() in normalized_text:
                return manufacturer.upper(), 1.0

        return None, 0.0
