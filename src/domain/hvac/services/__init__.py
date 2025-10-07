"""
HVAC Domain Services

Responsibility:
    Business operations that don't naturally fit into entities.
    Stateless services operating on domain objects.

Contains:
    To be implemented in Phase 2-3:
    - ParameterExtractor - Extract DN, PN, materials from text
    - MatchingEngine - Calculate similarity between descriptions
    - ScoringEngine - Calculate match scores with business rules

Does NOT contain:
    - Infrastructure concerns (file I/O, database)
    - Application orchestration (use Application services)
"""
