"""
Pytest Configuration for Redis Tests.

This local conftest overrides global conftest to avoid FastAPI dependencies
for unit tests.
"""

import pytest
