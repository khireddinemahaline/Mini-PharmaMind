"""
PharmaMind Test Suite

This package contains tests for the PharmaMind pharmaceutical research system.

Test Structure:
    - test_tools/: Tests for agent tools and utilities
    - test_agents/: Tests for agent functionality (to be added)
    - test_integration/: Integration tests (to be added)

Running Tests:
    # Run all tests
    pytest tests/
    
    # Run with coverage
    pytest --cov=. --cov-report=html tests/
    
    # Run specific test file
    pytest tests/test_tools/test_chembl.py
    
    # Run with verbose output
    pytest -v tests/

Requirements:
    - pytest
    - pytest-asyncio (for async tests)
    - pytest-cov (for coverage reports)
"""
