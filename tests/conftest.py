import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_robots_checker():
    """Mock for RobotsChecker to bypass network calls in tests."""
    checker = MagicMock()
    checker.get_crawl_delay.return_value = 2.0
    checker.can_fetch.return_value = True
    return checker

@pytest.fixture
def tmp_data_raw(tmp_path):
    """Provides a temporary path for raw data saving."""
    data_raw = tmp_path / "data/raw"
    data_raw.mkdir(parents=True, exist_ok=True)
    return data_raw
