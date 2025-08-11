# pytest configuration
import pytest
import tempfile
import os
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def valid_config_path():
    """Path to a valid test configuration file."""
    return str(Path(__file__).parent / "fixtures" / "valid_config.json")


@pytest.fixture
def incomplete_config_path():
    """Path to an incomplete test configuration file."""
    return str(Path(__file__).parent / "fixtures" / "incomplete_config.json")


@pytest.fixture
def invalid_config_path():
    """Path to an invalid JSON configuration file."""
    return str(Path(__file__).parent / "fixtures" / "invalid_config.json")


@pytest.fixture
def nonexistent_config_path():
    """Path to a config file that doesn't exist."""
    return "/path/that/does/not/exist/config.json"


@pytest.fixture
def mock_model_file(temp_dir):
    """Create a mock model file for testing."""
    model_path = os.path.join(temp_dir, "test_model.gguf")
    # Create a file with some content
    with open(model_path, 'wb') as f:
        f.write(b"fake model data" * 1000)  # Make it a reasonable size
    return model_path


@pytest.fixture
def sample_llm_params():
    """Sample LLM parameters for testing."""
    return {
        "n_ctx": 1024,
        "n_batch": 4,
        "n_threads": 2,
        "temperature": 0.5,
        "max_tokens": 512,
        "use_mlock": False,
        "use_mmap": True,
        "n_gpu_layers": 0
    }
