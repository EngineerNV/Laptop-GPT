"""
Tests for config_utils.py module.

This module tests:
- Configuration file loading
- Error handling for missing/invalid configs
- Environment setup
- Logging configuration
"""
import pytest
import json
import os
import tempfile
import logging
from unittest.mock import patch, MagicMock

# Import the modules to test
import sys
sys.path.append('..')
from config_utils import load_config, setup_logging, setup_environment


class TestLoadConfig:
    """Test configuration loading functionality."""
    
    def test_load_valid_config(self, valid_config_path):
        """Test loading a valid configuration file."""
        config = load_config(valid_config_path)
        
        # Verify structure
        assert "model" in config
        assert "llm_params" in config
        assert "environment" in config
        assert "app" in config
        
        # Verify specific values
        assert config["model"]["repo"] == "test/model-repo"
        assert config["llm_params"]["n_ctx"] == 1024
        assert config["app"]["name"] == "Test-GPT"

    def test_load_nonexistent_config(self, nonexistent_config_path):
        """Test that FileNotFoundError is raised for missing config."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_config(nonexistent_config_path)
        
        assert "Required configuration file" in str(exc_info.value)

    def test_load_invalid_json_config(self, invalid_config_path):
        """Test that JSONDecodeError is raised for invalid JSON."""
        with pytest.raises(json.JSONDecodeError):
            load_config(invalid_config_path)

    def test_load_config_with_custom_path(self, temp_dir):
        """Test loading config from a custom path."""
        custom_config = {
            "model": {"repo": "custom/repo", "file": "custom.gguf", "cache_dir": "/tmp"},
            "llm_params": {"n_ctx": 512},
            "app": {"name": "Custom-GPT"}
        }
        
        custom_path = os.path.join(temp_dir, "custom_config.json")
        with open(custom_path, 'w') as f:
            json.dump(custom_config, f)
        
        config = load_config(custom_path)
        assert config["model"]["repo"] == "custom/repo"
        assert config["llm_params"]["n_ctx"] == 512


class TestSetupLogging:
    """Test logging configuration."""
    
    def test_setup_logging_info_level(self):
        """Test setting up logging at INFO level."""
        setup_logging("INFO")
        logger = logging.getLogger(__name__)
        
        assert logger.isEnabledFor(logging.INFO)
        assert not logger.isEnabledFor(logging.DEBUG)

    def test_setup_logging_debug_level(self):
        """Test setting up logging at DEBUG level."""
        setup_logging("DEBUG")
        logger = logging.getLogger("test")
        
        assert logger.isEnabledFor(logging.DEBUG)
        assert logger.isEnabledFor(logging.INFO)

    def test_setup_logging_error_level(self):
        """Test setting up logging at ERROR level."""
        setup_logging("ERROR")
        logger = logging.getLogger(__name__)
        
        assert logger.isEnabledFor(logging.ERROR)
        assert not logger.isEnabledFor(logging.WARNING)


class TestSetupEnvironment:
    """Test environment setup functionality."""
    
    @patch('warnings.filterwarnings')
    def test_setup_environment_debug_mode(self, mock_filter_warnings, valid_config_path):
        """Test environment setup in debug mode."""
        config = load_config(valid_config_path)
        
        with patch.dict(os.environ, {}, clear=True):
            setup_environment(config, debug_mode=True, log_level="INFO")
            
            # Check environment variables are set
            assert os.environ.get("LLAMA_LOG_LEVEL") == "debug"
            assert os.environ.get("LLAMA_CPP_VERBOSE") == "0"  # From config
            assert os.environ.get("GGML_LOG_LEVEL") == "2"     # From config

    @patch('warnings.filterwarnings')
    def test_setup_environment_normal_mode(self, mock_filter_warnings, valid_config_path):
        """Test environment setup in normal mode."""
        config = load_config(valid_config_path)
        
        with patch.dict(os.environ, {}, clear=True):
            setup_environment(config, debug_mode=False, log_level="INFO")
            
            # Check environment variables are set
            assert os.environ.get("LLAMA_LOG_LEVEL") == "info"

    @patch('warnings.filterwarnings')
    def test_setup_environment_error_level(self, mock_filter_warnings, valid_config_path):
        """Test environment setup with ERROR log level."""
        config = load_config(valid_config_path)
        
        with patch.dict(os.environ, {}, clear=True):
            setup_environment(config, debug_mode=False, log_level="ERROR")
            
            # Check that warnings are filtered
            mock_filter_warnings.assert_called()
            assert os.environ.get("LLAMA_LOG_LEVEL") == "error"

    def test_setup_environment_missing_environment_section(self):
        """Test environment setup with missing environment section in config."""
        config = {
            "model": {"repo": "test", "file": "test.gguf", "cache_dir": "/tmp"},
            "llm_params": {"n_ctx": 1024}
            # No "environment" section
        }
        
        with patch.dict(os.environ, {}, clear=True):
            # Should not raise an error
            setup_environment(config, debug_mode=False, log_level="INFO")
            assert os.environ.get("LLAMA_LOG_LEVEL") == "info"


class TestConfigIntegration:
    """Integration tests for configuration functionality."""
    
    def test_complete_config_workflow(self, temp_dir):
        """Test the complete workflow from config file to environment setup."""
        # Create a complete config
        config_data = {
            "model": {
                "repo": "test/integration-model",
                "file": "integration.gguf",
                "cache_dir": temp_dir
            },
            "llm_params": {
                "n_ctx": 2048,
                "temperature": 0.7,
                "max_tokens": 1024
            },
            "environment": {
                "custom_var": "test_value",
                "another_var": "123"
            },
            "app": {
                "name": "Integration-Test-GPT",
                "version": "1.0.0"
            }
        }
        
        config_path = os.path.join(temp_dir, "integration_config.json")
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        # Load and setup
        config = load_config(config_path)
        
        with patch.dict(os.environ, {}, clear=True):
            setup_environment(config, debug_mode=False, log_level="INFO")
            
            # Verify config loaded correctly
            assert config["app"]["name"] == "Integration-Test-GPT"
            
            # Verify environment variables set
            assert os.environ.get("CUSTOM_VAR") == "test_value"
            assert os.environ.get("ANOTHER_VAR") == "123"
