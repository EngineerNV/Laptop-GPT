"""
Tests for cli.py module.

This module tests:
- Command line argument parsing
- Configuration help functionality
- Version handling
- Argument validation
"""
import pytest
import sys
from unittest.mock import patch
from io import StringIO

# Import the modules to test
sys.path.append('..')
from cli import parse_arguments, get_config_help


class TestParseArguments:
    """Test command line argument parsing."""
    
    def test_parse_default_arguments(self):
        """Test parsing with default arguments."""
        with patch.object(sys, 'argv', ['main.py']):
            args = parse_arguments()
            
            assert args.log_level == "INFO"
            assert args.debug == False
            assert args.config == "config.json"
            assert args.help_config == False

    def test_parse_debug_flag(self):
        """Test parsing with debug flag."""
        with patch.object(sys, 'argv', ['main.py', '--debug']):
            args = parse_arguments()
            
            assert args.debug == True
            assert args.log_level == "INFO"  # Default should remain

    def test_parse_log_level(self):
        """Test parsing with different log levels."""
        log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        
        for level in log_levels:
            with patch.object(sys, 'argv', ['main.py', '--log-level', level]):
                args = parse_arguments()
                assert args.log_level == level

    def test_parse_custom_config(self):
        """Test parsing with custom config file."""
        with patch.object(sys, 'argv', ['main.py', '--config', 'custom.json']):
            args = parse_arguments()
            
            assert args.config == "custom.json"

    def test_parse_help_config_flag(self):
        """Test parsing with help-config flag."""
        with patch.object(sys, 'argv', ['main.py', '--help-config']):
            with pytest.raises(SystemExit) as exc_info:
                with patch('builtins.print') as mock_print:
                    parse_arguments()
            
            # Should exit with code 0 (success)
            assert exc_info.value.code == 0
            # Should have printed the help
            mock_print.assert_called()

    def test_parse_combined_arguments(self):
        """Test parsing with multiple arguments combined."""
        test_args = ['main.py', '--debug', '--log-level', 'DEBUG', '--config', 'test.json']
        
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            
            assert args.debug == True
            assert args.log_level == "DEBUG"
            assert args.config == "test.json"

    def test_parse_version_flag(self):
        """Test parsing with version flag."""
        with patch.object(sys, 'argv', ['main.py', '--version']):
            with pytest.raises(SystemExit) as exc_info:
                with patch('builtins.print') as mock_print:
                    parse_arguments()
            
            # Should exit with code 0 (success)
            assert exc_info.value.code == 0

    def test_parse_invalid_log_level(self):
        """Test parsing with invalid log level."""
        with patch.object(sys, 'argv', ['main.py', '--log-level', 'INVALID']):
            with pytest.raises(SystemExit) as exc_info:
                with patch('sys.stderr', new_callable=StringIO):
                    parse_arguments()
            
            # Should exit with error code
            assert exc_info.value.code != 0


class TestGetConfigHelp:
    """Test configuration help functionality."""
    
    def test_get_config_help_content(self):
        """Test that config help returns expected content."""
        help_text = get_config_help()
        
        # Check for key sections
        assert "Configuration File Help" in help_text
        assert "config.json" in help_text
        assert "Structure:" in help_text
        
        # Check for main configuration sections
        assert '"model"' in help_text
        assert '"custom_llm_params"' in help_text
        assert '"environment"' in help_text
        assert '"app"' in help_text

    def test_get_config_help_examples(self):
        """Test that config help includes examples."""
        help_text = get_config_help()
        
        # Check for example values
        assert "n_ctx" in help_text
        assert "temperature" in help_text
        assert "Parameter Guidelines" in help_text

    def test_get_config_help_explanations(self):
        """Test that config help includes parameter explanations."""
        help_text = get_config_help()
        
        # Check for explanations
        assert "Context window size" in help_text
        assert "Sampling temperature" in help_text
        assert "Number of threads" in help_text

    def test_get_config_help_guidance(self):
        """Test that config help includes usage guidance."""
        help_text = get_config_help()
        
        # Check for guidance
        assert "optional" in help_text.lower()
        assert "default" in help_text.lower()
        assert "customize" in help_text.lower()


class TestCLIIntegration:
    """Integration tests for CLI functionality."""
    
    def test_help_config_workflow(self):
        """Test the complete help-config workflow."""
        with patch.object(sys, 'argv', ['main.py', '--help-config']):
            with pytest.raises(SystemExit) as exc_info:
                with patch('builtins.print') as mock_print:
                    parse_arguments()
            
            # Verify exit code
            assert exc_info.value.code == 0
            
            # Verify help was printed
            mock_print.assert_called_once()
            printed_text = mock_print.call_args[0][0]
            assert "Configuration File Help" in printed_text

    def test_argument_parsing_edge_cases(self):
        """Test edge cases in argument parsing."""
        # Test with no arguments
        with patch.object(sys, 'argv', ['main.py']):
            args = parse_arguments()
            assert hasattr(args, 'log_level')
            assert hasattr(args, 'debug')
            assert hasattr(args, 'config')

    def test_boolean_flags_behavior(self):
        """Test boolean flag behavior."""
        # Test debug flag present
        with patch.object(sys, 'argv', ['main.py', '--debug']):
            args = parse_arguments()
            assert args.debug is True
        
        # Test debug flag absent
        with patch.object(sys, 'argv', ['main.py']):
            args = parse_arguments()
            assert args.debug is False

    def test_string_argument_handling(self):
        """Test string argument handling."""
        test_cases = [
            (['main.py', '--config', 'path/to/config.json'], 'path/to/config.json'),
            (['main.py', '--config', 'simple.json'], 'simple.json'),
            (['main.py', '--config', '/absolute/path.json'], '/absolute/path.json'),
        ]
        
        for argv, expected_config in test_cases:
            with patch.object(sys, 'argv', argv):
                args = parse_arguments()
                assert args.config == expected_config
