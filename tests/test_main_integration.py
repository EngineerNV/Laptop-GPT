"""
Integration tests for the main application.

This module tests:
- Complete application workflow
- Configuration loading and validation
- Error handling and user feedback
- Integration between all modules
"""
import pytest
import sys
import json
import tempfile
import os
from unittest.mock import patch, MagicMock
from io import StringIO

# Import the modules to test
sys.path.append('..')
import main
from config_utils import load_config


class TestMainApplicationFlow:
    """Test the main application flow."""
    
    def test_main_with_missing_config(self, nonexistent_config_path):
        """Test main function behavior with missing config file."""
        test_args = ['main.py', '--config', nonexistent_config_path]
        
        with patch.object(sys, 'argv', test_args):
            with patch('main.Console') as mock_console_class:
                mock_console = MagicMock()
                mock_console_class.return_value = mock_console
                
                result = main.main()
                
                # Should return error code 1
                assert result == 1
                
                # Should have printed error message
                mock_console.print.assert_called()
                error_calls = [call for call in mock_console.print.call_args_list 
                             if 'Required configuration file' in str(call)]
                assert len(error_calls) > 0

    def test_main_with_incomplete_config(self, incomplete_config_path):
        """Test main function with incomplete configuration."""
        test_args = ['main.py', '--config', incomplete_config_path]
        
        with patch.object(sys, 'argv', test_args):
            with patch('main.Console') as mock_console_class:
                mock_console = MagicMock()
                mock_console_class.return_value = mock_console
                
                result = main.main()
                
                # Should return error code 1
                assert result == 1
                
                # Should have printed missing configuration error
                mock_console.print.assert_called()

    @patch('main.download_model')
    @patch('main.setup_llm')
    @patch('builtins.input', side_effect=[':q'])  # Simulate user quitting
    def test_main_successful_flow(self, mock_input, mock_setup_llm, mock_download, 
                                valid_config_path, temp_dir):
        """Test successful main application flow."""
        # Setup mocks
        mock_model_path = os.path.join(temp_dir, "test_model.gguf")
        mock_download.return_value = mock_model_path
        
        mock_llm = MagicMock()
        mock_setup_llm.return_value = mock_llm
        
        test_args = ['main.py', '--config', valid_config_path]
        
        with patch.object(sys, 'argv', test_args):
            with patch('main.Console') as mock_console_class:
                mock_console = MagicMock()
                mock_console_class.return_value = mock_console
                
                result = main.main()
                
                # Should return success code 0
                assert result == 0
                
                # Verify download was called
                mock_download.assert_called_once()
                
                # Verify LLM setup was called
                mock_setup_llm.assert_called_once()

    @patch('main.download_model')
    @patch('main.setup_llm')
    def test_main_llm_setup_failure(self, mock_setup_llm, mock_download, 
                                   valid_config_path, temp_dir):
        """Test main function when LLM setup fails."""
        # Setup mocks
        mock_model_path = os.path.join(temp_dir, "test_model.gguf")
        mock_download.return_value = mock_model_path
        mock_setup_llm.side_effect = Exception("Failed to load model")
        
        test_args = ['main.py', '--config', valid_config_path]
        
        with patch.object(sys, 'argv', test_args):
            with patch('main.Console') as mock_console_class:
                mock_console = MagicMock()
                mock_console_class.return_value = mock_console
                
                result = main.main()
                
                # Should return error code 1
                assert result == 1
                
                # Should have printed LLM failure message
                error_calls = [call for call in mock_console.print.call_args_list 
                             if 'Failed to load LLM' in str(call)]
                assert len(error_calls) > 0

    def test_main_keyboard_interrupt(self, valid_config_path):
        """Test main function handling of keyboard interrupt."""
        test_args = ['main.py', '--config', valid_config_path]
        
        with patch.object(sys, 'argv', test_args):
            with patch('main.load_config', side_effect=KeyboardInterrupt()):
                with patch('builtins.print') as mock_print:
                    result = main.main()
                    
                    # Should return error code 1
                    assert result == 1
                    
                    # Should have printed exit message
                    mock_print.assert_called_with("\nExiting...")


class TestConfigurationIntegration:
    """Test configuration integration with main application."""
    
    def test_config_validation_in_main(self, temp_dir):
        """Test that main properly validates all required config sections."""
        # Create config missing required sections
        incomplete_configs = [
            {},  # Empty config
            {"model": {}},  # Missing model details
            {"model": {"repo": "test"}, "llm_params": {}},  # Missing model file/cache
            {"model": {"repo": "test", "file": "test.gguf", "cache_dir": "/tmp"}},  # Missing llm_params
        ]
        
        for i, config_data in enumerate(incomplete_configs):
            config_path = os.path.join(temp_dir, f"incomplete_config_{i}.json")
            with open(config_path, 'w') as f:
                json.dump(config_data, f)
            
            test_args = ['main.py', '--config', config_path]
            
            with patch.object(sys, 'argv', test_args):
                with patch('main.Console') as mock_console_class:
                    mock_console = MagicMock()
                    mock_console_class.return_value = mock_console
                    
                    result = main.main()
                    
                    # Should fail for incomplete config
                    assert result == 1

    def test_environment_setup_integration(self, valid_config_path):
        """Test that environment setup works correctly with main."""
        test_args = ['main.py', '--config', valid_config_path, '--debug']
        
        with patch.object(sys, 'argv', test_args):
            with patch('main.download_model') as mock_download:
                with patch('main.setup_llm') as mock_setup_llm:
                    with patch('builtins.input', side_effect=[':q']):
                        with patch('main.Console') as mock_console_class:
                            mock_console = MagicMock()
                            mock_console_class.return_value = mock_console
                            
                            # Create a mock model file for the test
                            mock_download.return_value = "/tmp/test_model.gguf"
                            mock_setup_llm.return_value = MagicMock()
                            
                            with patch.dict(os.environ, {}, clear=True):
                                result = main.main()
                                
                                # Should succeed
                                assert result == 0
                                
                                # Verify environment was set up
                                assert "LLAMA_LOG_LEVEL" in os.environ


class TestUserInteraction:
    """Test user interaction aspects of the application."""
    
    @patch('main.download_model')
    @patch('main.setup_llm')
    def test_user_prompt_and_response_cycle(self, mock_setup_llm, mock_download, 
                                          valid_config_path, temp_dir):
        """Test the user prompt and response cycle."""
        # Setup mocks
        mock_model_path = os.path.join(temp_dir, "test_model.gguf")
        mock_download.return_value = mock_model_path
        
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Test response from LLM"
        mock_setup_llm.return_value = mock_llm
        
        # Simulate user input: one question, then quit
        user_inputs = ["Hello, how are you?", ":q"]
        
        test_args = ['main.py', '--config', valid_config_path]
        
        with patch.object(sys, 'argv', test_args):
            with patch('builtins.input', side_effect=user_inputs):
                with patch('builtins.print') as mock_print:
                    with patch('main.Console') as mock_console_class:
                        mock_console = MagicMock()
                        mock_console_class.return_value = mock_console
                        
                        result = main.main()
                        
                        # Should complete successfully
                        assert result == 0
                        
                        # Verify LLM was invoked
                        mock_llm.invoke.assert_called_once()
                        
                        # Verify response was printed
                        response_printed = any("Test response from LLM" in str(call) 
                                             for call in mock_print.call_args_list)
                        assert response_printed

    @patch('main.download_model')
    @patch('main.setup_llm')
    def test_quit_command_variations(self, mock_setup_llm, mock_download, 
                                   valid_config_path, temp_dir):
        """Test different ways to quit the application."""
        quit_commands = [":q", ":Q"]
        
        for quit_cmd in quit_commands:
            # Setup mocks
            mock_model_path = os.path.join(temp_dir, "test_model.gguf")
            mock_download.return_value = mock_model_path
            
            mock_llm = MagicMock()
            mock_setup_llm.return_value = mock_llm
            
            test_args = ['main.py', '--config', valid_config_path]
            
            with patch.object(sys, 'argv', test_args):
                with patch('builtins.input', return_value=quit_cmd):
                    with patch('builtins.print') as mock_print:
                        with patch('main.Console') as mock_console_class:
                            mock_console = MagicMock()
                            mock_console_class.return_value = mock_console
                            
                            result = main.main()
                            
                            # Should exit successfully
                            assert result == 0
                            
                            # Should print goodbye message
                            goodbye_printed = any("Exiting Laptop-GPT" in str(call) 
                                                for call in mock_print.call_args_list)
                            assert goodbye_printed
