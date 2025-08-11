"""
Tests for llm_utils.py module.

This module tests:
- LLM setup with various parameters
- Model downloading functionality
- Prompt building for DeepSeek models
- Error handling for missing models and invalid parameters
"""
import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

# Import the modules to test
import sys
sys.path.append('..')
from llm_utils import setup_llm, download_model, build_deepseek_prompt


class TestSetupLLM:
    """Test LLM setup functionality."""
    
    def test_setup_llm_with_valid_model(self, mock_model_file, sample_llm_params):
        """Test setting up LLM with a valid model file."""
        with patch('llm_utils.LlamaCpp') as mock_llama:
            mock_instance = MagicMock()
            mock_llama.return_value = mock_instance
            
            llm = setup_llm(mock_model_file, sample_llm_params, debug_mode=False)
            
            # Verify LlamaCpp was called with correct parameters
            mock_llama.assert_called_once()
            call_args = mock_llama.call_args
            
            assert call_args[1]['model_path'] == mock_model_file
            assert call_args[1]['n_ctx'] == 1024
            assert call_args[1]['temperature'] == 0.5
            assert call_args[1]['verbose'] == False
            assert llm == mock_instance

    def test_setup_llm_with_missing_model(self, sample_llm_params):
        """Test that FileNotFoundError is raised for missing model file."""
        nonexistent_path = "/path/that/does/not/exist/model.gguf"
        
        with pytest.raises(FileNotFoundError) as exc_info:
            setup_llm(nonexistent_path, sample_llm_params)
        
        assert "Model file not found" in str(exc_info.value)
        assert nonexistent_path in str(exc_info.value)

    def test_setup_llm_debug_mode(self, mock_model_file, sample_llm_params):
        """Test LLM setup in debug mode."""
        with patch('llm_utils.LlamaCpp') as mock_llama:
            with patch.dict(os.environ, {}, clear=True):
                setup_llm(mock_model_file, sample_llm_params, debug_mode=True)
                
                # Verify debug mode doesn't set restrictive env vars
                assert os.environ.get("LLAMA_CPP_VERBOSE") != "0"
                assert os.environ.get("GGML_LOG_LEVEL") != "2"
                
                # Verify verbose=True was passed
                call_args = mock_llama.call_args
                assert call_args[1]['verbose'] == True

    def test_setup_llm_production_mode(self, mock_model_file, sample_llm_params):
        """Test LLM setup in production mode (non-debug)."""
        with patch('llm_utils.LlamaCpp') as mock_llama:
            with patch.dict(os.environ, {}, clear=True):
                setup_llm(mock_model_file, sample_llm_params, debug_mode=False)
                
                # Verify production mode sets restrictive env vars
                assert os.environ.get("LLAMA_CPP_VERBOSE") == "0"
                assert os.environ.get("GGML_LOG_LEVEL") == "2"
                
                # Verify verbose=False was passed
                call_args = mock_llama.call_args
                assert call_args[1]['verbose'] == False

    def test_setup_llm_with_custom_params(self, mock_model_file):
        """Test LLM setup with custom parameters."""
        custom_params = {
            "n_ctx": 4096,
            "n_batch": 16,
            "n_threads": 8,
            "temperature": 0.8,
            "max_tokens": 2048,
            "use_mlock": True,
            "use_mmap": False,
            "n_gpu_layers": 5
        }
        
        with patch('llm_utils.LlamaCpp') as mock_llama:
            setup_llm(mock_model_file, custom_params)
            
            call_args = mock_llama.call_args[1]
            assert call_args['n_ctx'] == 4096
            assert call_args['n_batch'] == 16
            assert call_args['n_threads'] == 8
            assert call_args['temperature'] == 0.8
            assert call_args['max_tokens'] == 2048
            assert call_args['use_mlock'] == True
            assert call_args['use_mmap'] == False
            assert call_args['n_gpu_layers'] == 5


class TestDownloadModel:
    """Test model downloading functionality."""
    
    @patch('llm_utils.hf_hub_download')
    @patch('llm_utils.Console')
    def test_download_model_success(self, mock_console, mock_hf_download, temp_dir):
        """Test successful model download."""
        # Setup mocks
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance
        
        test_model_path = os.path.join(temp_dir, "downloaded_model.gguf")
        mock_hf_download.return_value = test_model_path
        
        # Create a fake downloaded file
        with open(test_model_path, 'wb') as f:
            f.write(b"fake model data" * 1000)
        
        # Test the download
        result = download_model("test/repo", "model.gguf", temp_dir)
        
        # Verify the result
        assert result == test_model_path
        mock_hf_download.assert_called_once_with(
            repo_id="test/repo",
            filename="model.gguf",
            cache_dir=temp_dir
        )
        
        # Verify console output
        assert mock_console_instance.print.call_count >= 3  # At least 3 print calls

    @patch('llm_utils.hf_hub_download')
    @patch('llm_utils.Console')
    def test_download_model_file_not_found_after_download(self, mock_console, mock_hf_download):
        """Test handling when downloaded file doesn't exist."""
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance
        
        # Return a path that doesn't exist
        mock_hf_download.return_value = "/nonexistent/path/model.gguf"
        
        with pytest.raises(FileNotFoundError) as exc_info:
            download_model("test/repo", "model.gguf")
        
        assert "Downloaded model file not found" in str(exc_info.value)

    @patch('llm_utils.hf_hub_download')
    @patch('llm_utils.Console')
    def test_download_model_empty_file(self, mock_console, mock_hf_download, temp_dir):
        """Test handling of empty downloaded file."""
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance
        
        empty_file_path = os.path.join(temp_dir, "empty_model.gguf")
        mock_hf_download.return_value = empty_file_path
        
        # Create an empty file
        with open(empty_file_path, 'w') as f:
            pass  # Empty file
        
        with pytest.raises(ValueError) as exc_info:
            download_model("test/repo", "model.gguf", temp_dir)
        
        assert "Downloaded model file is empty" in str(exc_info.value)

    @patch('llm_utils.hf_hub_download')
    @patch('llm_utils.Console')
    def test_download_model_with_default_cache_dir(self, mock_console, mock_hf_download, temp_dir):
        """Test download with default cache directory."""
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance
        
        test_model_path = os.path.join(temp_dir, "model.gguf")
        mock_hf_download.return_value = test_model_path
        
        # Create the file
        with open(test_model_path, 'wb') as f:
            f.write(b"test data")
        
        # Call without cache_dir (should use default)
        download_model("test/repo", "model.gguf", cache_dir=None)
        
        # Verify default cache dir was used
        call_args = mock_hf_download.call_args[1]
        expected_cache = str(Path.home() / ".cache" / "laptop-gpt")
        assert call_args['cache_dir'] == expected_cache

    @patch('llm_utils.hf_hub_download')
    @patch('llm_utils.Console')
    def test_download_model_hub_error(self, mock_console, mock_hf_download):
        """Test handling of HuggingFace Hub download errors."""
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance
        
        # Simulate download failure
        mock_hf_download.side_effect = Exception("Network error")
        
        with pytest.raises(Exception) as exc_info:
            download_model("test/repo", "model.gguf")
        
        assert "Network error" in str(exc_info.value)
        # Verify error was printed to console
        mock_console_instance.print.assert_called()


class TestBuildDeepseekPrompt:
    """Test DeepSeek prompt building functionality."""
    
    def test_build_simple_prompt(self):
        """Test building a simple prompt."""
        user_message = "Hello, how are you?"
        result = build_deepseek_prompt(user_message)
        
        expected = "<｜User｜>Hello, how are you?<｜Assistant｜>"
        assert result == expected

    def test_build_prompt_with_special_characters(self):
        """Test building prompt with special characters."""
        user_message = "What's 2+2? Please explain the math!"
        result = build_deepseek_prompt(user_message)
        
        expected = "<｜User｜>What's 2+2? Please explain the math!<｜Assistant｜>"
        assert result == expected

    def test_build_prompt_with_empty_message(self):
        """Test building prompt with empty message."""
        user_message = ""
        result = build_deepseek_prompt(user_message)
        
        expected = "<｜User｜><｜Assistant｜>"
        assert result == expected

    def test_build_prompt_with_multiline_message(self):
        """Test building prompt with multiline message."""
        user_message = "First line\nSecond line\nThird line"
        result = build_deepseek_prompt(user_message)
        
        expected = "<｜User｜>First line\nSecond line\nThird line<｜Assistant｜>"
        assert result == expected

    def test_build_prompt_with_unicode(self):
        """Test building prompt with unicode characters."""
        user_message = "Hello 世界! How do you say 'hello' in Chinese?"
        result = build_deepseek_prompt(user_message)
        
        expected = "<｜User｜>Hello 世界! How do you say 'hello' in Chinese?<｜Assistant｜>"
        assert result == expected


class TestLLMUtilsIntegration:
    """Integration tests for LLM utilities."""
    
    @patch('llm_utils.LlamaCpp')
    @patch('llm_utils.hf_hub_download') 
    @patch('llm_utils.Console')
    def test_download_and_setup_workflow(self, mock_console, mock_hf_download, mock_llama, temp_dir):
        """Test the complete workflow from download to LLM setup."""
        # Setup mocks
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance
        
        model_path = os.path.join(temp_dir, "workflow_model.gguf")
        mock_hf_download.return_value = model_path
        
        mock_llm_instance = MagicMock()
        mock_llama.return_value = mock_llm_instance
        
        # Create the model file
        with open(model_path, 'wb') as f:
            f.write(b"fake model data" * 2000)
        
        # Test the workflow
        downloaded_path = download_model("test/workflow-repo", "workflow.gguf", temp_dir)
        
        llm_params = {
            "n_ctx": 1024,
            "temperature": 0.6,
            "max_tokens": 512
        }
        
        llm = setup_llm(downloaded_path, llm_params, debug_mode=False)
        
        # Verify the complete workflow
        assert downloaded_path == model_path
        assert llm == mock_llm_instance
        
        # Verify LLM was configured correctly
        mock_llama.assert_called_once()
        call_args = mock_llama.call_args[1]
        assert call_args['model_path'] == model_path
        assert call_args['n_ctx'] == 1024
        assert call_args['temperature'] == 0.6
