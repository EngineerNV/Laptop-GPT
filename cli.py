import argparse
import json
from typing import Dict, Any


def get_config_help() -> str:
    """Return help text explaining the configuration file structure."""
    return """
Configuration File Help (config.json):
=====================================

The config.json file allows you to customize various aspects of Laptop-GPT:

Structure:
{
  "model": {
    "repo": "HuggingFace repository ID",
    "file": "Model file name to download",
    "cache_dir": "Local directory to store the model"
  },
  "llm_params": {
    "n_ctx": "Context window size (tokens)",
    "n_batch": "Batch size for processing",
    "n_threads": "Number of threads to use",
    "temperature": "Sampling temperature (0.0-1.0)",
    "max_tokens": "Maximum tokens to generate",
    "use_mlock": "Lock model in memory (true/false)",
    "use_mmap": "Use memory mapping (true/false)",
    "n_gpu_layers": "Number of layers to offload to GPU"
  },
  "environment": {
    "llama_cpp_verbose": "Verbosity level for llama.cpp",
    "ggml_log_level": "GGML logging level"
  },
  "app": {
    "name": "Application name",
    "version": "Application version"
  }
}

Example values:
- n_ctx: 2048 (typical context window)
- temperature: 0.6 (balanced creativity)
- n_threads: 6 (adjust based on your CPU)
- n_gpu_layers: 0 (use CPU only, increase if you have compatible GPU)

The configuration file is optional. If not found, default values will be used.
You can modify config.json to customize the behavior without changing the code.
"""


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Laptop-GPT: AI Assistant for your laptop",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose output"
    )
    
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to configuration file (default: config.json)"
    )
    
    parser.add_argument(
        "--help-config",
        action="store_true",
        help="Show configuration file help and exit"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Laptop-GPT 0.1.2"
    )
    
    args = parser.parse_args()
    
    # Handle config help
    if args.help_config:
        print(get_config_help())
        exit(0)
    
    return args
