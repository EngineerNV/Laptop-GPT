import json
import logging
import os
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration settings
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        json.JSONDecodeError: If the config file contains invalid JSON
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        logging.error(f"Config file {config_path} not found. Please create a config.json file.")
        logging.error("Run 'python main.py --help-config' to see the configuration format.")
        raise FileNotFoundError(f"Required configuration file '{config_path}' not found")
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in config file {config_path}: {e}")
        raise


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True
    )


def setup_environment(config: Dict[str, Any], debug_mode: bool, log_level: str) -> None:
    """Configure environment variables and logging based on configuration and debug mode."""
    import warnings
    
    try:
        from langchain_core._api.deprecation import LangChainDeprecationWarning
    except ImportError:
        LangChainDeprecationWarning = None
    
    if debug_mode:
        # Enable full logging and warnings
        setup_logging("DEBUG")
        os.environ["LLAMA_LOG_LEVEL"] = "debug"
    else:
        # Use the specified log level or suppress warnings for non-debug modes
        setup_logging(log_level)
        if log_level in ["ERROR", "WARNING"]:
            warnings.filterwarnings("ignore")
        # Suppress LangChain deprecation warnings unless in debug mode
        if LangChainDeprecationWarning:
            warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
        os.environ["LLAMA_LOG_LEVEL"] = "error" if log_level == "ERROR" else "info"
    
    # Set environment variables from config
    env_config = config.get("environment", {})
    for key, value in env_config.items():
        os.environ[key.upper()] = str(value)
