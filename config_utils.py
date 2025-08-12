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
        Dictionary containing configuration settings with indexed models
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        json.JSONDecodeError: If the config file contains invalid JSON
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create indexed models dictionary for fast lookups
        models_list = config.get("available_models", [])
        config["_models_index"] = {model.get("id"): model for model in models_list if model.get("id")}
        
        # Backward compatibility: if config has old 'llm_params' key but no 'custom_llm_params', 
        # copy it to maintain compatibility with existing tests and code
        if "llm_params" in config and "custom_llm_params" not in config:
            config["custom_llm_params"] = config["llm_params"]
        # Also ensure llm_params exists for backward compatibility
        if "custom_llm_params" in config and "llm_params" not in config:
            config["llm_params"] = config["custom_llm_params"]
        
        return config
    except FileNotFoundError:
        logging.error(f"Config file {config_path} not found. Please create a config.json file.")
        logging.error("Run 'python main.py --help-config' to see the configuration format.")
        raise FileNotFoundError(f"Required configuration file '{config_path}' not found")
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in config file {config_path}: {e}")
        raise


def get_llm_params(config: Dict[str, Any], model_id: str = None) -> Dict[str, Any]:
    """
    Get LLM parameters based on use_recommended setting and model configuration.
    
    Args:
        config: Configuration dictionary with indexed models
        model_id: Optional model ID to get parameters for. If None, uses current selection.
        
    Returns:
        Dictionary containing LLM parameters to use
    """
    chat_selection = config.get("chat_selection", {})
    use_recommended = chat_selection.get("use_recommended", True)
    
    # Determine which model to get parameters for
    target_model_id = model_id or chat_selection.get("model_id")
    
    if use_recommended and target_model_id:
        # Fast dictionary lookup using the indexed models
        models_index = config.get("_models_index", {})
        model = models_index.get(target_model_id)
        
        if model:
            recommended_params = model.get("recommended_llm_params")
            if recommended_params:
                logging.info(f"Using recommended parameters for model {target_model_id}")
                return recommended_params.copy()
            else:
                logging.warning(f"No recommended parameters found for model {target_model_id}, falling back to custom")
        else:
            logging.warning(f"Model {target_model_id} not found in index, falling back to custom parameters")
    
    # Fall back to custom parameters
    custom_params = config.get("custom_llm_params", {})
    if use_recommended:
        logging.info("Using custom parameters as fallback")
    else:
        logging.info("Using custom parameters (recommended disabled)")
    
    return custom_params.copy()


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
