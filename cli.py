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
  "chat_selection": {
    "profile": "Profile name for prompt formatting",
    "model_id": "Model ID from available_models list",
    "use_recommended": "Use recommended settings for the model (true/false)"
  },
  "available_models": [
    {
      "id": "model-identifier",
      "name": "Human readable model name",
      "model": {
        "repo": "HuggingFace repository ID",
        "file": "Model file name to download",
        "cache_dir": "Local directory to store the model"
      },
      "recommended_profile": "Profile to use with this model",
      "recommended_llm_params": {
        "n_ctx": "Context window size optimized for this model",
        "n_batch": "Batch size optimized for this model",
        "n_threads": "Number of threads optimized for this model",
        "temperature": "Default temperature for this model",
        "max_tokens": "Default max tokens for this model",
        "use_mlock": "Memory locking (true/false)",
        "use_mmap": "Memory mapping (true/false)",
        "n_gpu_layers": "GPU layers for this model"
      }
    }
  ],
  "custom_llm_params": {
    "n_ctx": "Context window size (tokens) - used when use_recommended=false",
    "n_batch": "Batch size for processing",
    "n_threads": "Number of threads to use",
    "temperature": "Sampling temperature (0.0-1.0)",
    "max_tokens": "Maximum tokens to generate",
    "use_mlock": "Lock model in memory (true/false)",
    "use_mmap": "Use memory mapping (true/false)",
    "n_gpu_layers": "Number of layers to offload to GPU"
  },
  "profiles": {
    "profile_name": {
      "format": "Prompt format type",
      "stop_tokens": ["List", "of", "stop", "tokens"],
      "description": "Description of when to use this profile"
    }
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

Parameter Guidelines for 8GB MacBook Air:
- Small models (1-2B): n_ctx=1536, n_batch=10-16, n_gpu_layers=0-2
- Large models (7-8B): n_ctx=640-768, n_batch=2-3, n_gpu_layers=0
- Temperature: 0.3 (focused) to 0.7 (creative)
- use_recommended=true: Uses model-specific optimized settings
- use_recommended=false: Uses custom_llm_params for all models

The configuration file is optional. If not found, default values will be used.
You can modify config.json to customize the behavior without changing the code.
"""


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Laptop-GPT: AI Assistant with Model Catalog & Profile System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run with current config
  python main.py --list-models      # Browse available models
  python main.py --list-profiles    # See prompt profiles  
  python main.py --cache-models     # Download all models
  python main.py --debug            # Run with debug info
        """
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
        "--cache-models",
        action="store_true",
        help="Download and cache all models in the catalog, then exit"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available prompt profiles and exit"
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
    
    # Handle list models
    if args.list_models:
        from config_utils import load_config
        try:
            config = load_config(args.config)
            models = config.get("available_models", [])
            print("\nAvailable Models:")
            print("=" * 80)
            for i, model in enumerate(models, 1):
                model_id = model.get("id", "unknown-id")
                name = model.get("name", "Unknown Model")
                quant = model.get("quant", "Unknown")
                context = model.get("context_len_hint", "Unknown")
                notes = model.get("notes", "No description")
                profile = model.get("recommended_profile", "generic")
                format_type = model.get("format", "unknown")
                
                print(f"{i:2}. {name}")
                print(f"    ID: {model_id}")
                print(f"    Format: {format_type.upper()} | Quantization: {quant} | Context: {context}")
                print(f"    Recommended Profile: {profile}")
                print(f"    Notes: {notes}")
                print(f"    To use: Set 'model_id' to '{model_id}' in chat_selection")
                print()
        except FileNotFoundError:
            print("Config file not found. Please create config.json first.")
        except Exception as e:
            print(f"Error reading config: {e}")
        exit(0)
    
    # Handle cache models
    if args.cache_models:
        from config_utils import load_config
        try:
            config = load_config(args.config)
            models = config.get("available_models", [])
            print(f"\n🗄️  Caching {len(models)} models...")
            print("=" * 60)
            
            from llm_utils import download_model
            from rich.console import Console
            console = Console()
            
            success_count = 0
            for i, model in enumerate(models, 1):
                model_id = model.get("id", "unknown")
                name = model.get("name", "Unknown Model")
                model_config = model.get("model", {})
                
                print(f"\n{i}/{len(models)}. {name} (ID: {model_id})")
                print("-" * 40)
                
                try:
                    with console.status(f"[bold green]Downloading {model_id}...", spinner="dots"):
                        model_path = download_model(
                            model_config.get("repo", ""),
                            model_config.get("file", ""),
                            model_config.get("cache_dir", "")
                        )
                    print(f"✅ Downloaded: {model_path}")
                    success_count += 1
                except Exception as e:
                    print(f"❌ Failed: {e}")
            
            print(f"\n📊 Cache Summary: {success_count}/{len(models)} models downloaded successfully")
            if success_count == len(models):
                print("🎉 All models cached successfully!")
            else:
                print(f"⚠️  {len(models) - success_count} models failed to download")
                
        except FileNotFoundError:
            print("Config file not found. Please create config.json first.")
        except Exception as e:
            print(f"Error caching models: {e}")
        exit(0)
    
    # Handle list profiles
    if args.list_profiles:
        from config_utils import load_config
        try:
            config = load_config(args.config)
            profiles = config.get("profiles", {})
            print("\nAvailable Prompt Profiles:")
            print("=" * 50)
            for name, profile in profiles.items():
                desc = profile.get("description", "No description")
                format_type = profile.get("format", "unknown")
                print(f"• {name:20} - {desc}")
                print(f"  Format: {format_type}")
                print()
        except FileNotFoundError:
            print("Config file not found. Please create config.json first.")
        except Exception as e:
            print(f"Error reading config: {e}")
        exit(0)
    
    return args
