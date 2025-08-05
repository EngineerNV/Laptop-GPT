import logging
import sys
import argparse
import os
from pathlib import Path
from typing import Optional

# AI/ML Libraries
from huggingface_hub import hf_hub_download, repo_info
from langchain_community.llms import LlamaCpp

# System Utilities
import psutil
from rich.console import Console

def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

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
        "--version",
        action="version",
        version="Laptop-GPT 0.1.0"
    )
    
    return parser.parse_args()

# Model Management
def download_model(
    model_repo: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    model_file: str = "model.gguf",
    cache_dir: Optional[str] = None
) -> str:
    """
    Download a model from HuggingFace Hub.
    
    Args:
        model_repo: HuggingFace repository ID
        model_file: Specific model file to download 
        cache_dir: Local directory to cache the model (optional)
    
    Returns:
        str: Path to the downloaded model file
    """
    console = Console()
    
    if cache_dir is None:
        cache_dir = os.path.join(Path.home(), ".cache", "laptop-gpt")
    
    console.print(f"[yellow]Downloading model: {model_repo}")
    console.print(f"[yellow]File: {model_file}")
    console.print(f"[yellow]Cache directory: {cache_dir}")
    
    try:
        with console.status("[bold green]🐒 Downloading model... (this may take a few minutes)", spinner="monkey"):
            model_path = hf_hub_download(
                repo_id=model_repo,
                filename=model_file,
                cache_dir=cache_dir
            )
        
        console.print(f"[green]✓ Model downloaded successfully!")
        console.print(f"[green]Location: {model_path}")
        return model_path
        
    except Exception as e:
        console.print(f"[red]✗ Download failed: {e}")
        raise

def test_download_animation() -> None:
    """Test the download animation without actually downloading a model."""
    import time
    
    console = Console()
    
    console.print("[yellow]Testing download animation...")
    console.print("[yellow]This will simulate a 5-second download")
    
    try:
        with console.status("[bold green] Downloading model... (this may take a few minutes)", spinner="monkey"):
            # Simulate download time
            time.sleep(5)
        
        console.print("[green]✓ Test completed! Animation looks good!")
        
    except Exception as e:
        console.print(f"[red]✗ Test failed: {e}")
        raise

def main() -> int:
    """Main application entry point."""
    try:
        args = parse_arguments()
        setup_logging(args.log_level)
        
        logger = logging.getLogger(__name__)
        logger.info("Starting Laptop-GPT...")
        
        # TODO: Add main application logic here
        print("Welcome to Laptop-GPT!")
        print("This is the basic template. Add your functionality here.")
        test_download_animation()
        return 0
        
    except KeyboardInterrupt:
        print("\nExiting...")
        return 1
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return 1

if __name__ == "__main__":
    main()
