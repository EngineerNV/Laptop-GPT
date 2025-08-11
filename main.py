import logging
import sys
import argparse
import os
from pathlib import Path
from typing import Optional
import warnings

# AI/ML Libraries
from huggingface_hub import hf_hub_download, repo_info
from langchain_community.llms import LlamaCpp

# System Utilities
import psutil
from rich.console import Console

try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning
except ImportError:
    LangChainDeprecationWarning = None

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
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose output"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Laptop-GPT 0.1.2"
    )
    
    return parser.parse_args()

def setup_llm(model_path: str, debug_mode: bool = False) -> LlamaCpp:
    """Set up the LLM (Large Language Model) for inference, suppressing output unless debug."""
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Set environment variables to control llama.cpp verbosity
    if not debug_mode:
        os.environ["LLAMA_CPP_VERBOSE"] = "0"
        os.environ["GGML_LOG_LEVEL"] = "2"  # Error level only
    
    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=2048,
        n_batch=8,
        n_threads=6,
        temperature=0.6,
        max_tokens=1024,
        verbose=debug_mode,
        use_mlock=False,  # Disable memory locking which can cause issues
        use_mmap=True,    # Enable memory mapping for better performance
        n_gpu_layers=0    # Use CPU only to avoid GPU issues
    )
    return llm

def download_model(
    model_repo: str = "bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
    model_file: str = "DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf",
    cache_dir: Optional[str] = ".models/deepseek"
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
        model_path = hf_hub_download(
            repo_id=model_repo,
            filename=model_file,
            cache_dir=cache_dir
        )
        
        # Verify the downloaded file exists and is readable
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Downloaded model file not found: {model_path}")
        
        file_size = os.path.getsize(model_path)
        if file_size == 0:
            raise ValueError(f"Downloaded model file is empty: {model_path}")
        
        console.print(f"[green]Model file size: {file_size / (1024*1024):.1f} MB")
        
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

def build_deepseek_prompt(user_message: str) -> str:
    return f"<｜User｜>{user_message}<｜Assistant｜>"

def setup_environment_and_logging(debug_mode: bool, log_level: str):
    """Configure logging, warnings, and environment based on debug mode."""
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

def main() -> int:
    """Main application entry point."""
    try:
        args = parse_arguments()

        debug_mode = args.debug
        setup_environment_and_logging(debug_mode, args.log_level)
        
        logger = logging.getLogger(__name__)
        logger.info("Starting Laptop-GPT...")

        console = Console()
        if not console:
            raise Exception("rich.console package Not Found - check imports")
        
        console.print("[bold blue]Welcome to Laptop-GPT![/]")
        console.print("[green]Where the power of LLMs run from right on your computer[/]")
        console.print("[italic red]No matter how expensive they may be ;)[/]")

        with console.status("[bold green] Downloading model... (this may take a few minutes)", spinner="earth"):
            model_path = download_model()
        console.print(f"[green]✓ Model downloaded successfully!")
        console.print(f"[green]Location: {model_path}")
        
        try:
            with console.status("[bold blue] Setting up LLM...", spinner="clock"):
                llm = setup_llm(model_path, debug_mode=debug_mode)
            console.print(f"[green]✓ LLM ready!")
        except Exception as llm_error:
            console.print(f"[red]✗ Failed to load LLM: {llm_error}")
            console.print("[yellow]This could be due to:")
            console.print("[yellow]  - Corrupted model file")
            console.print("[yellow]  - Insufficient memory")
            console.print("[yellow]  - Incompatible model format")
            console.print("[yellow]Try downloading the model again or use a smaller model.")
            return 1

        while True:
            user_input = input("Prompt:")
            if user_input.lower() in [":q"]:
                print("Exiting Laptop-GPT. Goodbye!")
                return 0        
            prompt = build_deepseek_prompt(user_input)
            # Use invoke instead of __call__ to avoid deprecation warning
            response = llm.invoke(prompt)
            print(response)

    except KeyboardInterrupt:
        print("\nExiting...")
        return 1
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return 1

if __name__ == "__main__":
    main()
