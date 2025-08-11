import logging
import os
from pathlib import Path
from typing import Optional
from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp
from rich.console import Console


def setup_llm(model_path: str, llm_params: dict, debug_mode: bool = False) -> LlamaCpp:
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
        n_ctx=llm_params.get("n_ctx", 2048),
        n_batch=llm_params.get("n_batch", 8),
        n_threads=llm_params.get("n_threads", 6),
        temperature=llm_params.get("temperature", 0.6),
        max_tokens=llm_params.get("max_tokens", 1024),
        verbose=debug_mode,
        use_mlock=llm_params.get("use_mlock", False),
        use_mmap=llm_params.get("use_mmap", True),
        n_gpu_layers=llm_params.get("n_gpu_layers", 0)
    )
    return llm


def download_model(
    model_repo: str,
    model_file: str,
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


def build_deepseek_prompt(user_message: str) -> str:
    """Build a prompt specifically for DeepSeek models."""
    return f"<｜User｜>{user_message}<｜Assistant｜>"
