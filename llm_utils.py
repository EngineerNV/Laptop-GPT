import logging
import os
from pathlib import Path
from typing import Optional
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from rich.console import Console


def setup_llm(model_path: str, llm_params: dict, debug_mode: bool = False) -> Llama:
    """Set up the LLM (Large Language Model) for inference, suppressing output unless debug."""
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Set environment variables to control llama.cpp verbosity
    if not debug_mode:
        os.environ["LLAMA_CPP_VERBOSE"] = "0"
        os.environ["GGML_LOG_LEVEL"] = "2"  # Error level only
    
    llm = Llama(
        model_path=model_path,
        n_ctx=llm_params.get("n_ctx", 2048),
        n_batch=llm_params.get("n_batch", 8),
        n_threads=llm_params.get("n_threads", 6),
        verbose=debug_mode,
        use_mlock=llm_params.get("use_mlock", False),
        use_mmap=llm_params.get("use_mmap", True),
        n_gpu_layers=llm_params.get("n_gpu_layers", 0)
    )
    return llm


def format_prompt(user_input: str, model_type: str = "auto") -> str:
    """Format prompt for different model types."""
    if model_type == "deepseek":
        # Simple format for DeepSeek R1 models - these are reasoning models
        return f"Question: {user_input}\nAnswer:"
    elif model_type == "deepseek_chat":
        # Alternative DeepSeek format with tokens
        return f"<|User|>{user_input}<|Assistant|>"
    elif model_type == "llama" or model_type == "mistral":
        return f"[INST] {user_input} [/INST]"
    elif model_type == "llama3":
        # Llama 3.1+ format with new header system (begin_of_text is handled by chat template)
        return f"<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif model_type == "zephyr":
        # Zephyr format for TinyLlama and similar models
        return f"<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{user_input}</s>\n<|assistant|>\n"
    elif model_type == "chatml":
        return f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    else:
        # Generic format that works with most models
        return f"### Human: {user_input}\n### Assistant:"


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
