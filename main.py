import logging
import sys
import os
from typing import Dict, Any

# Local imports
from cli import parse_arguments
from config_utils import load_config, setup_environment
from llm_utils import setup_llm, download_model, build_deepseek_prompt

# System Utilities
from rich.console import Console

def main() -> int:
    """Main application entry point."""
    try:
        args = parse_arguments()

        # Load configuration - will exit if config file not found
        try:
            config = load_config(args.config)
        except FileNotFoundError as e:
            console = Console()
            console.print(f"[red]✗ {e}")
            console.print("[yellow]Hint: Use --help-config to see the configuration file format")
            return 1
        
        debug_mode = args.debug
        setup_environment(config, debug_mode, args.log_level)
        
        logger = logging.getLogger(__name__)
        logger.info("Starting Laptop-GPT...")

        console = Console()
        if not console:
            raise Exception("rich.console package Not Found - check imports")
        
        app_config = config.get("app", {})
        app_name = app_config.get("name", "Laptop-GPT")  # This one can have a reasonable default
        console.print(f"[bold blue]Welcome to {app_name}![/]")
        console.print("[green]Where the power of LLMs run from right on your computer[/]")
        console.print("[italic red]No matter how expensive they may be ;)[/]")

        # Get model configuration - all fields are required
        try:
            model_config = config["model"]
            model_repo = model_config["repo"]
            model_file = model_config["file"]
            cache_dir = model_config["cache_dir"]
        except KeyError as e:
            console.print(f"[red]✗ Missing required configuration: {e}")
            console.print("[yellow]Please ensure your config.json has all required model settings")
            console.print("[yellow]Hint: Use --help-config to see the configuration format")
            return 1

        with console.status("[bold green] Downloading model... (this may take a few minutes)", spinner="earth"):
            model_path = download_model(model_repo, model_file, cache_dir)
        console.print(f"[green]✓ Model downloaded successfully!")
        console.print(f"[green]Location: {model_path}")
        
        try:
            with console.status("[bold blue] Setting up LLM...", spinner="clock"):
                try:
                    llm_params = config["llm_params"]
                except KeyError:
                    console.print(f"[red]✗ Missing required configuration: 'llm_params'")
                    console.print("[yellow]Please ensure your config.json has LLM parameter settings")
                    console.print("[yellow]Hint: Use --help-config to see the configuration format")
                    return 1
                llm = setup_llm(model_path, llm_params, debug_mode=debug_mode)
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
