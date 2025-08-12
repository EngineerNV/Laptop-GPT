import logging
import sys
import os
from typing import Dict, Any

# Local imports
from cli import parse_arguments
from config_utils import load_config, setup_environment
from llm_utils import setup_llm, download_model, format_prompt

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

        # Get model configuration - check if chat_selection specifies a model from catalog
        try:
            # Get chat selection configuration
            chat_config = config.get("chat_selection", {})
            selected_model_id = chat_config.get("model_id")
            
            # If a model is specified in chat_selection, load it from catalog
            if selected_model_id:
                available_models = config.get("available_models", [])
                model_found = False
                
                for model_entry in available_models:
                    if model_entry["id"] == selected_model_id:
                        # Update model config from catalog
                        config["model"] = model_entry["model"].copy()
                        console.print(f"[blue]Using model from catalog: {model_entry['name']} (id: {selected_model_id})")
                        model_found = True
                        break
                
                if not model_found:
                    console.print(f"[yellow]Warning: Model ID '{selected_model_id}' not found in catalog, using default model")
            
            # Get model configuration - all fields are required
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
                from config_utils import get_llm_params
                llm_params = get_llm_params(config)
                if not llm_params:
                    console.print(f"[red]✗ No LLM parameters available")
                    console.print("[yellow]Please ensure your config.json has parameter settings")
                    console.print("[yellow]Hint: Use --help-config to see the configuration format")
                    return 1
                llm = setup_llm(model_path, llm_params, debug_mode=debug_mode)
            console.print(f"[green]✓ LLM ready!")
            
            # Load profile settings for prompt formatting
            profile_name = chat_config.get("profile", "generic")
            profiles = config.get("profiles", {})
            if profile_name not in profiles:
                console.print(f"[yellow]Warning: Profile '{profile_name}' not found, using generic profile")
                profile_name = "generic"
            
            profile = profiles.get(profile_name, {})
            
            # Get prompt formatting settings from profile
            prompt_format = profile.get("format", "auto")
            stop_tokens = profile.get("stop_tokens", ["User:", "Human:", "###", "\n\n"])
            show_thinking = profile.get("show_thinking", True)
            
            if debug_mode:
                console.print(f"[dim]Using profile: {profile_name} - {profile.get('description', 'No description')}")
                console.print(f"[dim]Format: {prompt_format}, Stop tokens: {stop_tokens}")
                console.print(f"[dim]Show thinking: {show_thinking}")
        except Exception as llm_error:
            console.print(f"[red]✗ Failed to load LLM: {llm_error}")
            console.print("[yellow]This could be due to:")
            console.print("[yellow]  - Corrupted model file")
            console.print("[yellow]  - Insufficient memory")
            console.print("[yellow]  - Incompatible model format")
            console.print("[yellow]Try downloading the model again or use a smaller model.")
            return 1

        while True:
            try:
                user_input = console.input("[bold cyan]You:[/] ")
            except EOFError:
                # Handle EOF gracefully (e.g., when input is piped)
                print("\nExiting Laptop-GPT. Goodbye!")
                return 0
                
            if user_input.lower() in [":q"]:
                print("Exiting Laptop-GPT. Goodbye!")
                return 0        
            
            # Format prompt and generate response using simple text generation
            prompt = format_prompt(user_input, prompt_format)
            if debug_mode:
                console.print(f"[dim]Debug - Formatted prompt: {repr(prompt)}")
            
            # Generate response with loading spinner
            with console.status("🤖 AI is thinking...", spinner="dots"):
                response = llm(
                    prompt,
                    max_tokens=llm_params.get("max_tokens", 1024),
                    temperature=llm_params.get("temperature", 0.7),
                    top_p=0.9,
                    stop=stop_tokens,
                    echo=False
                )
            
            # Get the response text
            response_text = response["choices"][0]["text"].strip()
            
            # Post-process thinking content based on profile settings
            if not show_thinking and ("</think>" in response_text or "<think>" in response_text):
                # Strip thinking process - keep only text after </think>
                if "</think>" in response_text:
                    response_text = response_text.split("</think>")[-1].strip()
                # If response starts with <think> and no </think>, it's all thinking
                elif response_text.startswith("<think>"):
                    response_text = "[Thinking... no final answer provided]"
            
            console.print(f"[green bold]AI: {response_text}")

    except KeyboardInterrupt:
        print("\nExiting...")
        return 1
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return 1

if __name__ == "__main__":
    main()
