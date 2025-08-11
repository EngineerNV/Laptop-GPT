# Laptop-GPT

This is a repository dedicated to using opensource models locally on your machine - whether that be a mac or windows laptop. Given hardware limitations the focus is to be able to get a smaller version running that will be compatiable with most systems. Then evaluate its usefullness - possibly with automation and a web agent.

## 🚀 Quickstart

### 1. Create and Activate a Virtual Environment

It is highly recommended to use a Python virtual environment for this project. This keeps dependencies isolated and your system clean.

#### On macOS/Linux:
```sh
python -m venv .venv
source .venv/bin/activate
```

#### On Windows (CMD):
```bat
python -m venv .venv
.venv\Scripts\activate
```

#### On Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install Requirements
```sh
pip install -r requirements.txt
```

### 3. Create Configuration File
The application requires a `config.json` file. You can use the provided one or create your own:
```sh
# Use the provided config.json or see configuration help
python main.py --help-config
```

### 4. Run Laptop-GPT
```sh
python main.py
```

## 🧪 Testing

Laptop-GPT includes a comprehensive test suite to ensure reliability as you develop new features.

### Quick Test Setup
```sh
# Install test dependencies
python run_tests.py --install-deps

# Run all tests
python run_tests.py

# Run with coverage report
python run_tests.py --coverage
```

### Learning from Tests
The test suite is designed to help you learn testing patterns:
- **Unit tests** for individual functions
- **Integration tests** for complete workflows  
- **Mocking** for external dependencies
- **Configuration testing** for validation

See [TESTING.md](TESTING.md) for detailed testing guide and learning materials.

### 4. Accessing the Virtual Environment

Once activated, your terminal prompt will usually change to show `(.venv)` at the beginning. While the environment is active:

- All Python and pip commands will use the isolated environment.
- To leave the virtual environment, simply run:

```sh
deactivate
```

If you open a new terminal, remember to activate the environment again before running any project commands.

## 📁 Project Structure

```
├── main.py              # Main application entry point
├── cli.py               # Command line argument parsing
├── config_utils.py      # Configuration loading and environment setup
├── llm_utils.py         # LLM setup, model downloading, and prompt building
├── config.json          # Configuration file (customizable)
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## ⚙️ Configuration

Laptop-GPT **requires** a `config.json` file to run. This file contains all the necessary settings for the application.

### Create Configuration File
A `config.json` file is provided with the repository. You can modify it or create your own.

### View Configuration Help
```sh
python main.py --help-config
```

### Configuration Options
- **Model settings**: Repository, file name, cache directory
- **LLM parameters**: Context size, temperature, threads, etc.
- **Environment**: Logging levels and verbosity settings

### Command Line Options
```sh
python main.py --help           # Show all options
python main.py --debug          # Enable debug mode
python main.py --log-level INFO # Set logging level
python main.py --config custom.json  # Use custom config file
```

## 🔧 Customization

### Model Configuration
Edit `config.json` to change:
- Model repository and file
- Context window size
- Temperature and other generation parameters
- Number of CPU threads to use

### Example Custom Config
```json
{
  "model": {
    "repo": "microsoft/DialoGPT-medium",
    "file": "pytorch_model.bin",
    "cache_dir": "./custom_models"
  },
  "llm_params": {
    "temperature": 0.8,
    "max_tokens": 512,
    "n_threads": 8
  }
}
```

## Design Choices

### Model Selection
- **DeepSeek-R1-Distill-Qwen-1.5B**: Chosen for efficiency and 8GB RAM compatibility
- **CPU-only inference**: Prioritizes compatibility over speed for universal laptop support
- **Quantized models**: 4-bit quantization for optimal memory usage

### Architecture
- **LangChain**: Provides flexible model integration and chain building
- **llama-cpp-python**: CPU-optimized inference engine
- **Local-first**: No cloud dependencies, complete offline capability
- **Memory-aware**: Designed for systems with as little as 8GB RAM

### Dependencies
- **Core AI**: LangChain ecosystem for model orchestration
- **Inference**: llama-cpp-python for efficient CPU inference
- **Model Management**: HuggingFace Hub for model downloading
- **System Utilities**: psutil for resource monitoring
- **CLI/UX**: Rich for beautiful terminal interfaces

### Hardware Requirements
- **Minimum RAM**: 8GB (with 3-4GB available for model)
- **Storage**: 2-3GB for model files
- **CPU**: Any modern x64 processor (Apple Silicon supported)
- **GPU**: Not required (CPU-only design)
