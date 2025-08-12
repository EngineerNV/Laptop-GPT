# Laptop-GPT

A powerful local AI assistant that runs entirely on your laptop. Laptop-GPT provides template-agnostic prompt formatting with profile-based configuration, giving you full control over quantized model interactions without relying on potentially broken chat templates.

## 🎯 Key Features

- **Profile-Based Prompt Control**: Easy switching between model formats (DeepSeek, Llama, ChatML, etc.)
- **Template-Independent**: No reliance on embedded chat templates that may be corrupted or incompatible
- **Quantized Model Optimization**: Designed specifically for GGUF quantized models with flexible prompt formatting
- **Reasoning Control**: Toggle between showing/hiding model thinking processes
- **8GB RAM Optimized**: All included models tested and optimized for 8GB systems (alpha v0.1.x)
- **Offline-First**: Complete local operation, no cloud dependencies

## 🧠 Why Template-Independent Matters

### The Quantized Model Challenge

Quantized models (GGUF format) often face significant challenges:

1. **Broken Chat Templates**: During quantization, chat templates can become corrupted or incompatible with llama-cpp-python
2. **Distillation Issues**: Models distilled from larger models may inherit incorrect templates from the parent model
3. **Training Inconsistencies**: Fine-tuning processes can introduce template mismatches between the model's training and its embedded metadata
4. **Framework Compatibility**: Templates that work in one framework (like Transformers) may not translate properly to llama-cpp

### Our Solution: Profile-Based Control

Instead of relying on embedded templates, Laptop-GPT gives you direct control:

```json
{
  "chat_selection": {
    "model_id": "deepseek-r1-1.5b",  // Model ID from catalog
    "profile": "deepseek",           // Prompt format profile
    "use_recommended": true          // Use model's recommended LLM parameters
  },
  "profiles": {
    "deepseek": {
      "format": "deepseek",
      "stop_tokens": ["Question:", "Answer:", "User:", "Human:", "\n\n"],
      "description": "DeepSeek models - direct Q&A format without reasoning"
    }
  }
}
```

This approach provides:
- **Guaranteed Compatibility**: Format works regardless of embedded templates
- **Precise Control**: Fine-tune stop tokens for optimal responses  
- **Easy Debugging**: See exactly what prompt format is being used
- **Model Flexibility**: Switch between formats without changing models

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
Copy and customize the provided `config.json` file:
```sh
# View available prompt profiles
python main.py --list-profiles

# See configuration help
python main.py --help-config
```

### 4. Browse and Select Models
```sh
# See all available models in the catalog
python main.py --list-models
```

Then edit your `config.json` to use a specific model:
```json
{
  "chat_selection": {
    "model_id": "tinyllama-1.1b",   // Model ID from catalog
    "profile": "llama",             // Prompt format profile
    "use_recommended": true         // Use model's recommended LLM parameters
  }
}
```

**About `use_recommended`**: When set to `true`, Laptop-GPT automatically uses the optimized parameters (`n_ctx`, `n_batch`, `temperature`, etc.) that are pre-configured for each model. This ensures optimal performance and memory usage for your hardware. You can override these by setting `use_recommended: false` and defining your own `custom_llm_params`.

### 5. List Available Profiles
```sh
# List available prompt profiles  
python main.py --list-profiles
```

Select a profile for prompt formatting:

- **`deepseek`** - Direct Q&A format without reasoning  
- **`deepseek_reasoning`** - Hidden reasoning (shows only final answer)
- **`deepseek_show_reasoning`** - Shows full thinking process
- **`llama`** - Llama/Mistral format
- **`chatml`** - ChatML format (GPT-4 style)
- **`generic`** - Universal format that works with most models

Then choose a model ID from the catalog using `--list-models`.

### 6. Run Laptop-GPT
```sh
python main.py

# With debug info to see prompt formatting
python main.py --debug

# Exit the chat with :q
```

### Quick Testing with Echo Commands

For quick testing without interactive mode, you can pipe questions directly:

```sh
# Quick test with a simple question
echo "What is 2+2?" | python main.py

# Test and immediately quit
echo -e "What is the capital of France?\n:q" | python main.py

# Debug mode for troubleshooting
echo "Hello AI" | python main.py --debug
```

This is especially useful for:
- **CI/CD testing**: Automated testing in pipelines
- **Profile validation**: Quick checks when switching models
- **Performance benchmarking**: Timing response generation
- **Troubleshooting**: Debugging prompt formatting issues

## ⚠️ Important Limitations (Alpha Version)

### Memory Requirements & Performance

**MacBook Air 8GB Memory Warning**: 
- Models may experience **degraded performance** with limited RAM
- Larger models (7B+) may cause system slowdown or memory pressure
- Consider using smaller models like TinyLlama (1.1B) or Llama 3.2 1B for optimal performance

**Recommended Memory Guidelines:**
- **8GB RAM**: Stick to 1-2B parameter models (TinyLlama, Llama 3.2 1B)
- **16GB RAM**: Up to 7B models work well (Mistral 7B, Llama 3.1 8B)
- **32GB+ RAM**: All models in catalog supported

**Alpha Version Model Selection**: The pre-configured models in our catalog are specifically chosen to work well on **8GB RAM systems**. While larger models (7B-8B) are included for users with more memory, the focus is on ensuring reliable performance on resource-constrained hardware typical of MacBook Air and similar laptops.

### Alpha Version Behavior (v0.1.x)

**No Conversation History**: This alpha version operates in **single-shot mode**:
- Each question is independent - no memory of previous exchanges
- Models don't maintain conversation context between messages
- Every prompt is treated as a fresh interaction

**Why No History in Alpha:**
- **Memory Optimization**: History storage would increase RAM usage significantly
- **Simplicity**: Single-shot is more predictable and easier to debug
- **Performance**: Avoids context length limitations and memory bloat
- **Testing Focus**: Allows validation of core prompt formatting without history complexity

**Future Versions**: Conversation history and multi-turn dialogue planned for v0.2.x series.

## Model Catalog

### Browsing Available Models

List all pre-configured models:
```sh
python main.py --list-models
```

**Example output:**
```
Available Models:
================================================================================
 1. DeepSeek R1 Distill Qwen 1.5B (GGUF Q5_K_M)
    Format: GGUF | Quantization: Q5_K_M | Context: 4k-8k
    Recommended Profile: deepseek
    Notes: Current model; balanced speed/quality for Qwen-style chat.

 2. TinyLlama 1.1B Chat v1.0
    Format: GGUF | Quantization: Q4_K_M | Context: 4k
    Recommended Profile: llama
    Notes: Ultra-fast, great for lightweight assistants and testing.
```

### Switching Models

Change models by updating your chat selection in `config.json`:
```json
{
  "chat_selection": {
    "model_id": "tinyllama-1.1b",   // Model ID from catalog
    "profile": "llama",             // Prompt format profile
    "use_recommended": true         // Use model's recommended LLM parameters
  }
}
```

**8GB RAM Focus**: All models in the catalog are selected for compatibility with 8GB systems in this alpha version. While larger models (7B-8B) are included, the primary focus is ensuring reliable performance on resource-constrained hardware.

### Model Catalog Features

Each model in the catalog includes:
- **Name & Description**: Clear identification and use case
- **Format**: GGUF, MLX, Transformers, etc.
- **Quantization**: Compression level (Q4_K_M, Q5_K_M, etc.)
- **Context Length**: Supported conversation history size
- **Recommended Profile**: Optimal prompt format for the model
- **Technical Notes**: Performance characteristics and requirements
- **8GB Compatibility**: All models tested for 8GB RAM systems in alpha v0.1.x

### Adding Custom Models

Extend the catalog in `config.json`:
```json
{
  "available_models": [
    {
      "id": "my-custom-model",
      "name": "My Custom Model",
      "model": {
        "repo": "username/model-repo",
        "file": "model.gguf",
        "cache_dir": ".models/custom"
      },
      "format": "gguf",
      "quant": "Q4_K_M",
      "context_len_hint": "4k",
      "notes": "Custom model for specific use case",
      "recommended_profile": "generic"
    }
  ]
}
```

## Profile-Based Configuration

### Available Profiles

List all available profiles:
```sh
python main.py --list-profiles
```

### Profile Structure
```json
{
  "chat_selection": {
    "model_id": "tinyllama-1.1b",    // Model ID from catalog
    "profile": "llama",              // Prompt format profile
    "use_recommended": true          // Use model's recommended LLM parameters
  },
  "profiles": {
    "llama": {
      "format": "llama",                           // Prompt format type
      "stop_tokens": ["[INST]", "[/INST]"],       // When to stop generating
      "description": "Llama format for Llama/Mistral models"
    }
  }
}
```

### Switching Models and Profiles

Change both model and profile in your chat selection:
```json
{
  "chat_selection": {
    "model_id": "mistral-7b",       // Model ID from catalog
    "profile": "llama",             // Prompt format profile
    "use_recommended": true         // Use model's recommended LLM parameters
  }
}
```

### Creating Custom Profiles

Add your own profile for new model types:
```json
{
  "profiles": {
    "my_custom_model": {
      "format": "auto",
      "stop_tokens": ["User:", "Bot:", "\n\n"],
      "description": "Custom format for my specific model"
    }
  }
}
```

## 🧪 Testing

Laptop-GPT includes a comprehensive test suite to ensure reliability as you develop new features. Our testing infrastructure has been thoroughly battle-tested through real debugging scenarios.

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

### Real-World Testing Experience: What We Learned

During development of this project, we encountered and solved several real testing challenges that demonstrate important testing principles:

#### The Hanging Test Problem

**Issue**: Integration tests would hang indefinitely instead of completing.

**Root Cause**: Mock configuration mismatch between test expectations and actual implementation:
- Tests were mocking `builtins.input` 
- Code actually uses `console.input()` from Rich library
- Result: Tests waited forever for input that would never be properly mocked

**Solution Process**:
1. **Identified the problem**: Used `--debug` mode and process monitoring
2. **Traced the mocking**: Examined what functions were actually being called
3. **Fixed systematically**: Updated all mock targets to match actual implementation

**Code Fix Example**:
```python
# ❌ Wrong - this doesn't work with Rich Console
with patch('builtins.input', side_effect=[':q']):
    result = main.main()

# ✅ Correct - mock the actual Console class and its input method
with patch('main.Console') as mock_console_class:
    mock_console = MagicMock()
    mock_console_class.return_value = mock_console
    mock_console.input.side_effect = [':q']
    result = main.main()
```

**Testing Lesson**: Always mock what's actually being called, not what you think should be called.

#### The LLM Response Format Challenge

**Issue**: Tests expected `llm.invoke()` but code calls `llm()` directly.

**Investigation Process**:
```python
# What we expected (wrong):
mock_llm.invoke.assert_called_once()

# What actually happens (correct):
mock_llm.assert_called_once()
```

**Response Format Learning**:
The LLM returns a specific structure that must be mocked correctly:
```python
# ✅ Correct mock response format
mock_llm.return_value = {
    "choices": [{"text": "Test response from LLM"}]
}
```

**Testing Lesson**: Study the actual API before writing mocks - don't assume the interface.

#### Console vs Print Output Verification

**Issue**: Tests couldn't find expected output in assertions.

**Discovery Process**:
```python
# ❌ Wrong - checking built-in print calls
response_printed = any("Test response" in str(call) 
                     for call in mock_print.call_args_list)

# ✅ Correct - checking Rich console.print calls  
response_printed = any("Test response" in str(call) 
                     for call in mock_console.print.call_args_list)
```

**Testing Lesson**: Verify output in the same place the application actually writes it.

### Memory & Performance Testing Formulas

Understanding these formulas helps you write better tests and optimize model performance:

#### KV Cache Memory Formula
```
KV_Cache_Memory = 2 * n_layers * n_ctx * hidden_size * bytes_per_param

Where:
- n_layers: Number of transformer layers
- n_ctx: Context window size (tokens)
- hidden_size: Model's hidden dimension
- bytes_per_param: 4 bytes (FP32) or 2 bytes (FP16)
```

**Example Calculation**:
```python
# TinyLlama 1.1B: 22 layers, 2048 hidden size, 1024 context
tinyllama_kv = 2 * 22 * 1024 * 2048 * 2  # FP16
# = ~185 MB KV cache

# Mistral 7B: 32 layers, 4096 hidden size, 1024 context  
mistral_kv = 2 * 32 * 1024 * 4096 * 2   # FP16
# = ~1.07 GB KV cache
```

**Testing Implication**: This explains why 1B models can handle much larger contexts than 7B models on the same hardware.

#### Total Memory Usage Formula
```
Total_Memory = Model_Weights + KV_Cache + Overhead

Model_Weights (quantized) = original_size * quantization_ratio
- Q4_K_M: ~0.5x original size
- Q5_K_M: ~0.625x original size  
- Q8_0: ~1.0x original size

Overhead = ~200-500MB (OS, Python, llama-cpp)
```

**Testing Strategy**: Use these formulas to predict memory usage and set appropriate test limits.

### Using simple_model_test.py

The `simple_model_test.py` script provides standalone model validation functions that are excluded from the main pytest suite to avoid interference.

#### What is simple_model_test.py?

This file contains utility functions for testing model functionality without the complexity of the full test suite:

```python
# Functions available (renamed from test_* to avoid pytest discovery):
validate_model_loading()     # Test basic model loading
validate_model_inference()   # Test text generation
validate_memory_usage()      # Monitor memory consumption
validate_performance()       # Benchmark inference speed
```

#### Running Individual Validations

```sh
# Run specific validation functions
python -c "from simple_model_test import validate_model_loading; validate_model_loading()"

# Test inference with custom prompt
python -c "
from simple_model_test import validate_model_inference
validate_model_inference('What is artificial intelligence?')
"

# Check memory usage during operation
python -c "from simple_model_test import validate_memory_usage; validate_memory_usage()"
```

#### Integration with Your Workflow

Use `simple_model_test.py` for:

1. **New Model Validation**: Test a new model before adding to catalog
```sh
# Edit simple_model_test.py to point to your model
python -c "from simple_model_test import validate_model_loading; validate_model_loading()"
```

2. **Performance Benchmarking**: Measure inference speed
```sh
python -c "
from simple_model_test import validate_performance
validate_performance(num_trials=5)
"
```

3. **Memory Profiling**: Monitor resource usage
```sh
# Install memory profiler if needed
pip install memory-profiler

# Run with memory monitoring
python -c "
from simple_model_test import validate_memory_usage
validate_memory_usage(verbose=True)
"
```

4. **Quick Debugging**: Test model behavior without full app startup
```sh
# Quick inference test
python -c "
from simple_model_test import validate_model_inference
result = validate_model_inference('Test prompt', debug=True)
print(f'Response: {result}')
"
```

#### Why simple_model_test.py is Separate

- **Pytest Exclusion**: Functions renamed from `test_*` to `validate_*` to avoid automatic discovery
- **Standalone Operation**: Can run without the full test infrastructure  
- **Resource Intensive**: Model loading tests take time and memory
- **Development Focus**: Designed for manual testing during development

#### Custom Validation Functions

Add your own validation functions to `simple_model_test.py`:

```python
def validate_custom_scenario():
    """Test a specific use case or model configuration."""
    # Your custom test logic here
    pass

def validate_profile_compatibility(profile_name):
    """Test a specific prompt profile with the current model."""
    # Test profile-specific behavior
    pass
```

### Testing Best Practices Learned

From our debugging experience, here are key testing principles:

1. **Mock What's Actually Called**: Don't assume - verify the actual function calls
2. **Test at the Right Level**: Integration tests should test real workflows  
3. **Understand the Implementation**: Study the code before writing tests
4. **Use Debug Mode**: Enable verbose output to understand test failures
5. **Test Memory Constraints**: Include resource usage in your test considerations
6. **Separate Concerns**: Keep heavy model tests separate from fast unit tests

### Test Suite Organization

```
tests/
├── test_cli.py              # Command-line interface tests
├── test_config_utils.py     # Configuration loading tests  
├── test_llm_utils.py        # LLM utility function tests
├── test_main_integration.py # Full application workflow tests
├── conftest.py              # Shared test fixtures
└── fixtures/                # Test data files
    ├── valid_config.json
    ├── invalid_config.json
    └── incomplete_config.json

simple_model_test.py         # Standalone model validation (excluded from pytest)
```

This organization separates fast unit tests from resource-intensive model tests, ensuring rapid development iteration while maintaining comprehensive coverage.

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
├── cli.py               # Command line argument parsing with profile support
├── config_utils.py      # Configuration loading and environment setup
├── llm_utils.py         # LLM setup, model downloading, and flexible prompt formatting
├── config.json          # Configuration file with prompt profiles
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🎛️ Prompt Format Control

### Understanding Prompt Formats

Different models expect different conversation formats:

```python
# DeepSeek Q&A format
"Question: What is AI?\nAnswer:"

# DeepSeek Chat format  
"<|User|>What is AI?<|Assistant|>"

# Llama/Mistral format
"[INST] What is AI? [/INST]"

# ChatML format
"<|im_start|>user\nWhat is AI?<|im_end|>\n<|im_start|>assistant\n"

# Generic format
"### Human: What is AI?\n### Assistant:"
```

### Stop Token Strategy

Stop tokens prevent models from continuing conversations indefinitely:

```json
{
  "stop_tokens": ["Question:", "Answer:", "User:", "Human:", "\n\n"]
}
```

**How it works:**
- Model generates: `"Answer: AI is... Question: What about..."`
- Stops at `"Question:"` → Returns: `"Answer: AI is..."`

### Reasoning Model Support

For models that show thinking processes:

```json
{
  "show_thinking": false  // Hide <think>...</think> content
}
```

**Input:**
```
<think>
The user is asking about AI. I should provide a clear, informative answer.
</think>

AI is the simulation of human intelligence in machines.
```

**Output (thinking hidden):**
```
AI is the simulation of human intelligence in machines.
```

## ⚙️ Configuration

### Quick Configuration
```sh
python main.py --list-profiles     # See available prompt profiles
python main.py --help-config       # View configuration help
python main.py --debug            # Enable debug mode
```

### Model Configuration
Edit `config.json` to customize:
```json
{
  "chat_selection": {
    "model_id": "deepseek-r1-1.5b", // Model ID from catalog
    "profile": "deepseek_reasoning", // Prompt format profile
    "use_recommended": true          // Use model's recommended LLM parameters
  },
  "model": {
    "repo": "bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
    "file": "DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf",
    "cache_dir": ".models/deepseek"
  },
  "llm_params": {
    "n_ctx": 2048,        // Context window size
    "temperature": 0.3,   // Response creativity (0.0-1.0)
    "max_tokens": 512,    // Maximum response length
    "n_threads": 6        // CPU threads to use
  }
}
```

### Advanced Configuration
- **Profile switching**: Change prompt formats without code changes
- **Stop token tuning**: Fine-tune response boundaries
- **Reasoning control**: Toggle thinking process visibility
- **Debug mode**: See exactly what prompts are sent to models

### Command Line Options
```sh
python main.py --help              # Show all options
python main.py --debug             # Enable debug mode with prompt visibility
python main.py --log-level INFO    # Set logging level
python main.py --config custom.json # Use custom config file
python main.py --list-profiles     # List available prompt profiles
```

## 🔧 Troubleshooting

### Common Issues with Quantized Models

**Problem**: Model generates weird tokens, loops, or doesn't follow chat format
**Solution**: Switch to a different profile:
```json
{
  "chat_selection": {
    "model_id": "tinyllama-1.1b",   // Current model ID
    "profile": "generic",           // Try universal format
    "use_recommended": true         // Use model's recommended LLM parameters
  }
}
```

**Problem**: Model shows thinking process when you don't want it
**Solution**: Use non-reasoning profile:
```json
{
  "chat_selection": {
    "model_id": "deepseek-r1-1.5b", // Current model ID
    "profile": "deepseek",          // Simple Q&A format
    "use_recommended": true         // Use model's recommended LLM parameters
  }
}
```

**Problem**: Model stops responding too early
**Solution**: Adjust stop tokens:
```json
{
  "profiles": {
    "my_profile": {
      "stop_tokens": ["\n\n"]  // Only stop on double newlines
    }
  }
}
```

### Debug Mode

Enable debug mode to see what's happening:
```sh
python main.py --debug

# Or test quickly with echo
echo "Test question" | python main.py --debug
```

This shows:
- Exact prompt format being used
- Active profile and settings
- Stop tokens in effect
- Model loading details

### Quick Testing & Validation

Use echo commands for rapid testing:
```sh
# Test model response quality
echo "What is artificial intelligence?" | python main.py

# Validate profile switching
echo "2+2=?" | python main.py --debug

# Check memory performance
echo -e "Simple test\n:q" | python main.py
```

**When to use echo testing:**
- After changing profiles or models
- When troubleshooting prompt formatting
- For performance benchmarking
- During development and testing

### Profile Issues

If a profile doesn't exist:
```
Warning: Profile 'nonexistent' not found, using generic profile
```

List available profiles:
```sh
python main.py --list-profiles
```

## Design Philosophy

### Template-Independent Architecture
Traditional approach (problematic):
```
Model File → Embedded Chat Template → llama-cpp → Output
     ↑              ↑                    ↑
   May be        Often broken       Incompatible
  quantized     or incompatible     with format
```

Our approach (reliable):
```
Model File → Laptop-GPT Profile → Custom Format → llama-cpp → Output
     ↑              ↑                    ↑              ↑
   Any GGUF    User-controlled    Works reliably  Predictable
```

### Why This Matters for Quantized Models

1. **Quantization Artifacts**: GGUF conversion can corrupt embedded templates
2. **Distillation Issues**: Smaller models inherit templates from incompatible parent models  
3. **Framework Differences**: Templates designed for Transformers may not work in llama-cpp
4. **Training Mismatches**: Fine-tuning can create discrepancies between model behavior and templates

### Our Solution Benefits

- **Guaranteed Compatibility**: Profiles always work regardless of model metadata
- **User Control**: You decide the exact format, not the model file
- **Easy Debugging**: See exactly what prompt is sent with `--debug`
- **Flexible**: Switch formats without re-downloading models
- **Extensible**: Add new profiles for new model types easily

### Model Selection Rationale
- **DeepSeek-R1-Distill-Qwen-1.5B**: Excellent reasoning capabilities in small package
- **GGUF Quantization**: Q5_K_M provides optimal quality/size balance
- **CPU-First Design**: Universal compatibility across hardware
- **Memory Efficiency**: Runs comfortably on 8GB systems

### Architecture Principles
- **Profile-Based Control**: Format templates separated from model files
- **Template Independence**: No reliance on potentially broken embedded templates
- **Debug Transparency**: Always know what prompt format is being used
- **Reasoning Flexibility**: Toggle thinking process visibility
- **Local-First**: Complete offline operation
- **Memory-Aware**: Optimized for resource-constrained environments

## 🙋 Frequently Asked Questions (FAQ)

### Memory & Performance

**Q: Why can I run a 1B model with a huge context, but not a 7B?**
A: Because hidden size is smaller in 1B models, so each token's KV cache footprint is much smaller. The KV cache grows with both context length and model size - a 7B model's attention states require significantly more memory per token than a 1B model.

**Q: Does quantization help with memory usage?**
A: Quantization (like Q4_K_M, Q5_K_M) shrinks model weights significantly, but doesn't touch KV cache memory usage — unless you use KV quantization. In llama-cpp, you can use `--cache-type` options like `q8_0` or `q4_0` to quantize the KV cache as well.

**Q: What's `n_gpu_layers` and should I use it?**
A: It's how many transformer layers you offload to GPU (Metal on Mac). It can speed up inference significantly but doesn't change KV cache memory usage much. On MacBook with limited unified memory, sometimes keeping everything on CPU is more stable.

**Q: What's the safest setting for my MacBook Air 8GB?**
A: Keep `n_ctx` around 1024–1536 for larger models (7B+), and increase only if you have memory headroom. Always set `use_mmap: true` and `use_mlock: false` on macOS. Start with 1B-2B models for best experience.

**Q: My system gets slow/hot when running larger models. What's happening?**
A: Large models can cause memory pressure, forcing macOS to swap to disk. This creates heat and slowdown. Try smaller models (1B-2B parameters) or reduce `n_ctx` to 512-1024. Monitor Activity Monitor for memory pressure.

### Configuration & Parameters

**Q: What do all these parameters in `llm_params` actually do?**
A: Here's a breakdown:
- `n_ctx`: Context window size (how much conversation history the model remembers)
- `temperature`: Creativity/randomness (0.0=deterministic, 1.0=very creative)  
- `top_p`: Nucleus sampling (0.9 is usually good - lower = more focused)
- `max_tokens`: Maximum response length
- `n_threads`: CPU threads to use (usually 4-8 for good performance)
- `use_mmap`: Memory mapping for efficient loading (always `true` on macOS)
- `use_mlock`: Lock model in RAM (set to `false` on macOS to avoid issues)

**Q: How do I choose the right `temperature` value?**
A: 
- **0.0-0.3**: Factual, deterministic responses (good for coding, math)
- **0.3-0.7**: Balanced creativity and accuracy (general chat)
- **0.7-1.0**: Creative writing, brainstorming
- **1.0+**: Very creative but potentially incoherent

**Q: What's the difference between `n_ctx` and `max_tokens`?**
A: `n_ctx` is the total context window (input + output), while `max_tokens` is the maximum length of just the response. Think of `n_ctx` as the model's "memory" and `max_tokens` as "how long it can talk."

### Model Selection & Profiles

**Q: Which model should I start with?**
A: For 8GB RAM systems:
1. **TinyLlama 1.1B**: Ultra-fast, great for testing and simple tasks
2. **DeepSeek-R1-Distill-Qwen-1.5B**: Current default, excellent reasoning
3. **Llama 3.2 1B**: Good balance of capability and speed

For 16GB+ RAM:
- **Mistral 7B**: Excellent general capability
- **Llama 3.1 8B**: Strong reasoning and coding

**Q: What's the difference between reasoning profiles?**
A:
- `deepseek`: Direct answers only (hides thinking process)
- `deepseek_reasoning`: Shows final answer but hides internal reasoning
- `deepseek_show_reasoning`: Shows full thinking process in `<think>` tags
- `generic`: Universal format that works with most models

**Q: My model keeps generating weird tokens or doesn't stop properly. Help?**
A: This is usually a prompt format issue. Try:
1. Switch to `generic` profile first
2. Use `--debug` to see the exact prompt being sent
3. Adjust `stop_tokens` in your profile
4. Some quantized models need specific stop tokens - check the model's documentation

**Q: Can I use models not in the catalog?**
A: Yes! Add them to `available_models` in `config.json`:
```json
{
  "available_models": [
    {
      "id": "my-model",
      "name": "My Custom Model",
      "model": {
        "repo": "username/repo-name",
        "file": "model-file.gguf",
        "cache_dir": ".models/custom"
      },
      "format": "gguf",
      "quant": "Q4_K_M",
      "context_len_hint": "4k",
      "notes": "Custom model for specific use case",
      "recommended_profile": "generic",
      "recommended_llm_params": {
        "n_ctx": 1536,
        "n_batch": 8,
        "n_threads": 4,
        "temperature": 0.3,
        "max_tokens": 512,
        "use_mlock": false,
        "use_mmap": true,
        "n_gpu_layers": 0
      }
    }
  ]
}
```

### Troubleshooting

**Q: The model downloaded but won't load. What's wrong?**
A: Common causes:
1. **Corrupted download**: Try deleting the model file and re-downloading
2. **Wrong format**: Ensure you're using GGUF format files
3. **Insufficient memory**: Try a smaller model or reduce `n_ctx`
4. **File permissions**: Check that the cache directory is writable

**Q: Responses are very slow. How can I speed things up?**
A: Try these optimizations:
1. Increase `n_threads` (try 4-8)
2. Reduce `n_ctx` to 512-1024  
3. Use a smaller model (1B instead of 7B)
4. On Mac: Enable Metal GPU acceleration with `n_gpu_layers`
5. Ensure `use_mmap: true` for faster loading

**Q: The model gives different answers each time. Is this normal?**
A: Yes, unless `temperature` is 0.0. AI models are probabilistic by nature. To get more consistent responses:
- Lower `temperature` (try 0.1-0.3)
- Lower `top_p` (try 0.7-0.8)
- Some randomness is usually desirable for natural conversation

**Q: I get "import errors" or "module not found" - what's missing?**
A: Make sure you:
1. Activated your virtual environment: `source .venv/bin/activate`
2. Installed requirements: `pip install -r requirements.txt`
3. Have Python 3.8+ installed
4. On macOS, you might need Xcode command line tools: `xcode-select --install`

### Development & Customization

**Q: How do I create a custom prompt profile?**
A: Add a new profile to `config.json`:
```json
{
  "profiles": {
    "my_custom": {
      "format": "auto",
      "stop_tokens": ["Human:", "AI:", "\n\n"],
      "description": "My custom format for specific use case",
      "show_thinking": true
    }
  }
}
```

**Q: Can I run this in a script or automation?**
A: Yes! Use echo commands:
```bash
echo "What is 2+2?" | python main.py
echo -e "Question 1\n:q" | python main.py --debug
```

**Q: How do I contribute new models to the catalog?**
A: We welcome contributions! Test your model thoroughly on 8GB systems and submit a PR with:
- Model entry in the catalog format
- Performance notes and memory requirements
- Recommended profile and configuration
- Basic functionality validation

**Q: Is there conversation history? Why does it forget previous messages?**
A: This alpha version (v0.1.x) operates in single-shot mode - each message is independent. This is intentional for:
- Memory optimization on 8GB systems
- Simplicity and reliability
- Testing core functionality

Conversation history is planned for v0.2.x series.
