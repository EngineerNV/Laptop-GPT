# Laptop-GPT

This is a repository dedicated to using opensource models locally on your machine - whether that be a mac or windows laptop. Given hardware limitations the focus is to be able to get a smaller version running that will be compatiable with most systems. Then evaluate its usefullness - possibly with automation and a web agent.

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
