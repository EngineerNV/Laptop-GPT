# Testing Guide for Laptop-GPT

This document explains how to run and understand the tests for Laptop-GPT.

## Quick Start

### Install Test Dependencies
```bash
python run_tests.py --install-deps
```

### Run All Tests
```bash
python run_tests.py
```

### Run with Coverage
```bash
python run_tests.py --coverage
```

## Test Structure

### Test Files
- `test_config_utils.py` - Tests configuration loading and validation
- `test_llm_utils.py` - Tests LLM setup, model downloading, and prompts  
- `test_cli.py` - Tests command line interface
- `test_main_integration.py` - Integration tests for the main application

### Test Categories

#### Unit Tests
Focus on individual functions and classes:
```bash
python run_tests.py --type unit
```

#### Integration Tests  
Test how components work together:
```bash
python run_tests.py --type integration
```

#### Fast Tests
Skip slow operations (like actual downloads):
```bash
python run_tests.py --type fast
```

## Learning from the Tests

### 1. Configuration Testing (`test_config_utils.py`)

**Key Learning Points:**
- **Mocking**: How to mock file operations and environment variables
- **Exception Testing**: Using `pytest.raises()` to test error conditions
- **Fixtures**: Reusable test data and setup

**Example Test Pattern:**
```python
def test_load_nonexistent_config(self, nonexistent_config_path):
    with pytest.raises(FileNotFoundError) as exc_info:
        load_config(nonexistent_config_path)
    assert "Required configuration file" in str(exc_info.value)
```

**What This Teaches:**
- How to test that your code properly handles missing files
- How to verify error messages contain expected content
- How to use fixtures for test data

### 2. LLM Testing (`test_llm_utils.py`)

**Key Learning Points:**
- **Mocking External Dependencies**: How to mock HuggingFace Hub and LlamaCpp
- **File System Testing**: Creating temporary files for testing
- **Parameter Validation**: Testing that functions are called with correct arguments

**Example Test Pattern:**
```python
@patch('llm_utils.LlamaCpp')
def test_setup_llm_with_valid_model(self, mock_llama, mock_model_file, sample_llm_params):
    mock_instance = MagicMock()
    mock_llama.return_value = mock_instance
    
    llm = setup_llm(mock_model_file, sample_llm_params, debug_mode=False)
    
    # Verify LlamaCpp was called with correct parameters
    call_args = mock_llama.call_args[1]
    assert call_args['model_path'] == mock_model_file
    assert call_args['n_ctx'] == 1024
```

**What This Teaches:**
- How to mock complex external libraries
- How to verify function calls and their parameters
- How to test without actually downloading large model files

### 3. CLI Testing (`test_cli.py`)

**Key Learning Points:**
- **Argument Parsing**: How to test command line interfaces
- **System Exit Handling**: Testing functions that call `exit()`
- **Output Capture**: Testing printed output

**Example Test Pattern:**
```python
def test_parse_debug_flag(self):
    with patch.object(sys, 'argv', ['main.py', '--debug']):
        args = parse_arguments()
        assert args.debug == True
```

**What This Teaches:**
- How to simulate command line arguments in tests
- How to test boolean flags and string arguments
- How to handle functions that print and exit

### 4. Integration Testing (`test_main_integration.py`)

**Key Learning Points:**
- **End-to-End Testing**: Testing complete workflows
- **Complex Mocking**: Coordinating multiple mocks
- **User Interaction**: Simulating user input and responses

**Example Test Pattern:**
```python
@patch('main.download_model')
@patch('main.setup_llm')
@patch('builtins.input', side_effect=[':q'])
def test_main_successful_flow(self, mock_input, mock_setup_llm, mock_download, valid_config_path):
    # Test complete application flow
    result = main.main()
    assert result == 0
```

**What This Teaches:**
- How to test complete application workflows
- How to mock user input and system interactions
- How to verify that multiple components work together correctly

## Test Fixtures (conftest.py)

**Key Learning Points:**
- **Reusable Test Data**: Creating fixtures for common test needs
- **Temporary Resources**: Managing temporary files and directories
- **Test Isolation**: Ensuring tests don't interfere with each other

**Example Fixtures:**
```python
@pytest.fixture
def temp_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)  # Cleanup
```

## Common Testing Patterns

### 1. Testing Error Conditions
```python
def test_error_handling(self):
    with pytest.raises(SpecificException) as exc_info:
        function_that_should_fail()
    assert "expected error message" in str(exc_info.value)
```

### 2. Mocking External Dependencies
```python
@patch('module.external_dependency')
def test_with_mock(self, mock_dependency):
    mock_dependency.return_value = "test_value"
    result = function_under_test()
    assert result == "expected_result"
    mock_dependency.assert_called_once()
```

### 3. Testing Configuration
```python
def test_config_validation(self, config_fixture):
    config = load_config(config_fixture)
    assert "required_key" in config
    assert config["required_key"]["sub_key"] == "expected_value"
```

### 4. Testing File Operations
```python
def test_file_handling(self, temp_dir):
    test_file = os.path.join(temp_dir, "test.txt")
    with open(test_file, 'w') as f:
        f.write("test content")
    
    result = function_that_reads_file(test_file)
    assert result == "expected_result"
```

## Running Specific Tests

### Run a Single Test File
```bash
python -m pytest tests/test_config_utils.py -v
```

### Run a Single Test Class
```bash
python -m pytest tests/test_config_utils.py::TestLoadConfig -v
```

### Run a Single Test Method
```bash
python -m pytest tests/test_config_utils.py::TestLoadConfig::test_load_valid_config -v
```

### Run Tests Matching a Pattern
```bash
python -m pytest -k "config" -v  # Run all tests with "config" in the name
```

## Understanding Test Output

### Success
```
test_config_utils.py::TestLoadConfig::test_load_valid_config PASSED
```

### Failure
```
test_config_utils.py::TestLoadConfig::test_load_valid_config FAILED
E   AssertionError: assert 'actual' == 'expected'
```

### Coverage Report
```
Name                 Stmts   Miss  Cover   Missing
--------------------------------------------------
config_utils.py         45      3    93%   23-25
llm_utils.py           67      5    93%   45, 78-81
```

## Best Practices for Learning

1. **Start Small**: Run individual test files to understand one module at a time
2. **Read Test Names**: Test names describe what they're testing
3. **Examine Fixtures**: Understand how test data is created and managed
4. **Look at Mocks**: See how external dependencies are simulated
5. **Check Assertions**: Understand what each test is verifying
6. **Run with Coverage**: See which parts of your code are tested

## Next Steps

1. Install dependencies: `python run_tests.py --install-deps`
2. Run all tests: `python run_tests.py --coverage`
3. Study the test files to understand the patterns
4. Try modifying tests to see how they break
5. Write your own tests for new features you add
