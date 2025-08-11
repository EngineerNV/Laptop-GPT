#!/usr/bin/env python3
"""
Test runner script for Laptop-GPT

This script provides an easy way to run the test suite with various options.
"""
import subprocess
import sys
import os
from pathlib import Path


def run_tests(test_type="all", verbose=False, coverage=False):
    """
    Run the test suite.
    
    Args:
        test_type: Type of tests to run ("all", "unit", "integration", "fast")
        verbose: Whether to run in verbose mode
        coverage: Whether to generate coverage reports
    """
    # Change to project root directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Add coverage if requested
    if coverage:
        cmd.extend(["--cov=.", "--cov-report=term-missing", "--cov-report=html"])
    
    # Select test type
    if test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "fast":
        cmd.extend(["-m", "not slow"])
    
    # Add test directory
    cmd.append("tests/")
    
    print(f"Running command: {' '.join(cmd)}")
    return subprocess.run(cmd)


def install_test_dependencies():
    """Install test dependencies."""
    deps = [
        "pytest>=7.0.0",
        "pytest-mock>=3.10.0", 
        "pytest-cov>=4.0.0",
        "pytest-datadir>=1.4.1"
    ]
    
    cmd = ["python", "-m", "pip", "install"] + deps
    print(f"Installing test dependencies: {' '.join(deps)}")
    return subprocess.run(cmd)


def main():
    """Main function for test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Laptop-GPT Test Runner")
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "fast"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Run tests in verbose mode"
    )
    parser.add_argument(
        "--coverage", "-c",
        action="store_true", 
        help="Generate coverage reports"
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install test dependencies"
    )
    
    args = parser.parse_args()
    
    if args.install_deps:
        result = install_test_dependencies()
        if result.returncode != 0:
            print("Failed to install test dependencies")
            return result.returncode
    
    return run_tests(args.type, args.verbose, args.coverage).returncode


if __name__ == "__main__":
    sys.exit(main())
