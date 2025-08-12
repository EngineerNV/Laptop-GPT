#!/usr/bin/env python3
"""
Simple Model Test - Test all models with their recommended profiles.

This validates that each model in the catalog:
1. Has valid configuration
2. Can be downloaded 
3. Can be loaded with llama-cpp-python
4. Can generate responses with its recommended profile

Usage:
    python simple_model_test.py                 # Test all models
    python simple_model_test.py --model tinyllama # Test specific model
    python simple_model_test.py --list           # Show available models
    python simple_model_test.py --small          # Test only small models (1-2B)
"""

import argparse
import sys
import json
import tempfile
import shutil
from pathlib import Path

from config_utils import load_config
from llm_utils import setup_llm, download_model, format_prompt


def validate_model(model_id: str, config_path: str = "config.json") -> bool:
    """Test a single model end-to-end."""
    print(f"\n🧪 Testing model: {model_id}")
    print("=" * 50)
    
    try:
        # Load config
        config = load_config(config_path)
        
        # Find the model
        models = config.get("available_models", [])
        model_data = None
        for model in models:
            if model.get("id") == model_id:
                model_data = model
                break
        
        if not model_data:
            print(f"❌ Model '{model_id}' not found in config")
            return False
        
        print(f"📋 Model: {model_data.get('name', 'Unknown')}")
        
        # Get model configuration
        model_config = model_data["model"]
        repo = model_config["repo"]
        file = model_config["file"]
        cache_dir = model_config["cache_dir"]
        
        print(f"📦 Repository: {repo}")
        print(f"📄 File: {file}")
        
        # Test download
        print("\n⬇️  Testing download...")
        try:
            model_path = download_model(repo, file, cache_dir)
            print(f"✅ Download successful: {model_path}")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
        
        # Test setup
        print("\n🔧 Testing model setup...")
        try:
            llm_params = config.get("llm_params", {})
            # Use smaller parameters for testing
            test_params = {
                "n_ctx": 256,
                "n_batch": 2,
                "n_threads": 1,
                "temperature": 0.1,
                "max_tokens": 20,
                "use_mlock": False,
                "use_mmap": True,
                "n_gpu_layers": 0
            }
            
            llm = setup_llm(model_path, test_params, debug_mode=False)
            print("✅ Model setup successful")
        except Exception as e:
            print(f"❌ Model setup failed: {e}")
            return False
        
        # Test prompt formatting
        print("\n📝 Testing prompt formatting...")
        profiles = config.get("profiles", {})
        recommended_profile = model_data.get("recommended_profile", "generic")
        
        if recommended_profile not in profiles:
            print(f"⚠️  Profile '{recommended_profile}' not found, using 'generic'")
            recommended_profile = "generic"
        
        profile = profiles.get(recommended_profile, {})
        format_type = profile.get("format", "auto")
        
        test_question = "What is 2+2?"
        formatted_prompt = format_prompt(test_question, format_type)
        
        print(f"✅ Using profile: {recommended_profile} (format: {format_type})")
        print(f"📝 Formatted prompt: {repr(formatted_prompt[:100])}...")
        
        # Test response generation
        print("\n🤖 Testing response generation...")
        try:
            stop_tokens = profile.get("stop_tokens", ["Human:", "User:", "\n\n"])
            
            response = llm(
                formatted_prompt,
                max_tokens=test_params["max_tokens"],
                temperature=test_params["temperature"],
                stop=stop_tokens,
                echo=False
            )
            
            response_text = response["choices"][0]["text"].strip()
            
            if response_text:
                print(f"✅ Response generated: {repr(response_text)}")
                
                # Basic validation - should contain some form of "4"
                if "4" in response_text or "four" in response_text.lower():
                    print("✅ Response appears correct (contains '4')")
                else:
                    print("⚠️  Response may be incorrect (no '4' found)")
                
                return True
            else:
                print("❌ Empty response generated")
                return False
                
        except Exception as e:
            print(f"❌ Response generation failed: {e}")
            return False
    
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False


def validate_all_models(config_path: str = "config.json", small_only: bool = False) -> dict:
    """Test all models in the catalog."""
    print("🧪 Testing All Models with Recommended Profiles")
    print("=" * 60)
    
    try:
        config = load_config(config_path)
        models = config.get("available_models", [])
        
        if small_only:
            # Filter to small models only (1-2B parameters)
            small_models = []
            for model in models:
                name = model.get("name", "").lower()
                if any(size in name for size in ["1.1b", "1.5b", "1b", "2b"]):
                    small_models.append(model)
            models = small_models
            print(f"Testing {len(models)} small models (1-2B parameters)")
        else:
            print(f"Testing {len(models)} models")
        
        if not models:
            print("No models found to test")
            return {}
        
        results = {}
        for i, model_data in enumerate(models, 1):
            model_id = model_data.get("id", "unknown")
            print(f"\n{i}/{len(models)}. Testing {model_id}")
            print("-" * 40)
            
            success = validate_model(model_id, config_path)
            results[model_id] = success
            
            if success:
                print(f"✅ {model_id} - PASSED")
            else:
                print(f"❌ {model_id} - FAILED")
        
        # Print summary
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        
        passed = sum(1 for success in results.values() if success)
        total = len(results)
        
        for model_id, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{model_id:20} {status}")
        
        print(f"\nTotal: {passed}/{total} models passed")
        if passed == total:
            print("🎉 All models working correctly!")
        else:
            print(f"⚠️  {total - passed} models need attention")
        
        return results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return {}
    """Test a single model end-to-end."""
    print(f"\n🧪 Testing model: {model_id}")
    print("=" * 50)
    
    try:
        # Load config
        config = load_config(config_path)
        
        # Find the model
        models = config.get("available_models", [])
        model_data = None
        for model in models:
            if model.get("id") == model_id:
                model_data = model
                break
        
        if not model_data:
            print(f"❌ Model '{model_id}' not found in config")
            return False
        
        print(f"📋 Model: {model_data.get('name', 'Unknown')}")
        
        # Get model configuration
        model_config = model_data["model"]
        repo = model_config["repo"]
        file = model_config["file"]
        cache_dir = model_config["cache_dir"]
        
        print(f"📦 Repository: {repo}")
        print(f"📄 File: {file}")
        
        # Test download
        print("\n⬇️  Testing download...")
        try:
            model_path = download_model(repo, file, cache_dir)
            print(f"✅ Download successful: {model_path}")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
        
        # Test setup
        print("\n🔧 Testing model setup...")
        try:
            llm_params = config.get("llm_params", {})
            # Use smaller parameters for testing
            test_params = {
                "n_ctx": 256,
                "n_batch": 2,
                "n_threads": 1,
                "temperature": 0.1,
                "max_tokens": 20,
                "use_mlock": False,
                "use_mmap": True,
                "n_gpu_layers": 0
            }
            
            llm = setup_llm(model_path, test_params, debug_mode=False)
            print("✅ Model setup successful")
        except Exception as e:
            print(f"❌ Model setup failed: {e}")
            return False
        
        # Test prompt formatting
        print("\n📝 Testing prompt formatting...")
        profiles = config.get("profiles", {})
        recommended_profile = model_data.get("recommended_profile", "generic")
        
        if recommended_profile not in profiles:
            print(f"⚠️  Profile '{recommended_profile}' not found, using 'generic'")
            recommended_profile = "generic"
        
        profile = profiles.get(recommended_profile, {})
        format_type = profile.get("format", "auto")
        
        test_question = "What is 2+2?"
        formatted_prompt = format_prompt(test_question, format_type)
        
        print(f"✅ Using profile: {recommended_profile} (format: {format_type})")
        print(f"📝 Formatted prompt: {repr(formatted_prompt[:100])}...")
        
        # Test response generation
        print("\n🤖 Testing response generation...")
        try:
            stop_tokens = profile.get("stop_tokens", ["Human:", "User:", "\n\n"])
            
            response = llm(
                formatted_prompt,
                max_tokens=test_params["max_tokens"],
                temperature=test_params["temperature"],
                stop=stop_tokens,
                echo=False
            )
            
            response_text = response["choices"][0]["text"].strip()
            
            if response_text:
                print(f"✅ Response generated: {repr(response_text)}")
                
                # Basic validation - should contain some form of "4"
                if "4" in response_text or "four" in response_text.lower():
                    print("✅ Response appears correct (contains '4')")
                else:
                    print("⚠️  Response may be incorrect (no '4' found)")
                
                return True
            else:
                print("❌ Empty response generated")
                return False
                
        except Exception as e:
            print(f"❌ Response generation failed: {e}")
            return False
    
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False


def list_models(config_path: str = "config.json"):
    """List available models."""
    try:
        config = load_config(config_path)
        models = config.get("available_models", [])
        
        print("\nAvailable models for testing:")
        print("=" * 40)
        
        for model in models:
            model_id = model.get("id", "unknown")
            name = model.get("name", "Unknown")
            quant = model.get("quant", "Unknown")
            profile = model.get("recommended_profile", "generic")
            
            print(f"ID: {model_id}")
            print(f"  Name: {name}")
            print(f"  Quantization: {quant}")
            print(f"  Profile: {profile}")
            print()
            
    except Exception as e:
        print(f"Error loading config: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Simple model testing for Laptop-GPT",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--model",
        help="Model ID to test (if not specified, tests all models)"
    )
    
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit"
    )
    
    parser.add_argument(
        "--small",
        action="store_true",
        help="Test only small models (1-2B parameters)"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_models(args.config)
        return 0
    
    print("🧪 Simple Model Tester for Laptop-GPT")
    print("====================================")
    
    if args.model:
        # Test single model
        success = validate_model(args.model, args.config)
        if success:
            print(f"\n🎉 Model '{args.model}' test PASSED")
            return 0
        else:
            print(f"\n💥 Model '{args.model}' test FAILED")
            return 1
    else:
        # Test all models
        results = validate_all_models(args.config, args.small)
        if not results:
            return 1
        
        # Return 0 if all passed, 1 if any failed
        all_passed = all(results.values())
        return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
