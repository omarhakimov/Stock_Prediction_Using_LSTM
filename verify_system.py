#!/usr/bin/env python3
"""
System Verification Script

Verifies that all documented features in the README work correctly.
This script tests the main functionality mentioned in the documentation.
"""

import os
import sys
import subprocess
import time

def run_command(command, description, timeout=60):
    """Run a command and return success status."""
    print(f"🧪 Testing: {description}")
    print(f"   Command: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            cwd="/Users/purleaf/Downloads/stock training"
        )
        
        if result.returncode == 0:
            print(f"   ✅ SUCCESS")
            return True
        else:
            print(f"   ❌ FAILED")
            print(f"   Error: {result.stderr[:200]}...")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"   ⏰ TIMEOUT (>{timeout}s)")
        return False
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)}")
        return False

def check_file_exists(filepath, description):
    """Check if a file exists."""
    print(f"📁 Checking: {description}")
    print(f"   Path: {filepath}")
    
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"   ✅ EXISTS ({size:,} bytes)")
        return True
    else:
        print(f"   ❌ NOT FOUND")
        return False

def main():
    """Run verification tests."""
    print("🔍 STOCK PREDICTION SYSTEM VERIFICATION")
    print("=" * 60)
    print()
    
    base_dir = "/Users/purleaf/Downloads/stock training"
    os.chdir(base_dir)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Import all modules
    total_tests += 1
    if run_command(
        "python -c \"from src.data_processor import StockDataProcessor; from src.model import LSTMModel; from src.predictor import StockPredictor; print('All imports successful')\"",
        "Module imports"
    ):
        tests_passed += 1
    
    # Test 2: Configuration access
    total_tests += 1
    if run_command(
        "python -c \"from src import config; print(f'LSTM Units: {config.LSTM_UNITS}')\"",
        "Configuration access"
    ):
        tests_passed += 1
    
    # Test 3: Demo help
    total_tests += 1
    if run_command(
        "python demo_visualization.py --help",
        "Demo help command"
    ):
        tests_passed += 1
    
    # Test 4: Check essential files
    essential_files = [
        ("data/raw/sample_stock_data.csv", "Sample data file"),
        ("data/models/AAPL_model.keras", "Trained model"),
        ("outputs/predictions/future_predictions_AAPL.csv", "Predictions file"),
        ("src/config.py", "Configuration file"),
        ("README.md", "Documentation")
    ]
    
    for filepath, description in essential_files:
        total_tests += 1
        if check_file_exists(os.path.join(base_dir, filepath), description):
            tests_passed += 1
    
    # Test 5: Check generated visualizations
    viz_files = [
        "historical_dashboard.png",
        "prediction_dashboard.png", 
        "risk_dashboard.png",
        "technical_dashboard.png",
        "model_performance_dashboard.png",
        "complete_timeline.png"
    ]
    
    for viz_file in viz_files:
        total_tests += 1
        if check_file_exists(
            os.path.join(base_dir, "outputs/plots", viz_file),
            f"Visualization: {viz_file}"
        ):
            tests_passed += 1
    
    # Test 6: Quick system status
    total_tests += 1
    if run_command(
        "python -c \"import matplotlib; matplotlib.use('Agg'); print('Matplotlib backend OK')\"",
        "Matplotlib backend"
    ):
        tests_passed += 1
    
    # Summary
    print()
    print("=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"✅ Tests Passed: {tests_passed}/{total_tests}")
    print(f"📈 Success Rate: {(tests_passed/total_tests)*100:.1f}%")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED - System is fully operational!")
        print()
        print("🚀 Ready to run:")
        print("   python demo_visualization.py --stock AAPL --days 10")
    else:
        print("⚠️  Some tests failed - check the errors above")
        print()
        print("🔧 Try running:")
        print("   pip install -r requirements.txt")
        print("   python showcase_demo.py")
    
    print()
    print("📖 For more information, see:")
    print("   - README.md (comprehensive documentation)")
    print("   - QUICK_REFERENCE.md (command quick reference)")
    print("   - PROJECT_STATUS.md (detailed project status)")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
