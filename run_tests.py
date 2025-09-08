#!/usr/bin/env python3
"""
Test runner for the Cost Forecasting Project.
Runs all tests with proper setup and reporting.
"""

import unittest
import sys
import os
from pathlib import Path
import coverage
import argparse

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_tests(test_pattern='test_*.py', verbosity=2, with_coverage=True):
    """Run all tests with optional coverage reporting."""
    
    # Initialize coverage if requested
    cov = None
    if with_coverage:
        cov = coverage.Coverage(source=['app', 'data_generation', 'ml_models'])
        cov.start()
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = project_root / 'tests'
    
    suite = loader.discover(
        start_dir=str(start_dir),
        pattern=test_pattern,
        top_level_dir=str(project_root)
    )
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity, buffer=True)
    result = runner.run(suite)
    
    # Generate coverage report
    if with_coverage and cov:
        cov.stop()
        cov.save()
        
        print("\n" + "="*80)
        print("COVERAGE REPORT")
        print("="*80)
        
        cov.report(show_missing=True)
        
        # Generate HTML coverage report
        html_dir = project_root / 'htmlcov'
        cov.html_report(directory=str(html_dir))
        print(f"\nDetailed HTML coverage report generated in: {html_dir}")
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nResult: {'PASSED' if success else 'FAILED'}")
    
    return success

def run_specific_test_suite(suite_name, verbosity=2):
    """Run a specific test suite."""
    
    suite_modules = {
        'models': 'tests.test_models',
        'routes': 'tests.test_routes',
        'services': 'tests.test_services'
    }
    
    if suite_name not in suite_modules:
        print(f"Unknown test suite: {suite_name}")
        print(f"Available suites: {list(suite_modules.keys())}")
        return False
    
    # Import and run specific module
    module_name = suite_modules[suite_name]
    loader = unittest.TestLoader()
    
    try:
        module = __import__(module_name, fromlist=[''])
        suite = loader.loadTestsFromModule(module)
        
        runner = unittest.TextTestRunner(verbosity=verbosity, buffer=True)
        result = runner.run(suite)
        
        success = len(result.failures) == 0 and len(result.errors) == 0
        print(f"\n{suite_name.upper()} tests: {'PASSED' if success else 'FAILED'}")
        
        return success
        
    except ImportError as e:
        print(f"Failed to import test suite {suite_name}: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are available for testing."""
    
    missing_deps = []
    
    try:
        import flask
    except ImportError:
        missing_deps.append('flask')
    
    try:
        import pandas
    except ImportError:
        missing_deps.append('pandas')
    
    try:
        import numpy
    except ImportError:
        missing_deps.append('numpy')
    
    try:
        import statsmodels
    except ImportError:
        missing_deps.append('statsmodels')
    
    try:
        import coverage
    except ImportError:
        print("Warning: coverage not available. Install with: pip install coverage")
    
    if missing_deps:
        print("Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main test runner function."""
    
    parser = argparse.ArgumentParser(description="Run tests for Cost Forecasting Project")
    parser.add_argument('--suite', '-s', choices=['models', 'routes', 'services'],
                       help='Run specific test suite only')
    parser.add_argument('--pattern', '-p', default='test_*.py',
                       help='Test file pattern (default: test_*.py)')
    parser.add_argument('--no-coverage', action='store_true',
                       help='Disable coverage reporting')
    parser.add_argument('--verbose', '-v', action='count', default=2,
                       help='Increase verbosity (use -v, -vv, etc.)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    
    # Set verbosity
    verbosity = 1 if args.quiet else min(args.verbose, 3)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    print("Cost Forecasting Project - Test Suite")
    print("="*80)
    
    success = True
    
    if args.suite:
        # Run specific test suite
        success = run_specific_test_suite(args.suite, verbosity)
    else:
        # Run all tests
        success = run_tests(
            test_pattern=args.pattern,
            verbosity=verbosity,
            with_coverage=not args.no_coverage
        )
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())