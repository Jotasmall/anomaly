"""
Test Suite for Colombian Water Level Anomaly Detection System
==============================================================
Run this script to verify your installation is working correctly.
"""

import sys
import warnings
warnings.filterwarnings('ignore')


def test_imports():
    """Test that all required libraries can be imported."""
    print("\n" + "="*70)
    print("TEST 1: Checking Required Libraries")
    print("="*70)
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib'
    }
    
    missing_packages = []
    
    for import_