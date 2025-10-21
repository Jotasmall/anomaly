"""
Example Usage Scripts for Colombian Water Anomaly Detector
===========================================================
Demonstrates various use cases and comparisons of different methods.
"""

import pandas as pd
import numpy as np
from water_anomaly_detector import ColombianWaterAnomalyDetector, generate_colombian_data
import matplotlib.pyplot as plt


def example_1_basic_usage():
    """
    Example 1: Basic usage with default settings
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage")
    print("="*70)
    
    # Generate synthetic data
    df = generate_colombian_data(n_days=365)
    
    # Create detector with default settings
    detector = ColombianWaterAnomalyDetector()
    
    # Run pipeline
    detector.load_data(df).preprocess().decompose().detect_anomalies()
    
    # Get results
    results = detector.get_results()
    summary = detector.get_anomaly_summary()
    
    print(f"\nDetected {summary['total_anomalies']} anomalies")
    print(f"Floods: {summary['flood_events']}, Droughts: {summary['drought_events']}")
    
    # Save results to CSV
    results.to_csv('results_basic.csv', index=False)
    print("\nResults saved to 'results_basic.csv'")
    
    # Plot
    detector.plot_results()


def example_2_compare_imputation():
    """
    Example 2: Compare different imputation methods
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Comparing Imputation Methods")
    print("="*70)
    
    # Generate data
    df = generate_colombian_data(n_days=365)
    
    methods = ['linear', 'forward', 'seasonal']
    results_dict = {}
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    for idx, method in enumerate(methods):
        print(f"\nTesting {method} imputation...")
        
        detector = ColombianWaterAnomalyDetector(
            imputation_method=method,
            seasonal_method='stl',
            anomaly_method='zscore'
        )
        
        detector.load_data(df).preprocess().decompose().detect_anomalies()
        results = detector.get_results()
        results_dict[method] = results
        
        # Plot comparison
        axes[idx].plot(results['timestamp'], results['water_level'], 
                      label=f'{method.capitalize()} Imputation')
        anomalies = results[results['is_anomaly']]
        axes[idx].scatter(anomalies['timestamp'], anomalies['water_level'],
                         c='red', s=50, marker='x', label='Anomalies')
        axes[idx].set_title(f'{method.capitalize()} Imputation - '
                           f'{len(anomalies)} anomalies detected')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('imputation_comparison.png', dpi=300)
    print("\nComparison plot saved to 'imputation_comparison.png'")
    plt.show()


def example_3_compare_seasonal():
    """
    Example 3: Compare STL vs Prophet decomposition
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Comparing Seasonal Decomposition Methods")
    print("="*70)
    
    df = generate_colombian_data(n_days=730)  # 2 years
    
    methods = ['stl', 'prophet']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for idx, method in enumerate(methods):
        print(f"\nTesting {method} decomposition...")
        
        detector = ColombianWaterAnomalyDetector(
            imputation_method='linear',
            seasonal_method=method,
            anomaly_method='zscore'
        )
        
        detector.load_data(df).preprocess().decompose().detect_anomalies()
        results = detector.get_results()
        
        # Plot trend
        axes[idx, 0].plot(results['timestamp'], results['trend'], 
                         label=f'{method.upper()} Trend')
        axes[idx, 0].set_title(f'{method.upper()} - Trend')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Plot seasonal
        axes[idx, 1].plot(results['timestamp'], results['seasonal'], 
                         label=f'{method.upper()} Seasonal', color='orange')
        axes[idx, 1].set_title(f'{method.upper()} - Seasonal (Bimodal)')
        axes[idx, 1].legend()
        axes[idx, 1].grid(True, alpha=0.3)
        
        # Plot residual
        axes[idx, 2].plot(results['timestamp'], results['residual'], 
                         label=f'{method.upper()} Residual', color='gray', alpha=0.5)
        axes[idx, 2].set_title(