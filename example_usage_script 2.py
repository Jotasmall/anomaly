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
        axes[idx, 2].set_title(f'{method.upper()} - Residual')
        axes[idx, 2].legend()
        axes[idx, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('seasonal_comparison.png', dpi=300)
    print("\nComparison plot saved to 'seasonal_comparison.png'")
    plt.show()


def example_4_compare_anomaly_detection():
    """
    Example 4: Compare all three anomaly detection methods
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Comparing Anomaly Detection Methods")
    print("="*70)
    
    df = generate_colombian_data(n_days=730)
    
    methods = ['zscore', 'isolation_forest', 'autoencoder']
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    comparison_results = []
    
    for idx, method in enumerate(methods):
        print(f"\nTesting {method} detection...")
        
        detector = ColombianWaterAnomalyDetector(
            imputation_method='linear',
            seasonal_method='prophet',
            anomaly_method=method,
            zscore_threshold=3.0
        )
        
        detector.load_data(df).preprocess().decompose().detect_anomalies()
        results = detector.get_results()
        summary = detector.get_anomaly_summary()
        
        # Store for comparison
        comparison_results.append({
            'method': method,
            'total_anomalies': summary['total_anomalies'],
            'floods': summary['flood_events'],
            'droughts': summary['drought_events'],
            'mean_score': summary['mean_anomaly_score']
        })
        
        # Plot
        axes[idx].plot(results['timestamp'], results['water_level'], 
                      'b-', alpha=0.4, label='Water Level')
        anomalies = results[results['is_anomaly']]
        axes[idx].scatter(anomalies['timestamp'], anomalies['water_level'],
                         c='red', s=80, marker='*', label='Anomaly', zorder=5)
        axes[idx].set_title(f'{method.upper()} - {len(anomalies)} anomalies detected')
        axes[idx].set_ylabel('Water Level (m)')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Date')
    
    plt.tight_layout()
    plt.savefig('anomaly_detection_comparison.png', dpi=300)
    print("\nComparison plot saved to 'anomaly_detection_comparison.png'")
    plt.show()
    
    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    print(f"{'Method':<20} {'Total':<10} {'Floods':<10} {'Droughts':<10} {'Avg Score':<10}")
    print("-"*70)
    for result in comparison_results:
        print(f"{result['method']:<20} {result['total_anomalies']:<10} "
              f"{result['floods']:<10} {result['droughts']:<10} "
              f"{result['mean_score']:<10.2f}")


def example_5_real_world_csv():
    """
    Example 5: Load data from CSV file
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Loading Real Data from CSV")
    print("="*70)
    
    # Create example CSV file
    print("\nCreating example CSV file...")
    df = generate_colombian_data(n_days=365)
    df.to_csv('sample_water_data.csv', index=False)
    print("Sample data saved to 'sample_water_data.csv'")
    
    # Load from CSV
    print("\nLoading data from CSV...")
    df_loaded = pd.read_csv('sample_water_data.csv')
    
    # Process
    detector = ColombianWaterAnomalyDetector(
        imputation_method='seasonal',
        seasonal_method='prophet',
        anomaly_method='isolation_forest'
    )
    
    detector.load_data(df_loaded).preprocess().decompose().detect_anomalies()
    
    # Get and save results
    results = detector.get_results()
    results.to_csv('anomaly_results.csv', index=False)
    print("\nResults saved to 'anomaly_results.csv'")
    
    # Export only anomalies
    anomalies_only = results[results['is_anomaly']]
    anomalies_only.to_csv('detected_anomalies.csv', index=False)
    print(f"Detected {len(anomalies_only)} anomalies saved to 'detected_anomalies.csv'")
    
    detector.plot_results()


def example_6_sensitivity_analysis():
    """
    Example 6: Test sensitivity to Z-score threshold
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Z-Score Threshold Sensitivity Analysis")
    print("="*70)
    
    df = generate_colombian_data(n_days=730)
    
    thresholds = [2.0, 2.5, 3.0, 3.5, 4.0]
    sensitivity_results = []
    
    for threshold in thresholds:
        print(f"\nTesting threshold = {threshold}...")
        
        detector = ColombianWaterAnomalyDetector(
            imputation_method='linear',
            seasonal_method='stl',
            anomaly_method='zscore',
            zscore_threshold=threshold
        )
        
        detector.load_data(df).preprocess().decompose().detect_anomalies()
        summary = detector.get_anomaly_summary()
        
        sensitivity_results.append({
            'threshold': threshold,
            'anomalies': summary['total_anomalies'],
            'floods': summary['flood_events'],
            'droughts': summary['drought_events']
        })
    
    # Plot sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Total anomalies vs threshold
    axes[0].plot([r['threshold'] for r in sensitivity_results],
                [r['anomalies'] for r in sensitivity_results],
                'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Z-Score Threshold')
    axes[0].set_ylabel('Number of Anomalies Detected')
    axes[0].set_title('Sensitivity to Threshold Parameter')
    axes[0].grid(True, alpha=0.3)
    
    # Flood vs drought breakdown
    thresholds_list = [r['threshold'] for r in sensitivity_results]
    floods = [r['floods'] for r in sensitivity_results]
    droughts = [r['droughts'] for r in sensitivity_results]
    
    x = np.arange(len(thresholds_list))
    width = 0.35
    
    axes[1].bar(x - width/2, floods, width, label='Floods', color='blue', alpha=0.7)
    axes[1].bar(x + width/2, droughts, width, label='Droughts', color='orange', alpha=0.7)
    axes[1].set_xlabel('Z-Score Threshold')
    axes[1].set_ylabel('Number of Events')
    axes[1].set_title('Flood vs Drought Detection by Threshold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(thresholds_list)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('sensitivity_analysis.png', dpi=300)
    print("\nSensitivity analysis plot saved to 'sensitivity_analysis.png'")
    plt.show()
    
    # Print table
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS TABLE")
    print("="*70)
    print(f"{'Threshold':<12} {'Total':<10} {'Floods':<10} {'Droughts':<10}")
    print("-"*70)
    for result in sensitivity_results:
        print(f"{result['threshold']:<12.1f} {result['anomalies']:<10} "
              f"{result['floods']:<10} {result['droughts']:<10}")


def example_7_batch_processing():
    """
    Example 7: Process multiple stations in batch
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Batch Processing Multiple Stations")
    print("="*70)
    
    # Simulate multiple monitoring stations
    stations = {
        'Rio_Magdalena': {'n_days': 730, 'seed': 42},
        'Rio_Cauca': {'n_days': 730, 'seed': 43},
        'Rio_Atrato': {'n_days': 730, 'seed': 44}
    }
    
    batch_results = {}
    
    for station_name, params in stations.items():
        print(f"\nProcessing {station_name}...")
        
        # Generate station-specific data
        df = generate_colombian_data(
            n_days=params['n_days'],
            seed=params['seed']
        )
        
        # Process
        detector = ColombianWaterAnomalyDetector(
            imputation_method='linear',
            seasonal_method='prophet',
            anomaly_method='isolation_forest'
        )
        
        detector.load_data(df).preprocess().decompose().detect_anomalies()
        
        # Store results
        results = detector.get_results()
        summary = detector.get_anomaly_summary()
        
        batch_results[station_name] = {
            'data': results,
            'summary': summary
        }
        
        # Save station-specific results
        results.to_csv(f'{station_name}_results.csv', index=False)
        print(f"  - {summary['total_anomalies']} anomalies detected")
        print(f"  - Results saved to '{station_name}_results.csv'")
    
    # Create summary report
    print("\n" + "="*70)
    print("BATCH PROCESSING SUMMARY")
    print("="*70)
    print(f"{'Station':<20} {'Total':<10} {'Floods':<10} {'Droughts':<10}")
    print("-"*70)
    
    for station_name, data in batch_results.items():
        summary = data['summary']
        print(f"{station_name:<20} {summary['total_anomalies']:<10} "
              f"{summary['flood_events']:<10} {summary['drought_events']:<10}")
    
    # Create comparative visualization
    fig, axes = plt.subplots(len(stations), 1, figsize=(15, 12))
    
    for idx, (station_name, data) in enumerate(batch_results.items()):
        results = data['data']
        axes[idx].plot(results['timestamp'], results['water_level'],
                      'b-', alpha=0.4, label='Water Level')
        anomalies = results[results['is_anomaly']]
        axes[idx].scatter(anomalies['timestamp'], anomalies['water_level'],
                         c='red', s=60, marker='*', label='Anomaly', zorder=5)
        axes[idx].set_title(f'{station_name} - {len(anomalies)} anomalies')
        axes[idx].set_ylabel('Water Level (m)')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    plt.savefig('batch_processing_results.png', dpi=300)
    print("\nBatch processing plot saved to 'batch_processing_results.png'")
    plt.show()


def example_8_custom_parameters():
    """
    Example 8: Advanced usage with custom parameters
    """
    print("\n" + "="*70)
    print("EXAMPLE 8: Custom Parameters for Specific Use Case")
    print("="*70)
    
    # Generate data
    df = generate_colombian_data(n_days=1095)  # 3 years
    
    # Create detector with custom parameters
    # Scenario: High-elevation Andean reservoir with strong seasonality
    detector = ColombianWaterAnomalyDetector(
        imputation_method='seasonal',      # Better for systematic gaps
        seasonal_method='prophet',          # Better for complex patterns
        anomaly_method='isolation_forest',  # More robust to outliers
        zscore_threshold=2.5,              # More sensitive (not used in IF)
        seasonal_period=365,                # Annual cycle
        trend_window=45                     # Longer smoothing for stable trend
    )
    
    print("\nCustom configuration:")
    print(f"  - Imputation: seasonal (handles systematic gaps)")
    print(f"  - Decomposition: prophet (captures complex bimodal patterns)")
    print(f"  - Detection: isolation_forest (robust to outliers)")
    print(f"  - Trend window: 45 days (smoother trend estimation)")
    
    # Run pipeline
    detector.load_data(df).preprocess().decompose().detect_anomalies()
    
    # Get detailed results
    results = detector.get_results()
    summary = detector.get_anomaly_summary()
    
    # Analyze anomaly patterns by season
    results['month'] = pd.to_datetime(results['timestamp']).dt.month
    results['season'] = results['month'].apply(lambda m: 
        'Wet1' if m in [4, 5] else
        'Wet2' if m in [10, 11] else
        'Dry1' if m in [1, 2, 3, 12] else
        'Dry2'
    )
    
    # Count anomalies by season
    season_anomalies = results[results['is_anomaly']].groupby('season').size()
    
    print("\n" + "="*70)
    print("SEASONAL ANOMALY DISTRIBUTION")
    print("="*70)
    print(season_anomalies)
    
    # Plot seasonal distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Time series
    ax1.plot(results['timestamp'], results['water_level'], 'b-', alpha=0.4)
    anomalies = results[results['is_anomaly']]
    ax1.scatter(anomalies['timestamp'], anomalies['water_level'],
               c='red', s=60, marker='*', label='Anomaly', zorder=5)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Water Level (m)')
    ax1.set_title('3-Year Analysis with Custom Parameters')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Seasonal bar chart
    season_anomalies.plot(kind='bar', ax=ax2, color=['orange', 'blue', 'orange', 'blue'])
    ax2.set_title('Anomalies by Season')
    ax2.set_xlabel('Season')
    ax2.set_ylabel('Number of Anomalies')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('custom_parameters_results.png', dpi=300)
    print("\nCustom parameters plot saved to 'custom_parameters_results.png'")
    plt.show()


def run_all_examples():
    """
    Run all examples sequentially
    """
    print("\n" + "="*70)
    print("RUNNING ALL EXAMPLES")
    print("="*70)
    
    examples = [
        ("Basic Usage", example_1_basic_usage),
        ("Imputation Comparison", example_2_compare_imputation),
        ("Seasonal Decomposition Comparison", example_3_compare_seasonal),
        ("Anomaly Detection Comparison", example_4_compare_anomaly_detection),
        ("Real World CSV", example_5_real_world_csv),
        ("Sensitivity Analysis", example_6_sensitivity_analysis),
        ("Batch Processing", example_7_batch_processing),
        ("Custom Parameters", example_8_custom_parameters)
    ]
    
    for name, func in examples:
        print(f"\n{'*'*70}")
        print(f"Running: {name}")
        print(f"{'*'*70}")
        try:
            func()
        except Exception as e:
            print(f"Error in {name}: {str(e)}")
            continue
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED!")
    print("="*70)


if __name__ == "__main__":
    # Run individual examples or all at once
    
    # Uncomment the example you want to run:
    
    example_1_basic_usage()
    # example_2_compare_imputation()
    # example_3_compare_seasonal()
    # example_4_compare_anomaly_detection()
    # example_5_real_world_csv()
    # example_6_sensitivity_analysis()
    # example_7_batch_processing()
    # example_8_custom_parameters()
    
    # Or run all examples:
    # run_all_examples()
