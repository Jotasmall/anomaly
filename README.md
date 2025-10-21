# Historical river levels anomaly detection

Colombian Water Level Anomaly Detection System
A comprehensive Python-based system for detecting anomalies in water level data from rivers and reservoirs in Colombia, designed to handle irregular timestamps, missing data, and bimodal seasonal patterns characteristic of tropical climates.
Features

Robust Preprocessing: Handles irregular timestamps and missing values through intelligent resampling and multiple imputation methods
Seasonal Modeling: Captures Colombia's bimodal rainfall pattern using STL decomposition and Prophet-style methods
Multiple Detection Methods: Three complementary anomaly detection approaches (Z-score, Isolation Forest, Autoencoder-based)
Comprehensive Visualization: Detailed plots for time series decomposition and anomaly identification
Batch Processing: Support for analyzing multiple monitoring stations simultaneously
Export Capabilities: Save results to CSV for further analysis

Installation
Requirements

Python 3.7 or higher
Required libraries:

pandas >= 1.3.0
numpy >= 1.21.0
scipy >= 1.7.0
scikit-learn >= 0.24.0
matplotlib >= 3.4.0 (for visualization)



Setup

Clone or download the repository

bash# Create a project directory
mkdir colombian_water_anomaly
cd colombian_water_anomaly

Install dependencies

bashpip install pandas numpy scipy scikit-learn matplotlib

Download the main scripts

water_anomaly_detector.py - Main detector class
example_usage.py - Example usage scripts
README.md - This file



Quick Start
Basic Usage
pythonfrom water_anomaly_detector import ColombianWaterAnomalyDetector
import pandas as pd

# Load your data (must have 'timestamp' and 'water_level' columns)
df = pd.read_csv('your_water_data.csv')

# Initialize detector
detector = ColombianWaterAnomalyDetector(
    imputation_method='linear',
    seasonal_method='prophet',
    anomaly_method='isolation_forest'
)

# Run the detection pipeline
detector.load_data(df) \
       .preprocess() \
       .decompose() \
       .detect_anomalies()

# Get results
results = detector.get_results()
summary = detector.get_anomaly_summary()

# Visualize
detector.plot_results()

# Save results
results.to_csv('anomaly_results.csv', index=False)
Input Data Format
Your CSV file should have the following structure:
csvtimestamp,water_level
2023-01-01,45.3
2023-01-02,46.1
2023-01-03,
2023-01-05,47.8
...
Required columns:

timestamp: Date/time in any standard format (YYYY-MM-DD, YYYY-MM-DD HH:MM:SS, etc.)
water_level: Water level measurements (float values, NaN or empty for missing data)

Notes:

Timestamps can be irregular (missing dates are handled automatically)
Missing water level values are imputed during preprocessing
At least 60% data coverage recommended for reliable results

Configuration Parameters
Imputation Methods
pythonimputation_method='linear'  # Options: 'linear', 'forward', 'seasonal'

linear: Linear interpolation between valid points (best for short gaps)
forward: Forward fill from last valid value (simple, fast)
seasonal: Uses seasonal averages (best for systematic gaps)

Seasonal Decomposition
pythonseasonal_method='stl'  # Options: 'stl', 'prophet'

stl: STL decomposition with Loess smoothing (classical, robust)
prophet: Fourier-based seasonality modeling (better for bimodal patterns)

Anomaly Detection
pythonanomaly_method='zscore'  # Options: 'zscore', 'isolation_forest', 'autoencoder'

zscore: Statistical threshold-based (fast, interpretable, threshold-tunable)
isolation_forest: Ensemble method using multiple features (robust to outliers)
autoencoder: Reconstruction error-based (captures complex patterns)

Additional Parameters
pythonzscore_threshold=3.0      # Z-score sensitivity (lower = more sensitive)
seasonal_period=365       # Days in seasonal cycle
trend_window=30           # Days for trend smoothing
Examples
The package includes comprehensive examples demonstrating various use cases:
Example 1: Basic Usage
bashpython example_usage.py
Runs a complete analysis with default settings on synthetic data.
Example 2: Compare Imputation Methods
pythonfrom example_usage import example_2_compare_imputation
example_2_compare_imputation()
Compares linear, forward, and seasonal imputation side-by-side.
Example 3: Compare Decomposition Methods
pythonfrom example_usage import example_3_compare_seasonal
example_3_compare_seasonal()
Compares STL vs Prophet for seasonal pattern extraction.
Example 4: Compare Detection Methods
pythonfrom example_usage import example_4_compare_anomaly_detection
example_4_compare_anomaly_detection()
Compares Z-score, Isolation Forest, and Autoencoder methods.
Example 5: Load Real Data from CSV
pythonfrom example_usage import example_5_real_world_csv
example_5_real_world_csv()
Demonstrates loading external CSV files and exporting results.
Example 6: Sensitivity Analysis
pythonfrom example_usage import example_6_sensitivity_analysis
example_6_sensitivity_analysis()
Tests how detection sensitivity changes with threshold parameters.
Example 7: Batch Processing
pythonfrom example_usage import example_7_batch_processing
example_7_batch_processing()
Processes multiple monitoring stations simultaneously.
Example 8: Custom Parameters
pythonfrom example_usage import example_8_custom_parameters
example_8_custom_parameters()
Advanced configuration for specific scenarios (e.g., Andean reservoirs).
Output Files
The system generates several output files:

anomaly_results.csv: Complete time series with all components and flags

Columns: timestamp, water_level, imputed, trend, seasonal, residual, is_anomaly, anomaly_score, anomaly_type


detected_anomalies.csv: Only rows with detected anomalies

Useful for incident reporting


Visualization plots: PNG files showing:

Time series with marked anomalies
Decomposition components (trend, seasonal, residual)
Comparison plots for different methods



Colombian Hydrological Context
Seasonal Patterns
Colombia experiences bimodal rainfall due to the Intertropical Convergence Zone (ITCZ):

First wet season: March-May (peak in April)
First dry season: June-September
Second wet season: October-November
Second dry season: December-February

Regional Variations

Andean region: Strong bimodal pattern, elevation effects
Pacific coast: Highest rainfall, less defined dry season
Amazon basin: More uniform rainfall, slight seasonality
Caribbean coast: Single wet season (May-November)

Climate Phenomena

El Niño: Causes droughts (below-average water levels)
La Niña: Causes floods (above-average water levels)

The algorithm is designed to detect extreme events (floods/droughts) while accounting for these natural seasonal variations.
Interpreting Results
Anomaly Types

Flood Events (positive residuals):

Sudden spikes in water level
Potential causes: Heavy rainfall, La Niña, dam releases
Marked in red on plots


Drought Events (negative residuals):

Sustained drops in water level
Potential causes: El Niño, extended dry season, over-extraction
Marked in orange on plots



Anomaly Scores

Higher scores indicate stronger deviations from normal patterns
Scores are method-specific but normalized for comparison
Typical ranges:

Z-score: 3-10 (threshold at 3)
Isolation Forest: 0-5 (threshold ~2.5)
Autoencoder: 0-8 (threshold ~2.5)



Quality Indicators

Data coverage: >60% recommended, >80% ideal
Imputed values: Check imputed column in results
Seasonal fit: Visual inspection of seasonal component

Troubleshooting
Common Issues

"Data coverage < 60%" warning

Solution: Use 'seasonal' imputation method or collect more data


Too many/few anomalies detected

Solution: Adjust zscore_threshold (higher = fewer anomalies)
Try different detection methods


Irregular seasonal pattern

Solution: Use 'prophet' seasonal method
Check if seasonal_period should be adjusted


Poor trend estimation

Solution: Increase trend_window parameter
Check for outliers affecting trend calculation


Missing timestamps

Solution: System automatically handles this through resampling
Ensure timestamps are in standard format



Performance Considerations

Hourly data: Can be memory-intensive for >2 years

Consider weekly aggregation for long-term trends


Daily data: Optimal for most use cases

Recommended for 1-5 years of analysis


Weekly data: Good for multi-decade analysis

May miss short-duration flood events



Advanced Usage
Custom Regional Patterns
For specific Colombian regions, adjust parameters:
python# Pacific Coast (high rainfall, less defined seasons)
detector = ColombianWaterAnomalyDetector(
    seasonal_method='prophet',
    trend_window=45,  # Longer smoothing
    seasonal_period=365
)

# Andean Highlands (strong bimodal, elevation effects)
detector = ColombianWaterAnomalyDetector(
    seasonal_method='prophet',
    anomaly_method='isolation_forest',
    trend_window=30
)

# Amazon Basin (subtle seasonality)
detector = ColombianWaterAnomalyDetector(
    seasonal_method='stl',
    zscore_threshold=2.5,  # More sensitive
    trend_window=60
)
Integrating Climate Indices
For advanced users, consider incorporating ENSO indices:
python# Download ONI (Oceanic Niño Index) data
# Adjust seasonal expectations based on El Niño/La Niña phase
# Strong El Niño → expect lower water levels
# La Niña → expect higher water levels
Multi-Station Network Analysis
pythonfrom example_usage import example_7_batch_processing

# Process entire monitoring network
stations = ['Station_A', 'Station_B', 'Station_C']
# Cross-validate anomalies across stations
# If anomaly appears at multiple stations → likely real event
# If isolated → potential sensor error
Contributing
Contributions are welcome! Areas for improvement:

Integration with real-time data streams
Deep learning LSTM autoencoder implementation
Spatial correlation analysis for multi-station networks
Integration with climate indices (ONI, SOI)
Uncertainty quantification (confidence intervals)

License
MIT License - Free for academic and commercial use
Citation
If you use this system in your research, please cite:
Colombian Water Level Anomaly Detection System (2025)
Advanced time series analysis for hydrological monitoring in tropical regions
Contact
For questions, issues, or collaborations:

Open an issue on GitHub
Contact your data science team

Acknowledgments

Designed for Colombian hydrological monitoring systems
Inspired by STL decomposition (Cleveland et al., 1990)
Prophet-style seasonality modeling (Facebook Research, 2017)
Isolation Forest algorithm (Liu et al., 2008)


Version: 1.0
Last Updated: 2025
Status: Production-ready
