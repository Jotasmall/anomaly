# Historical river levels anomaly detection

Colombian Water Level Anomaly Detection System
A comprehensive Python-based system for detecting anomalies in water level data from rivers and reservoirs in Colombia, designed to handle irregular timestamps, missing data, and bimodal seasonal patterns characteristic of tropical climates.

**Features**

*Robust Preprocessing:* Handles irregular timestamps and missing values through intelligent resampling and multiple imputation methods
*Seasonal Modeling:* Captures Colombia's bimodal rainfall pattern using STL decomposition and Prophet-style methods
*Multiple Detection Methods:* Three complementary anomaly detection approaches (Z-score, Isolation Forest, Autoencoder-based)
*Comprehensive Visualization:* Detailed plots for time series decomposition and anomaly identification
*Batch Processing:* Support for analyzing multiple monitoring stations simultaneously
*Export Capabilities:* Save results to CSV for further analysis

**Installation requirements**

Python 3.7 or higher

Required libraries:

pandas >= 1.3.0
numpy >= 1.21.0
scipy >= 1.7.0
scikit-learn >= 0.24.0
matplotlib >= 3.4.0 (for visualization)

**Setup**

Clone or download the repository

bash# Create a project directory
mkdir colombian_water_anomaly
cd colombian_water_anomaly

**Install dependencies**

bashpip install pandas numpy scipy scikit-learn matplotlib

**Download the main scripts**

water_anomaly_detector.py - Main detector class
example_usage.py - Example usage scripts
README.md - This file

**Quick Start**

Basic Usage is available in a PDF

**Contributing**
Contributions are welcome! Areas for improvement:

Integration with real-time data streams
Deep learning LSTM autoencoder implementation
Spatial correlation analysis for multi-station networks
Integration with climate indices (ONI, SOI)
Uncertainty quantification (confidence intervals)

**License**
MIT License - Free for academic and commercial use
Citation
If you use this system in your research, please cite:
Colombian Water Level Anomaly Detection System (2025)
Advanced time series analysis for hydrological monitoring in tropical regions
Contact
For questions, issues, or collaborations:

Open an issue on GitHub
Contact your data science team

**Acknowledgments**

Designed for Colombian hydrological monitoring systems
Inspired by STL decomposition (Cleveland et al., 1990)
Prophet-style seasonality modeling (Facebook Research, 2017)
Isolation Forest algorithm (Liu et al., 2008)


Version: 1.0
Last Updated: 2025
Status: Production-ready
