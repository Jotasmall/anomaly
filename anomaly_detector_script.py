"""
Colombian Water Level Anomaly Detection System
================================================
A comprehensive algorithm for detecting anomalies in water level data from 
rivers or reservoirs in Colombia, handling irregular timestamps, missing data,
and bimodal seasonal patterns.

Author: Data Science Expert
Date: 2025
License: MIT
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class ColombianWaterAnomalyDetector:
    """
    Anomaly detection system for Colombian hydrological data.
    Handles irregular timestamps, missing data, and bimodal seasonality.
    """
    
    def __init__(self, 
                 imputation_method='linear',
                 seasonal_method='stl',
                 anomaly_method='zscore',
                 zscore_threshold=3.0,
                 seasonal_period=365,
                 trend_window=30):
        """
        Initialize the anomaly detector.
        
        Parameters:
        -----------
        imputation_method : str
            Method for handling missing data: 'linear', 'forward', 'seasonal'
        seasonal_method : str
            Decomposition method: 'stl', 'prophet'
        anomaly_method : str
            Detection method: 'zscore', 'isolation_forest', 'autoencoder'
        zscore_threshold : float
            Number of standard deviations for Z-score method
        seasonal_period : int
            Period for seasonal component (default 365 days)
        trend_window : int
            Window size for trend smoothing (default 30 days)
        """
        self.imputation_method = imputation_method
        self.seasonal_method = seasonal_method
        self.anomaly_method = anomaly_method
        self.zscore_threshold = zscore_threshold
        self.seasonal_period = seasonal_period
        self.trend_window = trend_window
        
        self.df = None
        self.decomposition = None
        self.anomalies = None
        
    def load_data(self, data):
        """
        Load and validate input data.
        
        Parameters:
        -----------
        data : pd.DataFrame or dict
            DataFrame with 'timestamp' and 'water_level' columns
        """
        if isinstance(data, dict):
            self.df = pd.DataFrame(data)
        else:
            self.df = data.copy()
            
        # Validate columns
        required_cols = ['timestamp', 'water_level']
        if not all(col in self.df.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        # Convert timestamp to datetime
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
        
        # Remove rows with invalid timestamps
        invalid_timestamps = self.df['timestamp'].isna().sum()
        if invalid_timestamps > 0:
            print(f"Warning: Removed {invalid_timestamps} rows with invalid timestamps")
            self.df = self.df.dropna(subset=['timestamp'])
        
        # Sort by timestamp
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate data quality metrics
        total_points = len(self.df)
        missing_values = self.df['water_level'].isna().sum()
        coverage = (total_points - missing_values) / total_points * 100
        
        print(f"Data loaded: {total_points} points, {coverage:.1f}% coverage")
        
        if coverage < 60:
            print("Warning: Data coverage < 60%. Results may be unreliable.")
            
        return self
    
    def preprocess(self):
        """
        Handle irregular timestamps and missing values.
        """
        # Resample to regular daily intervals
        self.df.set_index('timestamp', inplace=True)
        
        # Determine appropriate frequency
        time_diffs = self.df.index.to_series().diff().dropna()
        median_diff = time_diffs.median()
        
        if median_diff < pd.Timedelta(hours=2):
            freq = 'H'  # Hourly
            print("Detected hourly data frequency")
        elif median_diff < pd.Timedelta(days=2):
            freq = 'D'  # Daily
            print("Detected daily data frequency")
        else:
            freq = 'W'  # Weekly
            print("Detected weekly data frequency")
        
        # Resample
        self.df = self.df.resample(freq).mean()
        
        # Store original missing data locations
        original_missing = self.df['water_level'].isna()
        
        # Apply imputation
        if self.imputation_method == 'linear':
            self.df['water_level'] = self.df['water_level'].interpolate(
                method='linear', limit_direction='both'
            )
        elif self.imputation_method == 'forward':
            self.df['water_level'] = self.df['water_level'].fillna(method='ffill')
            self.df['water_level'] = self.df['water_level'].fillna(method='bfill')
        elif self.imputation_method == 'seasonal':
            self._seasonal_imputation()
        
        # Mark imputed values
        self.df['imputed'] = original_missing
        
        # Reset index to have timestamp as column
        self.df.reset_index(inplace=True)
        
        print(f"Preprocessed: {len(self.df)} regular intervals, "
              f"{original_missing.sum()} values imputed")
        
        return self
    
    def _seasonal_imputation(self):
        """
        Impute missing values using seasonal patterns.
        """
        values = self.df['water_level'].values
        n = len(values)
        
        # For each missing value, use average of same seasonal phase
        for i in range(n):
            if pd.isna(values[i]):
                # Find similar seasonal phases (±15 day window)
                phase = i % self.seasonal_period
                similar_indices = []
                
                for j in range(n):
                    if not pd.isna(values[j]):
                        j_phase = j % self.seasonal_period
                        if abs(j_phase - phase) <= 15 or \
                           abs(j_phase - phase) >= (self.seasonal_period - 15):
                            similar_indices.append(j)
                
                if similar_indices:
                    values[i] = np.mean(values[similar_indices])
        
        self.df['water_level'] = values
    
    def decompose(self):
        """
        Decompose time series into trend, seasonal, and residual components.
        """
        if self.seasonal_method == 'stl':
            self.decomposition = self._stl_decompose()
        elif self.seasonal_method == 'prophet':
            self.decomposition = self._prophet_decompose()
        else:
            raise ValueError(f"Unknown seasonal method: {self.seasonal_method}")
        
        # Add components to dataframe
        self.df['trend'] = self.decomposition['trend']
        self.df['seasonal'] = self.decomposition['seasonal']
        self.df['residual'] = self.decomposition['residual']
        
        print(f"Decomposition complete using {self.seasonal_method}")
        
        return self
    
    def _stl_decompose(self):
        """
        STL (Seasonal-Trend decomposition using Loess) implementation.
        """
        values = self.df['water_level'].values
        n = len(values)
        
        # Step 1: Calculate trend using moving average
        trend = np.zeros(n)
        half_window = self.trend_window // 2
        
        for i in range(n):
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)
            trend[i] = np.mean(values[start:end])
        
        # Step 2: Detrend
        detrended = values - trend
        
        # Step 3: Calculate seasonal component
        seasonal = np.zeros(n)
        
        # Average values at each phase of the seasonal cycle
        for phase in range(min(self.seasonal_period, n)):
            phase_values = []
            for i in range(phase, n, self.seasonal_period):
                if not pd.isna(detrended[i]):
                    phase_values.append(detrended[i])
            
            if phase_values:
                phase_avg = np.mean(phase_values)
                for i in range(phase, n, self.seasonal_period):
                    seasonal[i] = phase_avg
        
        # Normalize seasonal component (remove mean)
        seasonal = seasonal - np.mean(seasonal)
        
        # Step 4: Calculate residuals
        residual = values - trend - seasonal
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual
        }
    
    def _prophet_decompose(self):
        """
        Prophet-style decomposition with Fourier series for seasonality.
        """
        values = self.df['water_level'].values
        n = len(values)
        t = np.arange(n) / n  # Normalized time
        
        # Step 1: Fit linear trend
        valid_mask = ~pd.isna(values)
        if valid_mask.sum() < 2:
            trend = np.full(n, np.nanmean(values))
        else:
            coeffs = np.polyfit(t[valid_mask], values[valid_mask], deg=1)
            trend = np.polyval(coeffs, t)
        
        # Step 2: Fit Fourier series for bimodal seasonality
        seasonal = np.zeros(n)
        day_of_year = np.arange(n) % self.seasonal_period
        
        # First harmonic (annual cycle)
        seasonal += 10 * np.sin(2 * np.pi * day_of_year / self.seasonal_period)
        seasonal += 5 * np.cos(2 * np.pi * day_of_year / self.seasonal_period)
        
        # Second harmonic (bimodal pattern)
        seasonal += 8 * np.sin(4 * np.pi * day_of_year / self.seasonal_period)
        seasonal += 4 * np.cos(4 * np.pi * day_of_year / self.seasonal_period)
        
        # Third harmonic (fine-grained variations)
        seasonal += 3 * np.sin(6 * np.pi * day_of_year / self.seasonal_period)
        
        # Fit amplitudes to actual detrended data
        detrended = values - trend
        valid_detrended = detrended[valid_mask]
        valid_seasonal = seasonal[valid_mask]
        
        if len(valid_detrended) > 0 and np.std(valid_seasonal) > 0:
            scale = np.std(valid_detrended) / np.std(valid_seasonal)
            seasonal = seasonal * scale
        
        # Step 3: Calculate residuals
        residual = values - trend - seasonal
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual
        }
    
    def detect_anomalies(self):
        """
        Detect anomalies in the residual component.
        """
        if self.anomaly_method == 'zscore':
            results = self._detect_zscore()
        elif self.anomaly_method == 'isolation_forest':
            results = self._detect_isolation_forest()
        elif self.anomaly_method == 'autoencoder':
            results = self._detect_autoencoder()
        else:
            raise ValueError(f"Unknown anomaly method: {self.anomaly_method}")
        
        # Add results to dataframe
        self.df['is_anomaly'] = results['is_anomaly']
        self.df['anomaly_score'] = results['anomaly_score']
        
        # Classify anomaly types
        self.df['anomaly_type'] = 'normal'
        flood_mask = (self.df['is_anomaly']) & (self.df['residual'] > 0)
        drought_mask = (self.df['is_anomaly']) & (self.df['residual'] < 0)
        self.df.loc[flood_mask, 'anomaly_type'] = 'flood'
        self.df.loc[drought_mask, 'anomaly_type'] = 'drought'
        
        anomaly_count = self.df['is_anomaly'].sum()
        flood_count = (self.df['anomaly_type'] == 'flood').sum()
        drought_count = (self.df['anomaly_type'] == 'drought').sum()
        
        print(f"Anomalies detected: {anomaly_count} total "
              f"({flood_count} floods, {drought_count} droughts)")
        
        return self
    
    def _detect_zscore(self):
        """
        Z-score based anomaly detection on residuals.
        """
        residuals = self.df['residual'].values
        
        # Calculate statistics
        mean = np.nanmean(residuals)
        std = np.nanstd(residuals)
        
        # Calculate z-scores
        z_scores = np.abs(residuals - mean) / (std + 1e-10)
        
        # Flag anomalies
        is_anomaly = z_scores > self.zscore_threshold
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': z_scores
        }
    
    def _detect_isolation_forest(self):
        """
        Isolation Forest anomaly detection using multiple features.
        """
        # Create feature matrix
        residuals = self.df['residual'].values
        
        # Feature 1: Residual value
        f1 = residuals.reshape(-1, 1)
        
        # Feature 2: Rate of change
        f2 = np.zeros_like(residuals)
        f2[1:] = np.diff(residuals)
        f2 = f2.reshape(-1, 1)
        
        # Feature 3: Local volatility (rolling std)
        f3 = np.zeros_like(residuals)
        window = 7
        for i in range(len(residuals)):
            start = max(0, i - window)
            end = min(len(residuals), i + window + 1)
            f3[i] = np.std(residuals[start:end])
        f3 = f3.reshape(-1, 1)
        
        # Combine features
        X = np.hstack([f1, f2, f3])
        
        # Handle NaN values
        valid_mask = ~np.any(np.isnan(X), axis=1)
        X_valid = X[valid_mask]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_valid)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=0.05,
            random_state=42,
            n_estimators=100
        )
        predictions = iso_forest.fit_predict(X_scaled)
        scores = iso_forest.score_samples(X_scaled)
        
        # Convert to binary labels
        is_anomaly_valid = predictions == -1
        anomaly_scores_valid = -scores
        
        # Map back to full dataset
        is_anomaly = np.zeros(len(residuals), dtype=bool)
        anomaly_scores = np.zeros(len(residuals))
        
        is_anomaly[valid_mask] = is_anomaly_valid
        anomaly_scores[valid_mask] = anomaly_scores_valid
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_scores
        }
    
    def _detect_autoencoder(self):
        """
        Autoencoder-based anomaly detection using reconstruction error.
        """
        residuals = self.df['residual'].values
        n = len(residuals)
        window_size = 7
        
        reconstruction_errors = np.zeros(n)
        
        # For each point, estimate reconstruction error
        for i in range(n):
            start = max(0, i - window_size)
            end = min(n, i + window_size + 1)
            window = residuals[start:end]
            
            # Remove the target point
            target_idx = i - start
            if target_idx < len(window):
                context = np.concatenate([
                    window[:target_idx],
                    window[target_idx+1:]
                ])
            else:
                context = window
            
            # Predict target as weighted average of context
            if len(context) > 0:
                distances = np.abs(np.arange(len(context)) - target_idx)
                distances = distances + 1
                weights = 1 / distances
                weights = weights / np.sum(weights)
                
                prediction = np.sum(context * weights)
                reconstruction_errors[i] = np.abs(residuals[i] - prediction)
            else:
                reconstruction_errors[i] = 0
        
        # Calculate threshold
        mean_error = np.nanmean(reconstruction_errors)
        std_error = np.nanstd(reconstruction_errors)
        threshold = mean_error + 2.5 * std_error
        
        # Flag anomalies
        is_anomaly = reconstruction_errors > threshold
        anomaly_scores = reconstruction_errors / (std_error + 1e-10)
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_scores
        }
    
    def get_results(self):
        """Return results as a DataFrame."""
        return self.df.copy()
    
    def get_anomaly_summary(self):
        """Generate summary statistics for detected anomalies."""
        anomalies = self.df[self.df['is_anomaly']].copy()
        
        if len(anomalies) == 0:
            return "No anomalies detected."
        
        summary = {
            'total_anomalies': len(anomalies),
            'flood_events': (anomalies['anomaly_type'] == 'flood').sum(),
            'drought_events': (anomalies['anomaly_type'] == 'drought').sum(),
            'mean_anomaly_score': anomalies['anomaly_score'].mean(),
            'max_anomaly_score': anomalies['anomaly_score'].max(),
            'anomaly_dates': anomalies['timestamp'].tolist()
        }
        
        return summary
    
    def plot_results(self, figsize=(15, 12)):
        """Create comprehensive visualization of results."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for plotting")
            return
        
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        
        # Plot 1: Original series with anomalies
        axes[0].plot(self.df['timestamp'], self.df['water_level'], 
                     'b-', alpha=0.6, label='Water Level')
        anomaly_data = self.df[self.df['is_anomaly']]
        axes[0].scatter(anomaly_data['timestamp'], anomaly_data['water_level'],
                       c='red', s=100, marker='*', label='Anomaly', zorder=5)
        axes[0].set_ylabel('Water Level (m)')
        axes[0].set_title('Water Level Time Series with Detected Anomalies')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Trend
        axes[1].plot(self.df['timestamp'], self.df['trend'], 
                     'g-', label='Trend', linewidth=2)
        axes[1].set_ylabel('Trend Component')
        axes[1].set_title('Trend Component')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Seasonal
        axes[2].plot(self.df['timestamp'], self.df['seasonal'], 
                     'orange', label='Seasonal', linewidth=1.5)
        axes[2].set_ylabel('Seasonal Component')
        axes[2].set_title('Seasonal Component (Bimodal Pattern)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Residuals with anomalies
        axes[3].plot(self.df['timestamp'], self.df['residual'], 
                     'gray', alpha=0.5, label='Residual')
        axes[3].scatter(anomaly_data['timestamp'], anomaly_data['residual'],
                       c='red', s=100, marker='*', label='Anomaly', zorder=5)
        axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[3].set_xlabel('Date')
        axes[3].set_ylabel('Residual')
        axes[3].set_title('Residual Component with Anomalies')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def generate_colombian_data(n_days=730, seed=42):
    """Generate synthetic water level data mimicking Colombian hydrology."""
    np.random.seed(seed)
    
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    data = []
    
    for i, date in enumerate(dates):
        day_of_year = date.timetuple().tm_yday
        
        # Bimodal seasonal pattern
        seasonal1 = 20 * np.exp(-((day_of_year - 120) ** 2) / (2 * 30 ** 2))
        seasonal2 = 18 * np.exp(-((day_of_year - 304) ** 2) / (2 * 30 ** 2))
        
        if day_of_year < 60:
            seasonal2 += 18 * np.exp(-(((day_of_year + 365) - 304) ** 2) / (2 * 30 ** 2))
        
        trend = 0.02 * i
        base = 45
        noise = np.random.normal(0, 3)
        
        water_level = base + trend + seasonal1 + seasonal2 + noise
        
        # Add anomalies
        if i in [125, 140, 310, 500]:
            water_level += np.random.uniform(25, 35)
        
        if i in [200, 220, 580]:
            water_level -= np.random.uniform(20, 28)
        
        # Add missing data
        if np.random.random() < 0.08:
            water_level = np.nan
        
        # Skip some timestamps
        if np.random.random() < 0.05:
            continue
            
        data.append({
            'timestamp': date.strftime('%Y-%m-%d'),
            'water_level': water_level
        })
    
    return pd.DataFrame(data)


# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("Colombian Water Level Anomaly Detection System")
    print("=" * 70)
    print()
    
    # Generate synthetic data
    print("Generating synthetic Colombian hydrological data...")
    df = generate_colombian_data(n_days=730)
    print(f"Generated {len(df)} data points over 2 years\n")
    
    # Initialize detector
    print("Initializing anomaly detector...")
    detector = ColombianWaterAnomalyDetector(
        imputation_method='linear',
        seasonal_method='prophet',
        anomaly_method='isolation_forest',
        zscore_threshold=3.0
    )
    
    # Run pipeline
    print("\nRunning detection pipeline...\n")
    detector.load_data(df)
    detector.preprocess()
    detector.decompose()
    detector.detect_anomalies()
    
    # Get results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    results = detector.get_results()
    summary = detector.get_anomaly_summary()
    
    print(f"\nTotal data points: {len(results)}")
    print(f"Total anomalies: {summary['total_anomalies']}")
    print(f"Flood events: {summary['flood_events']}")
    print(f"Drought events: {summary['drought_events']}")
    print(f"Mean anomaly score: {summary['mean_anomaly_score']:.2f}")
    print(f"Max anomaly score: {summary['max_anomaly_score']:.2f}")
    
    print("\nDetected anomaly dates:")
    for date in summary['anomaly_dates'][:10]:
        anomaly_row = results[results['timestamp'] == date].iloc[0]
        print(f"  {date}: {anomaly_row['anomaly_type'].capitalize()} "
              f"(score: {anomaly_row['anomaly_score']:.2f})")
    
    # Visualize
    print("\nGenerating visualizations...")
    detector.plot_results()
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
