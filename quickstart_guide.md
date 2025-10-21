# Quick Start Guide

Get up and running with the Colombian Water Anomaly Detector in 5 minutes!

## Step 1: Installation (2 minutes)

### Download the files
You should have these files:
- `water_anomaly_detector.py` - Main detector class
- `example_usage.py` - Usage examples
- `test_suite.py` - Testing script
- `requirements.txt` - Package dependencies
- `README.md` - Full documentation
- `QUICKSTART.md` - This file

### Install dependencies

```bash
# Navigate to your project directory
cd colombian_water_anomaly

# Install required packages
pip install -r requirements.txt
```

**Or manually:**
```bash
pip install pandas numpy scipy scikit-learn matplotlib
```

## Step 2: Verify Installation (1 minute)

Run the test suite to ensure everything works:

```bash
python test_suite.py
```

You should see: `✅ System is operational!`

## Step 3: Run Your First Analysis (2 minutes)

### Option A: Use synthetic data (easiest)

```python
from water_anomaly_detector import ColombianWaterAnomalyDetector, generate_colombian_data

# Generate sample data
df = generate_colombian_data(n_days=365)

# Create and run detector
detector = ColombianWaterAnomalyDetector()
detector.load_data(df).preprocess().decompose().detect_anomalies()

# View results
summary = detector.get_anomaly_summary()
print(f"Detected {summary['total_anomalies']} anomalies")

# Visualize
detector.plot_results()
```

### Option B: Use your own CSV file

```python
from water_anomaly_detector import ColombianWaterAnomalyDetector
import pandas as pd

# Load your data (must have 'timestamp' and 'water_level' columns)
df = pd.read_csv('your_data.csv')

# Run analysis
detector = ColombianWaterAnomalyDetector()
detector.load_data(df).preprocess().decompose().detect_anomalies()

# Save results
results = detector.get_results()
results.to_csv('anomaly_results.csv', index=False)

# Visualize
detector.plot_results()
```

### Your CSV format should look like this:
```csv
timestamp,water_level
2023-01-01,45.3
2023-01-02,46.1
2023-01-03,
2023-01-05,47.8
```

## Step 4: Explore Examples

Run the included examples to see different features:

```bash
# Basic usage
python example_usage.py

# Or run specific examples in Python:
python -c "from example_usage import example_4_compare_anomaly_detection; example_4_compare_anomaly_detection()"
```

## Common Use Cases

### Scenario 1: Quick flood/drought detection
```python
detector = ColombianWaterAnomalyDetector(
    imputation_method='linear',
    seasonal_method='prophet',
    anomaly_method='isolation_forest'
)
```
**Best for:** General purpose, robust to outliers

### Scenario 2: Sensitive detection for early warning
```python
detector = ColombianWaterAnomalyDetector(
    imputation_method='seasonal',
    seasonal_method='prophet',
    anomaly_method='zscore',
    zscore_threshold=2.5  # More sensitive
)
```
**Best for:** Catching small anomalies early

### Scenario 3: High-quality data with few gaps
```python
detector = ColombianWaterAnomalyDetector(
    imputation_method='linear',
    seasonal_method='stl',
    anomaly_method='zscore'
)
```
**Best for:** Fast processing, clean data

### Scenario 4: Lots of missing data
```python
detector = ColombianWaterAnomalyDetector(
    imputation_method='seasonal',
    seasonal_method='prophet',
    anomaly_method='autoencoder'
)
```
**Best for:** Noisy data with systematic gaps

## Understanding Results

### Output columns explained:

- **timestamp**: Date/time of measurement
- **water_level**: Actual water level (meters)
- **imputed**: True if value was filled in (not original)
- **trend**: Long-term trend component
- **seasonal**: Seasonal variation (wet/dry seasons)
- **residual**: Unexpected deviation from trend+seasonal
- **is_anomaly**: True if flagged as anomaly
- **anomaly_score**: Strength of anomaly (higher = stronger)
- **anomaly_type**: 'flood', 'drought', or 'normal'

### Interpreting anomaly_type:

🔵 **Flood** - Water level much higher than expected
- Causes: Heavy rainfall, La Niña, dam release
- Action: Check flood risk areas

🟠 **Drought** - Water level much lower than expected
- Causes: El Niño, extended dry season, over-extraction
- Action: Check water supply concerns

## Troubleshooting

### Problem: "Module not found" error
**Solution:** Install missing package
```bash
pip install <package_name>
```

### Problem: Too many/few anomalies detected
**Solution:** Adjust sensitivity
```python
# For MORE anomalies (more sensitive):
detector = ColombianWaterAnomalyDetector(zscore_threshold=2.5)

# For FEWER anomalies (less sensitive):
detector = ColombianWaterAnomalyDetector(zscore_threshold=3.5)
```

### Problem: Poor seasonal pattern fit
**Solution:** Try Prophet method
```python
detector = ColombianWaterAnomalyDetector(seasonal_method='prophet')
```

### Problem: Data has many gaps
**Solution:** Use seasonal imputation
```python
detector = ColombianWaterAnomalyDetector(imputation_method='seasonal')
```

## Next Steps

1. **Read the full README.md** for detailed documentation
2. **Explore example_usage.py** for 8 comprehensive examples
3. **Customize parameters** for your specific river/reservoir
4. **Integrate with your monitoring system** for automated detection

## Getting Help

- Check **README.md** for comprehensive documentation
- Run **test_suite.py --full** for diagnostic information
- Review **example_usage.py** for practical examples
- Examine error messages carefully - they're designed to be helpful!

## Tips for Best Results

✅ **DO:**
- Use at least 1 year of data for reliable seasonal patterns
- Ensure timestamps are in standard format (YYYY-MM-DD)
- Validate detected anomalies with known events
- Compare multiple detection methods
- Export results to CSV for further analysis

❌ **DON'T:**
- Use less than 6 months of data
- Ignore the data coverage warnings
- Assume all detected anomalies are real events
- Use without understanding your regional climate
- Forget to check for sensor errors

## Colombian Climate Context

Remember that Colombia has **bimodal rainfall**:
- **Wet seasons**: April-May, October-November
- **Dry seasons**: December-March, June-September

The algorithm accounts for these patterns automatically, so anomalies are detected as *deviations from the expected seasonal pattern*, not from an overall average.

---

**Ready to analyze?** Start with the synthetic data example above, then move to your own data!

**Questions?** Check the README.md for detailed answers.

**Found a bug?** Run test_suite.py --full to diagnose.