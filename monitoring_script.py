import pandas as pd
import numpy as np
from scipy import stats
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_basic_drift_stats(reference_data, current_data):
    """Calculate basic drift statistics using scipy"""
    drift_results = {}

    # Get numeric columns only
    numeric_cols = reference_data.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col in current_data.columns:
            # Kolmogorov-Smirnov test for distribution drift
            ks_stat, p_value = stats.ks_2samp(reference_data[col], current_data[col])

            drift_results[col] = {
                'ks_statistic': float(ks_stat),
                'p_value': float(p_value),
                'drift_detected': bool(p_value < 0.05),  # Convert to Python bool
                'reference_mean': float(reference_data[col].mean()),
                'current_mean': float(current_data[col].mean()),
                'mean_difference': float(current_data[col].mean() - reference_data[col].mean())
            }

    return drift_results

def generate_html_report(drift_results):
    """Generate HTML drift report"""
    total_features = len(drift_results)
    drift_count = sum(1 for result in drift_results.values() if result['drift_detected'])
    drift_percentage = (drift_count / total_features * 100) if total_features > 0 else 0

    feature_rows = ""
    for feature, stats in drift_results.items():
        status = "DRIFT DETECTED" if stats['drift_detected'] else "NO DRIFT"
        bg_color = "#ffe6e6" if stats['drift_detected'] else "#e6ffe6"
        icon = "🚨" if stats['drift_detected'] else "✅"

        feature_rows += f"""
    <tr style="background-color: {bg_color};">
        <td>{icon} {feature}</td>
        <td>{status}</td>
        <td>{stats['ks_statistic']:.4f}</td>
        <td>{stats['p_value']:.4f}</td>
        <td>{stats['reference_mean']:.4f}</td>
        <td>{stats['current_mean']:.4f}</td>
        <td>{stats['mean_difference']:+.4f}</td>
    </tr>"""

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Data Drift Monitoring Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .summary {{ background-color: #e6f3ff; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; text-align: left; border: 1px solid #ddd; }}
        th {{ background-color: #4CAF50; color: white; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Data Drift Monitoring Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>

    <div class="summary">
        <h2>📊 Summary</h2>
        <p><strong>Total Features Analyzed:</strong> {total_features}</p>
        <p><strong>Features with Drift Detected:</strong> {drift_count}</p>
        <p><strong>Drift Detection Rate:</strong> {drift_percentage:.1f}%</p>
    </div>

    <h2>📈 Feature Analysis</h2>
    <table>
        <tr>
            <th>Feature</th>
            <th>Drift Status</th>
            <th>KS Statistic</th>
            <th>P-value</th>
            <th>Reference Mean</th>
            <th>Current Mean</th>
            <th>Mean Change</th>
        </tr>
        {feature_rows}
    </table>

    <div style="margin-top: 40px; color: #666;">
        <h3>📋 Methodology</h3>
        <p>This report uses the Kolmogorov-Smirnov test to detect distribution drift.</p>
        <p><strong>Drift Detection:</strong> p-value < 0.05 indicates significant change</p>
    </div>
</body>
</html>"""
    return html_content

def main():
    """Main monitoring function"""
    print("🚀 Starting Data Drift Monitoring...")

    try:
        print("📊 Loading reference data (January)...")
        reference_data = pd.read_parquet('data/raw/green_tripdata_2023-01.parquet')

        print("📊 Loading current data (February)...")
        current_data = pd.read_parquet('data/raw/green_tripdata_2023-02.parquet')

        print("🔄 Preprocessing data...")
        def preprocess(df):
            df = df.copy()  # Create a copy to avoid warnings
            df['duration'] = (df.lpep_dropoff_datetime - df.lpep_pickup_datetime).dt.total_seconds() / 60
            df = df[(df.duration >= 1) & (df.duration <= 60)]
            numeric_features = ['trip_distance', 'duration', 'fare_amount', 'tip_amount', 'total_amount']
            available_features = [col for col in numeric_features if col in df.columns]
            return df[available_features].dropna()

        ref_processed = preprocess(reference_data)
        cur_processed = preprocess(current_data)

        print("🔍 Calculating drift statistics...")
        drift_results = calculate_basic_drift_stats(ref_processed, cur_processed)

        print("📝 Generating reports...")
        html_report = generate_html_report(drift_results)

        with open('drift_report.html', 'w', encoding='utf-8') as f:
            f.write(html_report)

        # Convert any remaining numpy types to Python types for JSON serialization
        json_safe_results = {}
        for feature, stats in drift_results.items():
            json_safe_results[feature] = {
                'ks_statistic': float(stats['ks_statistic']),
                'p_value': float(stats['p_value']),
                'drift_detected': bool(stats['drift_detected']),
                'reference_mean': float(stats['reference_mean']),
                'current_mean': float(stats['current_mean']),
                'mean_difference': float(stats['mean_difference'])
            }

        with open('drift_results.json', 'w') as f:
            json.dump(json_safe_results, f, indent=2)

        print("✅ SUCCESS: Reports generated!")
        print("📄 HTML Report: drift_report.html")
        print("📄 JSON Results: drift_results.json")

        drift_count = sum(1 for result in drift_results.values() if result['drift_detected'])
        print(f"\n📊 SUMMARY: {drift_count}/{len(drift_results)} features show drift")

        if drift_count > 0:
            print("🚨 Features with drift detected:")
            for feature, stats in drift_results.items():
                if stats['drift_detected']:
                    print(f"   - {feature}: p-value = {stats['p_value']:.4f}")
        else:
            print("✅ No significant drift detected in any features")

    except FileNotFoundError as e:
        print(f"❌ ERROR: Data file not found - {e}")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
