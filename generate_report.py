# generate_report.py

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

print("--- Running Inside a Clean Docker Container ---")

try:
    print("Loading reference and current data...")
    reference_data = pd.read_parquet('data/raw/green_tripdata_2023-01.parquet')
    current_data = pd.read_parquet('data/raw/green_tripdata_2023-02.parquet')

    def preprocess(df):
        df['duration'] = (df.lpep_dropoff_datetime - df.lpep_pickup_datetime).dt.total_seconds() / 60
        df = df[(df.duration >= 1) & (df.duration <= 60)]
        return df

    reference_data = preprocess(reference_data)
    current_data = preprocess(current_data)

    print("Generating data drift report...")
    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset(stat_test='ks')])
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html("evidently_data_drift_report.html")

    print("\n✅ Report saved successfully to evidently_data_drift_report.html.")

except Exception as e:
    print(f"\n❌ An unexpected error occurred: {e}")
