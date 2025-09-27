# src/process_data.py

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import pickle
import os

def preprocess_data(input_dir: str, output_dir: str):
    """Reads raw parquet files, preprocesses them, and saves artifacts."""
    
    print("Starting data processing...")
    
    # Read the raw data
    df_train = pd.read_parquet(os.path.join(input_dir, 'green_tripdata_2023-01.parquet'))
    df_val = pd.read_parquet(os.path.join(input_dir, 'green_tripdata_2023-02.parquet'))

    # Feature Engineering: Calculate trip duration in minutes
    df_train['duration'] = (df_train.lpep_dropoff_datetime - df_train.lpep_pickup_datetime).dt.total_seconds() / 60
    df_val['duration'] = (df_val.lpep_dropoff_datetime - df_val.lpep_pickup_datetime).dt.total_seconds() / 60

    # Filter outliers
    df_train = df_train[(df_train.duration >= 1) & (df_train.duration <= 60)].copy()
    df_val = df_val[(df_val.duration >= 1) & (df_val.duration <= 60)].copy()

    # Define categorical and numerical features
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    
    df_train[categorical] = df_train[categorical].fillna(-1).astype('int').astype('str')
    df_val[categorical] = df_val[categorical].fillna(-1).astype('int').astype('str')
    
    # Use DictVectorizer to one-hot encode categorical features
    dv = DictVectorizer()
    
    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)
    
    val_dicts = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)

    # Extract target variable
    y_train = df_train['duration'].values
    y_val = df_val['duration'].values

    # Save the DictVectorizer and the processed datasets
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "dv.pkl"), "wb") as f:
        pickle.dump(dv, f)
    with open(os.path.join(output_dir, "X_train.pkl"), "wb") as f:
        pickle.dump(X_train, f)
    with open(os.path.join(output_dir, "X_val.pkl"), "wb") as f:
        pickle.dump(X_val, f)
    with open(os.path.join(output_dir, "y_train.pkl"), "wb") as f:
        pickle.dump(y_train, f)
    with open(os.path.join(output_dir, "y_val.pkl"), "wb") as f:
        pickle.dump(y_val, f)

    print("Data processing complete. Artifacts saved.")

if __name__ == "__main__":
    preprocess_data(input_dir='data/raw', output_dir='data/processed')
