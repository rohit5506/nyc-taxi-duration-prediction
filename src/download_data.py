# src/download_data.py

import requests
import os

def download_taxi_data(year=2023, months=[1, 2, 3]):
    """Downloads NYC Green Taxi data for a given year and months."""
    
    # Create the raw data directory if it doesn't exist
    raw_data_path = "data/raw"
    os.makedirs(raw_data_path, exist_ok=True)
    
    base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data"
    
    for month in months:
        # Format filename e.g., green_tripdata_2023-01.parquet
        filename = f"green_tripdata_{year}-{month:02d}.parquet"
        file_path = os.path.join(raw_data_path, filename)
        url = f"{base_url}/{filename}"
        
        if not os.path.exists(file_path):
            print(f"Downloading {filename}...")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Raise an exception for bad status codes
                
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Successfully downloaded {filename}")
            except requests.exceptions.RequestException as e:
                print(f"Failed to download {filename}. Error: {e}")
        else:
            print(f"{filename} already exists. Skipping download.")

if __name__ == "__main__":
    download_taxi_data()
