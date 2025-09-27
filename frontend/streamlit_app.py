import streamlit as st
import requests
import json
import pandas as pd

# Page setup
st.set_page_config(
    page_title="NYC Taxi Duration Predictor",
    page_icon="🚖",
    layout="wide"
)

# Title
st.title("🚖 NYC Taxi Duration Predictor")
st.write("Predict how long your taxi ride will take!")

# Get API URL from user
st.sidebar.header("Settings")
api_url = st.sidebar.text_input(
    "Enter your Cloud Run API URL:", 
    placeholder="https://your-service-name.run.app/predict"
)

# Main form
st.header("Enter Trip Details")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Pickup Location")
    pickup_options = {
        "Times Square": "142",
        "JFK Airport": "132", 
        "LaGuardia Airport": "138",
        "Penn Station": "161",
        "Custom": "custom"
    }
    pickup_choice = st.selectbox("Choose pickup location:", list(pickup_options.keys()))
    
    if pickup_choice == "Custom":
        pickup_id = st.number_input("Enter pickup location ID:", min_value=1, value=142)
    else:
        pickup_id = int(pickup_options[pickup_choice])
        st.info(f"Pickup ID: {pickup_id}")

with col2:
    st.subheader("Dropoff Location") 
    dropoff_options = {
        "Brooklyn Heights": "265",
        "Williamsburg": "261",
        "Times Square": "142",
        "JFK Airport": "132",
        "Custom": "custom"
    }
    dropoff_choice = st.selectbox("Choose dropoff location:", list(dropoff_options.keys()))
    
    if dropoff_choice == "Custom":
        dropoff_id = st.number_input("Enter dropoff location ID:", min_value=1, value=265)
    else:
        dropoff_id = int(dropoff_options[dropoff_choice])
        st.info(f"Dropoff ID: {dropoff_id}")

# Distance input
st.subheader("Trip Distance")
trip_distance = st.slider("Distance in miles:", min_value=0.1, max_value=20.0, value=5.0, step=0.1)

# Prediction button
if st.button("🔮 Get Prediction", type="primary"):
    if not api_url or "your-service-name" in api_url:
        st.error("⚠️ Please enter your actual Cloud Run API URL in the sidebar!")
    else:
        # Prepare request data
        data = {
            "PULocationID": str(pickup_id),
            "DOLocationID": str(dropoff_id), 
            "trip_distance": float(trip_distance)
        }
        
        try:
            # Make API call
            with st.spinner("Making prediction..."):
                response = requests.post(api_url, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                duration = result.get('predicted_duration_minutes', 0)
                
                # Show results
                st.success("✅ Prediction successful!")
                
                col3, col4, col5 = st.columns(3)
                with col3:
                    st.metric("Duration", f"{duration:.1f} minutes")
                with col4:
                    st.metric("In Hours", f"{duration/60:.2f} hours")
                with col5:
                    avg_speed = (trip_distance / duration) * 60 if duration > 0 else 0
                    st.metric("Avg Speed", f"{avg_speed:.1f} mph")
                
                st.balloons()
            else:
                st.error(f"❌ API Error: {response.status_code}")
                st.write("Response:", response.text)
                
        except requests.exceptions.Timeout:
            st.error("⏰ Request timed out. Check your API URL.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Info section
st.markdown("---")
st.info("""
**How to use:**
1. Enter your Cloud Run API URL in the sidebar
2. Choose pickup and dropoff locations  
3. Set the trip distance
4. Click 'Get Prediction' to see results!
""")
