import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from utils.leagues import get_league_metadata

def show():
    st.header('Use Case 5: Team Performance Over Time')
    
    # Get league metadata
    clubs, _ = get_league_metadata('epl')

    # Dropdown for selecting a team
    team = st.selectbox('Select Team:', clubs)
    
    # Calendar widgets for selecting time periods
    st.write("Select two different time periods to compare:")
    start_date_1 = st.date_input("Start Date for Time Period 1", value=datetime(2023, 1, 1))
    end_date_1 = st.date_input("End Date for Time Period 1", value=datetime(2023, 12, 31))
    start_date_2 = st.date_input("Start Date for Time Period 2", value=datetime(2022, 1, 1))
    end_date_2 = st.date_input("End Date for Time Period 2", value=datetime(2022, 12, 31))
    
    # Check if valid time periods are selected
    if start_date_1 >= end_date_1 or start_date_2 >= end_date_2:
        st.error("Please ensure that the start date is before the end date for both time periods.")
        return
    
    # Randomly generated data to simulate the metrics
    np.random.seed(42)
    metric_1_period_1 = np.random.normal(loc=0, scale=1, size=100)
    metric_1_period_2 = np.random.normal(loc=0.5, scale=1, size=100)
        
    metric_2_period_1_value = np.sum(np.random.randn(100))
    metric_2_period_2_value = np.sum(np.random.randn(100))
    
    # Plotting the Metrics
    st.write(f"**Metrics Comparison for {team}**")
    
    # Metric 1: Histogram
    st.write("### Metric 1: Histogram")
    fig, ax = plt.subplots()
    ax.hist(metric_1_period_1, bins=30, alpha=0.7, color='red', label='Time Period 1')
    ax.hist(metric_1_period_2, bins=30, alpha=0.7, color='blue', label='Time Period 2')
    ax.set_title("Metric 1 Distribution")
    ax.set_xlabel("Metric 1")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)
        
    # Metric 2: Pie Chart
    st.write("### Metric 2: Pie Chart")
    fig, ax = plt.subplots()
    ax.pie([metric_2_period_1_value, metric_2_period_2_value], 
           labels=['Time Period 1', 'Time Period 2'], 
           autopct='%1.1f%%', colors=['red', 'blue'], startangle=140)
    ax.set_title("Metric 2 Distribution")
    st.pyplot(fig)
    
    # Additional Metric: Line Chart Example
    st.write("### Metric 3: Line Chart Example")
    dates = pd.date_range(start='1/1/2023', periods=100)
    metric_3_period_1 = np.cumsum(np.random.randn(100))
    metric_3_period_2 = np.cumsum(np.random.randn(100))
    
    fig, ax = plt.subplots()
    ax.plot(dates, metric_3_period_1, color='red', label='Time Period 1')
    ax.plot(dates, metric_3_period_2, color='blue', label='Time Period 2')
    ax.set_title("Metric 3 Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Metric 3")
    ax.legend()
    st.pyplot(fig)

# Call the function to display the UI
show()
