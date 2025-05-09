import streamlit as st
import json
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import os
import glob
import time

# Set page config
st.set_page_config(
    page_title="Security Alert Dashboard",
    page_icon="🔒",
    layout="wide"
)

# Add auto-refresh
st.empty()
placeholder = st.empty()

def get_available_sessions():
    """Get list of available alert sessions"""
    pattern = os.path.join('data/alerts', "alerts_*.json")
    files = glob.glob(pattern)
    sessions = []
    
    for file in files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                sessions.append({
                    'filename': os.path.basename(file),
                    'start_time': data['session_info']['start_time'],
                    'end_time': data['session_info']['end_time'],
                    'total_alerts': data['alert_summary']['total_alerts']
                })
        except Exception as e:
            st.error(f"Error reading session file {file}: {str(e)}")
    
    return sorted(sessions, key=lambda x: x['start_time'], reverse=True)

def load_alerts(session_file):
    """Load alerts from JSON file with error handling"""
    try:
        with open(os.path.join('data/alerts', session_file), 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading alerts: {str(e)}")
        return None

def filter_alerts_by_timeframe(alerts_df, timeframe):
    """Filter alerts based on selected timeframe"""
    if timeframe == "All Time":
        return alerts_df
    
    now = datetime.now()
    if timeframe == "Last Hour":
        start_time = now - timedelta(hours=1)
    elif timeframe == "Last 6 Hours":
        start_time = now - timedelta(hours=6)
    elif timeframe == "Last 24 Hours":
        start_time = now - timedelta(hours=24)
    elif timeframe == "Last 7 Days":
        start_time = now - timedelta(days=7)
    else:
        return alerts_df
    
    return alerts_df[alerts_df['timestamp'] >= start_time]

def create_metrics(data):
    """Create key metrics display"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Alerts", data['alert_summary']['total_alerts'])
    
    with col2:
        st.metric("System Accuracy", data['system_performance']['accuracy'])
    
    with col3:
        st.metric("False Positives", data['system_performance']['false_positives'])

def create_category_chart(data, timeframe):
    """Create category distribution chart"""
    alerts_df = pd.DataFrame(data['alerts'])
    alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
    alerts_df = filter_alerts_by_timeframe(alerts_df, timeframe)
    
    category_counts = alerts_df['category'].value_counts()
    
    fig = px.pie(
        values=category_counts.values,
        names=category_counts.index,
        title=f"Alerts by Category ({timeframe})"
    )
    st.plotly_chart(fig, use_container_width=True)

def create_object_chart(data, timeframe):
    """Create object distribution chart"""
    alerts_df = pd.DataFrame(data['alerts'])
    alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
    alerts_df = filter_alerts_by_timeframe(alerts_df, timeframe)
    
    object_counts = alerts_df['object'].value_counts()
    
    fig = px.bar(
        x=object_counts.index,
        y=object_counts.values,
        title=f"Alerts by Object Type ({timeframe})"
    )
    st.plotly_chart(fig, use_container_width=True)

def create_timeline(data, timeframe):
    """Create alert timeline"""
    alerts_df = pd.DataFrame(data['alerts'])
    alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
    alerts_df = filter_alerts_by_timeframe(alerts_df, timeframe)
    
    fig = px.scatter(
        alerts_df,
        x='timestamp',
        y='confidence',
        color='category',
        size='confidence',
        title=f"Alert Timeline with Confidence Levels ({timeframe})"
    )
    st.plotly_chart(fig, use_container_width=True)

def create_alerts_table(data, timeframe):
    """Create alerts table"""
    alerts_df = pd.DataFrame(data['alerts'])
    alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
    alerts_df = filter_alerts_by_timeframe(alerts_df, timeframe)
    alerts_df = alerts_df.sort_values('timestamp', ascending=False)
    
    # Select and rename columns for display
    display_df = alerts_df[['timestamp', 'object', 'confidence', 'category', 'description']]
    display_df.columns = ['Time', 'Object', 'Confidence', 'Category', 'Description']
    
    st.dataframe(display_df, use_container_width=True)

def main():
    st.title("🔒 Security Alert Dashboard")
    
    # Add auto-refresh checkbox
    auto_refresh = st.checkbox("Enable Auto-refresh", value=True)
    if auto_refresh:
        st.empty()
        time.sleep(5)  # Refresh every 5 seconds
        st.experimental_rerun()
    
    # Get available sessions
    sessions = get_available_sessions()
    
    if not sessions:
        st.error("No sessions available. Please start the camera system first.")
        return
    
    # Session selection
    session_options = {f"{s['start_time']} - {s['end_time'] or 'Active'} ({s['total_alerts']} alerts)": s['filename'] 
                      for s in sessions}
    
    # Default to most recent session
    default_session = list(session_options.keys())[0]
    
    selected_session = st.selectbox(
        "Select Session",
        options=list(session_options.keys()),
        index=0,  # Select most recent session by default
        format_func=lambda x: x
    )
    
    # Time frame selection
    timeframe = st.selectbox(
        "Select Time Frame",
        options=["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days", "All Time"]
    )
    
    # Load data for selected session
    data = load_alerts(session_options[selected_session])
    if data is None:
        st.error("Failed to load alert data. Please check if the alerts file exists.")
        return
    
    # Create dashboard sections
    create_metrics(data)
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        create_category_chart(data, timeframe)
    
    with col2:
        create_object_chart(data, timeframe)
    
    # Create timeline
    st.subheader("Alert Timeline")
    create_timeline(data, timeframe)
    
    # Create alerts table
    st.subheader("Recent Alerts")
    create_alerts_table(data, timeframe)
    
    # System Performance Metrics
    st.subheader("System Performance")
    performance_data = data['system_performance']
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Frames Processed", f"{performance_data['total_frames_processed']:,}")
    
    with col2:
        st.metric("Average Processing Time", performance_data['average_processing_time'])
    
    with col3:
        st.metric("False Negatives", performance_data['false_negatives'])

if __name__ == "__main__":
    main() 