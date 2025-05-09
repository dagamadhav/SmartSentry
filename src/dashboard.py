import streamlit as st
import json
import pandas as pd
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Security Alert Dashboard",
    page_icon="🔒",
    layout="wide"
)

# Load the alerts data
def load_alerts():
    with open('data/alerts/alerts1.json', 'r') as f:
        return json.load(f)

# Main dashboard
def main():
    st.title("🔒 Security Alert Dashboard")
    
    # Load data
    data = load_alerts()
    
    # Create three columns for key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Alerts", data['alert_summary']['total_alerts'])
    
    with col2:
        st.metric("System Accuracy", data['system_performance']['accuracy'])
    
    with col3:
        st.metric("False Positives", data['system_performance']['false_positives'])
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Alerts by Category
        category_data = data['alert_statistics']['by_category']
        fig_category = px.pie(
            values=list(category_data.values()),
            names=list(category_data.keys()),
            title="Alerts by Category"
        )
        st.plotly_chart(fig_category, use_container_width=True)
    
    with col2:
        # Alerts by Object
        object_data = data['alert_statistics']['by_object']
        fig_object = px.bar(
            x=list(object_data.keys()),
            y=list(object_data.values()),
            title="Alerts by Object Type"
        )
        st.plotly_chart(fig_object, use_container_width=True)
    
    # Alerts Timeline
    st.subheader("Alert Timeline")
    alerts_df = pd.DataFrame(data['alerts'])
    alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
    fig_timeline = px.scatter(
        alerts_df,
        x='timestamp',
        y='confidence',
        color='category',
        size='confidence',
        title="Alert Timeline with Confidence Levels"
    )
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Recent Alerts Table
    st.subheader("Recent Alerts")
    alerts_df = alerts_df.sort_values('timestamp', ascending=False)
    st.dataframe(
        alerts_df[['timestamp', 'object', 'confidence', 'category', 'description']],
        use_container_width=True
    )
    
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