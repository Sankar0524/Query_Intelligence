import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import random

# Generate synthetic query performance data
def generate_query_data(num_records=500):
    timestamps = [datetime.now() - timedelta(minutes=random.randint(0, 1440)) for _ in range(num_records)]
    users = [f"user_{random.randint(1, 10)}" for _ in range(num_records)]
    warehouses = [f"warehouse_{random.randint(1, 5)}" for _ in range(num_records)]
    durations = np.random.exponential(scale=5, size=num_records)  # in seconds
    costs = np.random.exponential(scale=0.5, size=num_records)    # in credits
    failures = [random.choice([0, 1]) if random.random() < 0.1 else 0 for _ in range(num_records)]
    queries = [f"SELECT * FROM table_{random.randint(1, 20)} WHERE col = {random.randint(1, 100)}" for _ in range(num_records)]

    df = pd.DataFrame({
        "Timestamp": timestamps,
        "User": users,
        "Warehouse": warehouses,
        "Query": queries,
        "Duration_sec": durations,
        "Cost_credits": costs,
        "Failed": failures
    })
    return df

# Suggest optimization strategies
def suggest_optimizations(row):
    suggestions = []
    if row['Duration_sec'] > 10:
        suggestions.append("Consider adding clustering keys.")
    if "JOIN" in row['Query'].upper():
        suggestions.append("Prune unnecessary joins.")
    if row['Failed'] == 1:
        suggestions.append("Check query syntax and resource limits.")
    return suggestions

# Load or generate data
query_data = generate_query_data()
query_data['Suggestions'] = query_data.apply(suggest_optimizations, axis=1)

# Streamlit dashboard
st.title("Query Intelligence Engine Dashboard")

# Filters
st.sidebar.header("Filters")
selected_user = st.sidebar.selectbox("Select User", ["All"] + sorted(query_data['User'].unique()))
selected_warehouse = st.sidebar.selectbox("Select Warehouse", ["All"] + sorted(query_data['Warehouse'].unique()))
time_range = st.sidebar.slider("Select Time Range (hours ago)", 0, 24, (0, 24))

# Apply filters
filtered_data = query_data.copy()
if selected_user != "All":
    filtered_data = filtered_data[filtered_data['User'] == selected_user]
if selected_warehouse != "All":
    filtered_data = filtered_data[filtered_data['Warehouse'] == selected_warehouse]
start_time = datetime.now() - timedelta(hours=time_range[1])
end_time = datetime.now() - timedelta(hours=time_range[0])
filtered_data = filtered_data[(filtered_data['Timestamp'] >= start_time) & (filtered_data['Timestamp'] <= end_time)]

# Display metrics
st.subheader("Query Summary")
st.write(f"Total Queries: {len(filtered_data)}")
st.write(f"Failed Queries: {filtered_data['Failed'].sum()}")
st.write(f"Average Duration (sec): {filtered_data['Duration_sec'].mean():.2f}")
st.write(f"Average Cost (credits): {filtered_data['Cost_credits'].mean():.2f}")

# Heatmap by time and user
st.subheader("Heatmap: Query Duration by Time and User")
heatmap_data = filtered_data.copy()
heatmap_data['Hour'] = heatmap_data['Timestamp'].dt.hour
heatmap_pivot = heatmap_data.pivot_table(index='Hour', columns='User', values='Duration_sec', aggfunc='mean')
fig1 = px.imshow(heatmap_pivot, labels=dict(x="User", y="Hour", color="Avg Duration (sec)"), title="Query Duration Heatmap")
st.plotly_chart(fig1)

# Heatmap by warehouse and user
st.subheader("Heatmap: Query Cost by Warehouse and User")
heatmap_pivot2 = filtered_data.pivot_table(index='Warehouse', columns='User', values='Cost_credits', aggfunc='mean')
fig2 = px.imshow(heatmap_pivot2, labels=dict(x="User", y="Warehouse", color="Avg Cost (credits)"), title="Query Cost Heatmap")
st.plotly_chart(fig2)

# Show table with suggestions
st.subheader("Query Optimization Suggestions")
st.dataframe(filtered_data[['Timestamp', 'User', 'Warehouse', 'Query', 'Duration_sec', 'Cost_credits', 'Failed', 'Suggestions']])
