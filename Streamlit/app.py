import streamlit as st
import pandas as pd
import plotly.express as px
import graphviz
from datetime import datetime, timedelta
from queryController import LLMQueries
from modules import extract_unique_model_name

st.set_page_config(layout="wide", page_title="LLM Observability Dashboard")

# Instantiate LLMQueries object
llm = LLMQueries()

# Sidebar Controls
with st.sidebar:
    st.title("⚙️ Control Panel")

    # Project selection
    selected = st.selectbox("📂 Select Project", llm.get_dropdown_options())

    # Fetch min and max date range from DB for selected project
    min_date, max_date = llm.get_date_range_for_project(selected)

    # Date range filter
    start_date, end_date = st.date_input(
        "📅 Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        format="YYYY-MM-DD"
***REMOVED***

    # Token filter
    exclude_zero_tokens = st.checkbox("🚫 Exclude spans with 0 tokens", value=False)

    # Placeholder for latency slider (set later based on data)
    latency_slider_placeholder = st.empty()

st.title("📈 LLM Observability Dashboard")

# Metric Tiles
col1, col2, col3, col4 = st.columns(4)

with col1:
    value1 = llm.get_tile1_value(selected, start_date, end_date)
    st.metric(label="Total LLM Calls", value=value1)

with col2:
    value2 = llm.get_tile2_value(selected, start_date, end_date)
    st.metric(label="Avg Latency", value=value2)

with col3:
    value3 = llm.get_tile3_value(selected, start_date, end_date)
    st.metric(label="Prompt token count", value=value3)

with col4:
    value4 = llm.get_tile4_value(selected, start_date, end_date)
    st.metric(label="Completion token count", value=value4)

# Line and Bar Plots
def plot_line(df, y_col, title):
    if df.empty:
        st.warning(f"No data available for {title}.")
        return
    df['hour'] = pd.to_datetime(df['hour'])
    full_hours = pd.DataFrame({'hour': pd.date_range(start=df['hour'].min(), end=df['hour'].max(), freq='H')})
    df_full = pd.merge(full_hours, df, on='hour', how='left')
    df_full[y_col] = df_full[y_col].fillna(0)
    fig = px.line(df_full, x='hour', y=y_col, title=title)
    fig.update_layout(xaxis_title="Hour", yaxis_title=y_col)
    st.plotly_chart(fig, use_container_width=True)

def plot_bar(df, y_col="calls", title="Number of LLM Calls Per Day"):
    if df.empty:
        st.warning(f"No data available for {title}.")
        return
    fig = px.bar(
        df,
        x='date',
        y=y_col,
        title=title,
        labels={'date': 'Date', y_col: 'Number of LLM Calls'},
        text=y_col
***REMOVED***
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Calls",
        yaxis=dict(tickformat="d"),
        bargap=0.2
***REMOVED***
    st.plotly_chart(fig, use_container_width=True)

# Plot metrics
plot_line(llm.get_metric1_data(selected, start_date, end_date), 'count', "Traces over Time")
plot_bar(llm.get_metric2_data(selected, start_date, end_date), "llm_call_count", "LLM Call Count")

# Span Tree Visualization
st.subheader("🌳 Select Root Span (Top-Level Agent Call)")
df_roots = llm.get_root_spans(selected)
if df_roots.empty:
    st.info("No root-level spans found.")
else:
    trace_options = {
        f"{row['name']} @ {row['start_time']}": (row['trace_rowid'], row['span_id'])
        for _, row in df_roots.iterrows()
    }
    selected_label = st.selectbox("Choose a Root Span", list(trace_options.keys()))
    selected_trace_id, selected_root_span_id = trace_options[selected_label]
    df_trace = llm.get_span_tree_for_trace(selected_trace_id)

    def build_trace_tree(df_trace):
        dot = graphviz.Digraph()
        for _, row in df_trace.iterrows():
            label = f"{row['name']}\n{row['latency']:.2f}s"
            dot.node(row['span_id'], label)
            if pd.notna(row['parent_id']):
                dot.edge(row['parent_id'], row['span_id'])
        return dot

    if df_trace.empty:
        st.warning("No spans found for selected trace.")
    else:
        st.subheader(f"📡 Span Tree for: {selected_label}")
        st.graphviz_chart(build_trace_tree(df_trace))

# Span Analysis Section
st.subheader("📊 Span Analysis")
df_expensive = llm.get_expensive_spans(selected, start_date, end_date)

if df_expensive.empty:
    st.info("No span data available.")
else:
    filtered_df = df_expensive.copy()
    if exclude_zero_tokens:
        filtered_df = filtered_df[filtered_df["total_tokens"] > 0]

    if filtered_df.empty:
        st.warning("No spans left after applying token filter.")
    else:
        if (filtered_df["latency"] > 0).any():
            min_latency = float(filtered_df["latency"].min())
            max_latency = float(filtered_df["latency"].max())
            if min_latency < max_latency:
                latency_range = latency_slider_placeholder.slider(
                    "⏱️ Latency Range (seconds)",
                    min_value=min_latency,
                    max_value=max_latency,
                    value=(min_latency, max_latency),
                    step=0.1
            ***REMOVED***
                filtered_df = filtered_df[filtered_df["latency"].between(*latency_range)]

        if filtered_df.empty:
            st.warning("No spans match the latency range.")
        else:
            st.dataframe(
                filtered_df.sort_values(by=["total_tokens", "latency"], ascending=False),
                use_container_width=True
        ***REMOVED***

# Model name classfification

model_df= llm.getAttributes(selected,start_date, end_date) 

model_df['model_name']=model_df['attributes'].apply(extract_unique_model_name)

# Explode if multiple models found
model_df = model_df.explode('model_name')

# Drop rows with no model name
model_df = model_df[model_df['model_name'].notnull()]

# --- Group data by model_name and agent_name ---
grouped = model_df.groupby(['model_name', 'agent_name']).size().reset_index(name='count')

# --- Plot ---
fig = px.bar(
    grouped,
    x='model_name',
    y='count',
    color='agent_name',
    barmode='group',  # or 'stack' for stacked bar chart
    title='Agent Calls per Model'
)
st.plotly_chart(fig, use_container_width=True)
