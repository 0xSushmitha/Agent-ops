import streamlit as st
import pandas as pd
import plotly.express as px
from queryController import LLMQueries
from modules import extract_unique_model_name,agentCallsPerModel
st.set_page_config(layout="wide", page_title="Tredence Agent Ops Analytics")
st.markdown("""
<style>
/* General Layout */
.block-container {
    padding: 0.5rem 0.5rem;
}

/* Reusable Box Style */
.dashboard-box {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 16px;
    background-color: #ffffff;
    margin-bottom: 16px;
}

/* Header Box Style */
.title-box {
    border: 1px solid #ccc;
    border-radius: 6px;
    padding: 10px 15px;
    background-color: #f5f6fa;
    margin-bottom: 0.5rem;
}
.title-box h5 {
    font-size: 18px;
    font-weight: 600;
    margin: 0;
}

/* Smaller inputs */
.stSelectbox > div, .stDateInput > div > input {
    font-size: 14px;
    height: 34px !important;
    padding: 2px 8px;
}
.stCheckbox {
    margin-top: 30px !important;
}

h6 {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* Reduce vertical spacing between Streamlit rows */
[data-testid="stVerticalBlock"] {
    gap: 0.2rem !important;
}
[data-testid="stHorizontalBlock"] {
    gap: 0.25rem !important;  /* or even 0.1rem */
    margin-bottom: 0.2rem !important;
}
/* Optional: reduce plot title spacing */
.plot-title {
    margin-bottom: 0.3rem;
}

/* Optional: tighter padding inside columns */
[data-testid="column"] {
    padding-top: 0.3rem !important;
    padding-bottom: 0.3rem !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* Box container for charts and tables */
.plot-box {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 16px;
    background-color: #ffffff;
    margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
    transition: box-shadow 0.3s ease;
}
.plot-box:hover {
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.15);
}

/* Plot or table titles inside the box */
.plot-title {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)
llm = LLMQueries()

#st.markdown("<div class='boxed-section'><h5 style='margin: 0;'>LLM Observability Dashboard</h5></div>", unsafe_allow_html=True)

st.markdown("""
<div style='background-color:lightblue; padding:1px 1px; border-radius:1px; margin-bottom:1px'>
    <h5 style='margin:0; font-size:19px;'>Tredence Agent Ops Platform</h5>
</div>
""", unsafe_allow_html=True)

with st.expander("Filters",expanded=True):
# --- Filters Box ---
    col1, col2, col3, col4 = st.columns([0.4, 0.2, 0.2, 0.2],vertical_alignment="center")
    with col1:
        selected = st.selectbox("Projects", llm.get_dropdown_options(), label_visibility="visible")
    with col2:
        min_date, max_date = llm.get_date_range_for_project(selected)
        start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date, label_visibility="visible")
    with col3:
        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date, label_visibility="visible")
    with col4:
        st.empty

# --- Metric Section ---
val1 = llm.get_tile1_value(selected, start_date, end_date)
val2 = round(llm.get_tile2_value(selected, start_date, end_date), 2)
val3 = llm.get_tile3_value(selected, start_date, end_date)
val4 = llm.get_tile4_value(selected, start_date, end_date)

# metric_cols = st.columns(4,border=True)
# metric_cols[0].metric("Total LLM Calls", val1)
# metric_cols[1].metric("Avg Latency", val2)
# metric_cols[2].metric("Prompt Tokens", val3)
# metric_cols[3].metric("Completion Tokens", val4)

col1, col2, col3,col4 = st.columns([0.55,0.65, 1.7, 2],border=True) 
with col1:
    st.metric("Total LLM Calls", val1)
    st.divider()
    st.metric("Avg Latency", val2)

with col2:
    st.metric("Prompt Tokens", val3)
    st.divider()
    st.metric("Completion \n Tokens", val4)
with col3:
    trace_data=llm.get_trace_by_name(selected,start_date,end_date)
    st.markdown("Trace Counts Table")
    st.dataframe(trace_data, use_container_width=True,hide_index=True,height=230)
#     fig = px.bar(
#         trace_data,
#         x='trace_count',
#         y='name',
#         orientation='h',
#         title='Trace Counts by Name'
# ***REMOVED***
#     # Optional: clean layout for a small visual
#     fig.update_layout(
#         height=300,
#         margin=dict(l=80, r=20, t=40, b=20),
#         xaxis_title='Count',
#         yaxis_title='Name'
# ***REMOVED***
#     fig.update_traces(

#     textposition='inside',
#     texttemplate='%{x}',
#     marker_color="#6f6f92"  # soft purple similar to your image
# )
   # st.plotly_chart(fig, use_container_width=True)
   

with col4:
    fig1 = px.bar(pd.DataFrame({'x': ['A', 'B'], 'y': [10, 20]}), x='x', y='y', title='Example Chart 1')
    fig1.update_layout(
        height=230,
        margin=dict(l=80, r=20, t=40, b=20)
***REMOVED***
    st.plotly_chart(fig1, use_container_width=True)

# --- Charts ---
chart_col1, chart_col2 = st.columns(2,border=True)
with chart_col1:
    totaltraces=llm.get_metric2_data(selected,start_date,end_date)
    st.markdown("Traces Analytics")
    tab1,tab2=st.tabs(["Total traces","Traces over time"])
    status_summary = totaltraces.groupby('status')['trace_count'].sum().reset_index()
    with tab1:
        # Create the pie chart
        fig = px.pie(
            status_summary,
            names='status',
            values='trace_count',
            title='Distribution of Total Traces'
    ***REMOVED***
        fig.update_traces(textposition='inside', textinfo='percent+label+value')
        st.plotly_chart(fig,use_container_width=True)
    with tab2:
        fig = px.line(
            totaltraces,
            x='date',
            y='trace_count',
            color='status',
            markers=True,
            title='Trace Counts Over Time by Status'
    ***REMOVED***
        fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Number of Traces',
        legend_title='Status Code'
    ***REMOVED***  
        # Render in Streamlit
        st.plotly_chart(fig, use_container_width=True)  

with chart_col2:
    grouped = agentCallsPerModel(llm.getAttributes(selected,start_date, end_date))
    fig = px.bar(
        grouped,
        x='model_name',
        y='count',
        color='agent_name',
        barmode='group',  # or 'stack'
        title='Agent Calls per Model'
***REMOVED***
    st.plotly_chart(fig, use_container_width=True)



# # Traces Over Time Chart
# traces_df = llm.get_metric1_data(selected, start_date, end_date)
# if not traces_df.empty:
#     traces_df['hour'] = pd.to_datetime(traces_df['hour'])
#     fig1 = px.area(traces_df, x='hour', y='count')
#     fig1.update_layout(height=300, margin=dict(t=30, b=10))
#     with chart_col1:
#         st.markdown("<div class='plot-title'>Traces Over Time</div>", unsafe_allow_html=True)
#         st.plotly_chart(fig1, use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)

# # Model Usage Chart
# attr_df = llm.getAttributes(selected, start_date, end_date)
# attr_df['model_name'] = attr_df['attributes'].apply(extract_unique_model_name)
# attr_df = attr_df.explode('model_name').dropna()
# attr_df['date'] = pd.to_datetime(attr_df['start_time']).dt.date
# model_usage = attr_df.groupby(['model_name', 'date']).size().reset_index(name='count')
# fig2 = px.line(model_usage, x='date', y='count', color='model_name')
# fig2.update_layout(height=300, margin=dict(t=30, b=10))
# with chart_col2:
#     st.markdown("<div class='plot-title'>Model Usage Over Time</div>", unsafe_allow_html=True)
#     st.plotly_chart(fig2, use_container_width=True)
#     st.markdown("</div>", unsafe_allow_html=True)


# bottom_col1, bottom_col2 = st.columns(2,border=True)

# # Score Table
# with bottom_col1:
#     st.markdown("<div class='plot-title'>📋 Score Metrics</div>", unsafe_allow_html=True)
#     st.dataframe(score_df, height=220)
#     st.markdown("</div>", unsafe_allow_html=True)

# # Agent Calls Bar Chart
# with bottom_col2:
#     grouped = attr_df.groupby(['model_name', 'agent_name']).size().reset_index(name='count')
#     fig3 = px.bar(grouped, x='model_name', y='count', color='agent_name', barmode='group')
#     fig3.update_layout(height=300, margin=dict(t=30, b=10))
#     st.markdown("<div class='plot-title'>👥 Agent Calls per Model</div>", unsafe_allow_html=True)
#     st.plotly_chart(fig3, use_container_width=True)
#     st.markdown("</div>", unsafe_allow_html=True)

# --- Metric Section ---
# val1 = llm.get_tile1_value(selected, start_date, end_date)
# val2 = round(llm.get_tile2_value(selected, start_date, end_date), 2)
# val3 = llm.get_tile3_value(selected, start_date, end_date)
# val4 = llm.get_tile4_value(selected, start_date, end_date)

# metric_cols = st.columns(4,border=True)
# metric_cols[0].metric("Total LLM Calls", val1)
# metric_cols[1].metric("Avg Latency", val2)
# metric_cols[2].metric("Prompt Tokens", val3)
# metric_cols[3].metric("Completion Tokens", val4)

# trace_data=llm.get_trace_by_name(selected,start_date,end_date)
# fig = px.bar(
#     trace_data,
#     x='trace_count',
#     y='name',
#     orientation='h',
#     title='Trace Counts by Name'
# )

# # Optional: clean layout for a small visual
# fig.update_layout(
#     height=300,
#     margin=dict(l=80, r=20, t=40, b=20),
#     xaxis_title='Count',
#     yaxis_title='Name'
# )

# st.plotly_chart(fig, use_container_width=True)

# # --- Charts ---
# chart_col1, chart_col2 = st.columns(2,border=True)
# with chart_col1:
#     totaltraces=llm.get_metric2_data(selected,start_date,end_date)
#     st.markdown("Traces Analytics")
#     tab1,tab2=st.tabs(["Total traces","Traces over time"])
#     status_summary = totaltraces.groupby('status')['trace_count'].sum().reset_index()
#     with tab1:
#         # Create the pie chart
#         fig = px.pie(
#             status_summary,
#             names='status',
#             values='trace_count',
#             title='Distribution of Total Traces'
#     ***REMOVED***
#         fig.update_traces(textposition='inside', textinfo='percent+label+value')
#         st.plotly_chart(fig,use_container_width=True)
#     with tab2:
#         fig = px.line(
#             totaltraces,
#             x='date',
#             y='trace_count',
#             color='status',
#             markers=True,
#             title='Trace Counts Over Time by Status'
#     ***REMOVED***
#         fig.update_layout(
#         xaxis_title='Date',
#         yaxis_title='Number of Traces',
#         legend_title='Status Code'
#     ***REMOVED***  
#         # Render in Streamlit
#         st.plotly_chart(fig, use_container_width=True)  

# with chart_col2:
#     grouped = agentCallsPerModel(llm.getAttributes(selected,start_date, end_date))
#     fig = px.bar(
#         grouped,
#         x='model_name',
#         y='count',
#         color='agent_name',
#         barmode='group',  # or 'stack'
#         title='Agent Calls per Model'
# ***REMOVED***
#     st.plotly_chart(fig, use_container_width=True)
