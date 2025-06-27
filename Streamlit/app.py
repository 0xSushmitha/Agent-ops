import streamlit as st
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import plotly.express as px
import graphviz

# --- DB CONNECTION ---
***REMOVED***
***REMOVED***
        host=st.secrets["db_host"],
***REMOVED***
        dbname=st.secrets["db_name"],
        user=st.secrets["db_user"],
        password=st.secrets["db_password"]
***REMOVED***

# --- FETCH DROPDOWN OPTIONS ---
@st.cache_data
def get_dropdown_options():
    query = "SELECT DISTINCT name as category FROM projects ORDER BY name"
    with get_connection() as conn:
        df = pd.read_sql(query, conn)
    return df['category'].tolist()

def get_tile1_value(selected):
    query = "SELECT COUNT(*) AS value FROM spans where span_kind = 'LLM' AND trace_rowid in (select id from traces where project_rowid in (select id from projects where name= %s))"
    with get_connection() as conn:
        return pd.read_sql(query, conn, params=(selected,))['value'][0]

def get_tile2_value(selected):
    query = """SELECT 
    AVG(EXTRACT(EPOCH FROM end_time - start_time)) AS value 
FROM spans 
WHERE trace_rowid IN (
    SELECT id FROM traces 
    WHERE project_rowid IN (
        SELECT id FROM projects WHERE name = %s
***REMOVED***
)
"""
    with get_connection() as conn:
        return pd.read_sql(query, conn, params=(selected,))['value'][0]

def get_tile3_value(selected):
    query = """SELECT 
    SUM(llm_token_count_prompt) AS value 
FROM spans 
WHERE trace_rowid IN (
    SELECT id FROM traces 
    WHERE project_rowid IN (
        SELECT id FROM projects WHERE name = %s
***REMOVED***
)
"""
    with get_connection() as conn:
        return pd.read_sql(query, conn, params=(selected,))['value'][0]

def get_tile4_value(selected):
    query = """SELECT 
    SUM(llm_token_count_completion) AS value 
FROM spans 
WHERE trace_rowid IN (
    SELECT id FROM traces 
    WHERE project_rowid IN (
        SELECT id FROM projects WHERE name = %s
***REMOVED***
)
"""
    with get_connection() as conn:
        return pd.read_sql(query, conn, params=(selected,))['value'][0]


# --- SEPARATE QUERY FUNCTIONS ---
def get_metric1_data(selected_value):
    query = """
    SELECT
        DATE_TRUNC('hour', start_time) as hour,
        COUNT(DISTINCT trace_rowid) AS count
    FROM spans
    WHERE span_kind = 'LLM'
    AND trace_rowid IN (
        SELECT id FROM traces
        WHERE project_rowid IN (
        SELECT id FROM projects WHERE name = %s
    ***REMOVED***
***REMOVED***
    GROUP BY DATE_TRUNC('hour', start_time)
    ORDER BY hour;
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=(selected_value,))
    return df

def get_metric2_data(selected_value):
    query = """SELECT 
    DATE(start_time) AS date,
    COUNT(*) AS llm_call_count
FROM spans
WHERE span_kind = 'LLM'
  AND trace_rowid IN (
    SELECT id FROM traces
    WHERE project_rowid IN (SELECT id FROM projects WHERE name = %s)
  )
GROUP BY date
ORDER BY date;"""
    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=(selected_value,))
    return df

def get_metric3_data(selected_value):
    query = "select start_time, id from spans where trace_rowid in (select id from traces where project_rowid in (select id from projects where name= %s)) ORDER BY start_time"
    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=(selected_value,))
    return df

def get_metric4_data(selected_value):
    query = "select start_time, id from spans where trace_rowid in (select id from traces where project_rowid in (select id from projects where name= %s)) ORDER BY start_time"
    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=(selected_value,))
    return df

def get_all_traces_with_spans(project_name):
    query = """
    SELECT 
        trace_rowid,
        span_id,
        parent_id,
        name,
        EXTRACT(EPOCH FROM (end_time - start_time)) AS latency 
    FROM spans 
    WHERE trace_rowid IN (
        SELECT id FROM traces 
        WHERE project_rowid IN (
            SELECT id FROM projects WHERE name = %s
    ***REMOVED***
***REMOVED***
    ORDER BY trace_rowid, start_time;
    """
    with get_connection() as conn:
        return pd.read_sql(query, conn, params=(project_name,))
    
def get_root_spans(project_name):
    query = """
    SELECT s.trace_rowid, s.span_id, s.name, s.start_time
    FROM spans s
    JOIN traces t ON s.trace_rowid = t.id
    WHERE s.parent_id IS NULL
      AND t.project_rowid = (
        SELECT id FROM projects WHERE name = %s
  ***REMOVED***
    ORDER BY s.start_time DESC
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=(project_name,))
    return df

def get_span_tree_for_trace(trace_id):
    query = """
    SELECT span_id, parent_id, name,
           EXTRACT(EPOCH FROM (end_time - start_time)) AS latency
    FROM spans
    WHERE trace_rowid = %s
    ORDER BY start_time
    """
    with get_connection() as conn:
        return pd.read_sql(query, conn, params=(trace_id,))
    
def get_expensive_spans(project_name):
    query = """
    SELECT
        s.span_id,
        s.name,
        EXTRACT(EPOCH FROM (s.end_time - s.start_time)) AS latency,
        s.llm_token_count_prompt AS prompt_tokens,
        s.llm_token_count_completion AS completion_tokens,
        (COALESCE(s.llm_token_count_prompt, 0) + COALESCE(s.llm_token_count_completion, 0)) AS total_tokens
    FROM spans s
    JOIN traces t ON s.trace_rowid = t.id
    WHERE t.project_rowid = (
        SELECT id FROM projects WHERE name = %s
***REMOVED***
    ORDER BY total_tokens DESC, latency DESC
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=(project_name,))
    return df

def paginate_dataframe(df, label="Data", rows_per_page=20):
    total_rows = len(df)
    total_pages = (total_rows - 1) // rows_per_page + 1

    if total_rows == 0:
        st.warning(f"No {label.lower()} to display.")
        return

    page = st.number_input(f"{label} Page", min_value=1, max_value=total_pages, value=1, step=1)
    start_idx = (page - 1) * rows_per_page
    end_idx = start_idx + rows_per_page
    st.write(f"Showing {label.lower()} rows {start_idx + 1} to {min(end_idx, total_rows)} of {total_rows}")
    st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True)

# --- STREAMLIT APP UI ---
st.title("📈 LLM Observability Dashboard")

# Load dropdown values
options = get_dropdown_options()
selected = st.selectbox("Select Project:", options)
selected='news-app'
col1, col2, col3, col4 = st.columns(4)

with col1:
    value1 = get_tile1_value(selected)
    st.metric(label="Total LLM Calls", value=value1)

with col2:
    value2 = get_tile2_value(selected)
    st.metric(label="Avg Latency", value=value2)

with col3:
    value3 = get_tile3_value(selected)
    st.metric(label="Prompt token count", value=value3)

with col4:
    value4 = get_tile4_value(selected)
    st.metric(label="Completion token count", value=value4)


# --- GET AND PLOT EACH METRIC ---
import streamlit as st
import pandas as pd
import plotly.express as px

def plot_line(df, y_col, title):
    if df.empty:
        st.warning(f"No data available for {title}.")
        return

    # Ensure 'hour' is a datetime column
    df['hour'] = pd.to_datetime(df['hour'])

    # Create full hourly range from min to max hour
    full_hours = pd.DataFrame({'hour': pd.date_range(start=df['hour'].min(), end=df['hour'].max(), freq='H')})

    # Merge to fill missing hours
    df_full = pd.merge(full_hours, df, on='hour', how='left')
    df_full[y_col] = df_full[y_col].fillna(0)

    # Plot the result
    fig = px.line(df_full, x='hour', y=y_col, title=title)
    fig.update_layout(xaxis_title="Hour", yaxis_title=y_col)
    st.plotly_chart(fig, use_container_width=True)

# --- GET AND PLOT EACH METRIC ---
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
        text=y_col  # shows value on top of each bar
***REMOVED***
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Calls",
        yaxis=dict(tickformat="d"),
        bargap=0.2
***REMOVED***
    st.plotly_chart(fig, use_container_width=True)

# Get and plot each metric separately
df1 = get_metric1_data(selected)
plot_line(df1, 'count', "Traces over Time")

df2 = get_metric2_data(selected)
plot_bar(df2, "llm_call_count", "llm_call_count")

# df3 = get_metric3_data(selected)
# plot_metric("Metric 3 Over Time", df3, "metric3")

# df4 = get_metric4_data(selected)
# plot_metric("Metric 4 Over Time", df4, "metric4")

# Tree for each trace

#1. Show Root Span Dropdown
st.subheader("🌳 Select Root Span (Top-Level Agent Call)")

df_roots = get_root_spans(selected)

if df_roots.empty:
    st.info("No root-level spans found.")
else:
    trace_options = {
        f"{row['name']} @ {row['start_time']}": (row['trace_rowid'], row['span_id'])
        for _, row in df_roots.iterrows()
    }

    selected_label = st.selectbox("Choose a Root Span", list(trace_options.keys()))
    selected_trace_id, selected_root_span_id = trace_options[selected_label]

    # 2. Fetch full span tree for selected trace
    df_trace = get_span_tree_for_trace(selected_trace_id)

    # 3. Visualize with Graphviz
    if df_trace.empty:
        st.warning("No spans found for selected trace.")
    else:
        def build_trace_tree(df_trace):
            dot = graphviz.Digraph()
            for _, row in df_trace.iterrows():
                label = f"{row['name']}\n{row['latency']:.2f}s"
                dot.node(row['span_id'], label)
                if pd.notna(row['parent_id']):
                    dot.edge(row['parent_id'], row['span_id'])
            return dot

        st.subheader(f"📡 Span Tree for: {selected_label}")
        tree = build_trace_tree(df_trace)
        st.graphviz_chart(tree)


# Fetch data
st.subheader("📊 Span Analysis")

# --- Get the full data ---
df_expensive = get_expensive_spans(selected)

if df_expensive.empty:
    st.info("No span data available.")
else:
    # =========================
    # SECTION 1 – TOKEN FILTER
    # =========================
    st.markdown("### 💰 Token Usage Spans")

    exclude_zero_tokens = st.checkbox("Exclude spans with 0 total tokens", value=False)
    token_df = df_expensive.copy()

    if exclude_zero_tokens:
        token_df = token_df[token_df["total_tokens"] > 0]

    if token_df.empty:
        st.warning("No spans with token usage available after filtering.")
    else:
        min_tokens = int(token_df["total_tokens"].min())
        max_tokens = int(token_df["total_tokens"].max())

        if min_tokens < max_tokens:
            token_range = st.slider(
                "Select Total Token Range",
                min_value=min_tokens,
                max_value=max_tokens,
                value=(min_tokens, max_tokens),
                step=50
        ***REMOVED***
            token_df = token_df[
                token_df["total_tokens"].between(*token_range)
            ]

        st.dataframe(
            token_df.sort_values(by="total_tokens", ascending=False).head(20),
            use_container_width=True
    ***REMOVED***

    # =========================
    # SECTION 2 – LATENCY FILTER
    # =========================
    st.markdown("### 🕒 High Latency Spans")

    latency_df = df_expensive.copy()
    non_zero_latency = latency_df[latency_df["latency"] > 0]

    if non_zero_latency.empty:
        st.info("All spans have 0 latency. Skipping latency range filter.")
        st.dataframe(latency_df.head(20), use_container_width=True)
    else:
        min_latency = float(non_zero_latency["latency"].min())
        max_latency = float(non_zero_latency["latency"].max())

        latency_range = st.slider(
            "Select Latency Range (seconds)",
            min_value=min_latency,
            max_value=max_latency,
            value=(min_latency, max_latency),
            step=0.1
    ***REMOVED***

        latency_df = latency_df[
            latency_df["latency"].between(*latency_range)
        ]

        st.dataframe(
            latency_df.sort_values(by="latency", ascending=False).head(20),
            use_container_width=True
    ***REMOVED***
