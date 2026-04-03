import joblib
import streamlit as st
import pandas as pd
import openpyxl
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
import plotly.graph_objects as go
import streamlit.components.v1 as components

st.set_page_config(page_title="CubeSat Diagnostic Tool", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .anomaly-card {
        background: linear-gradient(135deg, #1a1f2e, #16213e);
        border: 1px solid #30363d;
        border-left: 4px solid #f85149;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .anomaly-card.high { border-left-color: #f85149; }
    .anomaly-card.medium { border-left-color: #d29922; }
    .anomaly-card.low { border-left-color: #3fb950; }
    .score-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 13px;
        font-weight: bold;
        float: right;
    }
    .score-high { background: #3d1a1a; color: #f85149; }
    .score-medium { background: #2d2208; color: #d29922; }
    .score-low { background: #0d2818; color: #3fb950; }
    .col-tag {
        display: inline-block;
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 4px;
        padding: 2px 8px;
        margin: 2px;
        font-size: 12px;
        color: #8b949e;
    }
    .section-header {
        font-size: 14px;
        font-weight: 600;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 24px 0 12px 0;
        border-bottom: 1px solid #21262d;
        padding-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛰 CubeSat Diagnostic Tool")
st.write("Upload satellite telemetry data to detect anomalies.")

df = pd.DataFrame({'Column 1': [1, 2, 3, 4]})

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.xlsx'):
        return pd.read_excel(uploaded_file)
    else:
        return pd.read_csv(uploaded_file)

def compute_features_for_column(values, i, window_size, col):
    features = {}
    features['window_start'] = i
    features['window_end'] = i + window_size
    features['column'] = col
    features['mean'] = np.mean(values)
    features['var'] = np.var(values)
    features['std'] = np.std(values)
    features['kurtosis'] = stats.kurtosis(values)
    features['skew'] = stats.skew(values)
    peaks, _ = find_peaks(values)
    features['n_peaks'] = len(peaks)
    features['smooth10_n_peaks'] = len(find_peaks(np.convolve(values, np.ones(10)/10, mode='valid'))[0])
    features['smooth20_n_peaks'] = len(find_peaks(np.convolve(values, np.ones(20)/20, mode='valid'))[0])
    diff1 = np.diff(values)
    diff2 = np.diff(diff1)
    features['diff_peaks'] = len(find_peaks(diff1)[0])
    features['diff2_peaks'] = len(find_peaks(diff2)[0])
    features['diff_var'] = np.var(diff1)
    features['diff2_var'] = np.var(diff2)
    features['gaps_squared'] = 0
    features['len_weighted'] = len(values)
    features['var_div_duration'] = features['var'] / max(len(values), 1)
    features['sampling'] = 1
    features['duration'] = len(values)
    features['len'] = len(values)
    features['var_div_len'] = features['var'] / max(len(values), 1)
    return features

def compute_features_per_column(df, window_size=500):
    exclude_cols = [
        # Timestamp components
        'mes', 'dia', 'hora', 'year', 'segundo', 'minuto',
        # Orbital position and velocity — predictable orbital motion, not health signals
        'x_ecef', 'y_ecef', 'z_ecef',
        'x_eci', 'y_eci', 'z_eci',
        'vx_eci', 'vy_eci', 'vz_eci',
        'longitude', 'latitude', 'altitude',
        # Counters — always increasing, not health signals
        'package_counter'
    ]
    numeric_df = df.select_dtypes(include=[np.number])
    results = []
    for col in numeric_df.columns:
        if col.lower() in [e.lower() for e in exclude_cols]:
            continue
        col_values = numeric_df[col].dropna().values
        if col_values.std() < 1e-6:
            continue
        unique_vals = np.unique(col_values)
        if len(unique_vals) < max(3, len(col_values) * 0.05):
            continue
        col_values = (col_values - col_values.mean()) / col_values.std()
        for i in range(0, len(numeric_df), window_size):
            window_values = col_values[i:i+window_size]
            if len(window_values) < 10:
                continue
            features = compute_features_for_column(window_values, i, window_size, col)
            results.append(features)
    return pd.DataFrame(results)

def severity_label(score):
    if score >= 0.85:
        return "high", "🔴 High"
    elif score >= 0.70:
        return "medium", "🟡 Medium"
    else:
        return "low", "🟢 Low"

def plot_anomaly_window(df, plot_col, window_start, window_end, score):
    start = max(0, window_start - 250)
    end = min(len(df), window_end + 250)
    window_data = df[plot_col].iloc[start:end].reset_index(drop=True)

    # Use timestamps if available and valid, otherwise use row numbers
    timestamp_col = None
    for col in ['UTC_Timestamp', 'timestamp', 'time', 'Time', 'date', 'Date']:
        if col in df.columns:
            timestamp_col = col
            break

    if timestamp_col:
        try:
            raw = df[timestamp_col].iloc[start:end].reset_index(drop=True)
            
            # Handle Quetzal-1 format: "16:43:55 - 28/04/2020"
            def parse_quetzal_timestamp(ts):
                try:
                    parts = str(ts).split(' - ')
                    if len(parts) == 2:
                        return pd.to_datetime(parts[1] + ' ' + parts[0], dayfirst=True)
                    return pd.to_datetime(ts, dayfirst=True)
                except:
                    return pd.NaT

            timestamps = raw.apply(parse_quetzal_timestamp)
            
            if timestamps.notna().sum() > len(timestamps) * 0.5:
                x = timestamps
                x_label = "Time"
            else:
                x = list(range(len(window_data)))
                x_label = f"Row offset from {start:,}"
        except:
            x = list(range(len(window_data)))
            x_label = f"Row offset from {start:,}"
    else:
        x = list(range(len(window_data)))
        x_label = f"Row offset from {start:,}"

    severity_class, _ = severity_label(score)
    shade_color = {
        "high": "rgba(248, 81, 73, 0.15)",
        "medium": "rgba(210, 153, 34, 0.15)",
        "low": "rgba(63, 185, 80, 0.15)"
    }[severity_class]
    border_color = {
        "high": "#f85149",
        "medium": "#d29922",
        "low": "#3fb950"
    }[severity_class]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=window_data,
        mode='lines',
        line=dict(color='#58a6ff', width=1.5),
        name=plot_col
    ))

    # Anomaly region markers
    anomaly_x_start = x.iloc[window_start - start] if hasattr(x, 'iloc') else window_start - start
    anomaly_x_end = x.iloc[min(window_end - start, len(x) - 1)] if hasattr(x, 'iloc') else min(window_end - start, len(window_data) - 1)

    fig.add_vrect(
        x0=anomaly_x_start,
        x1=anomaly_x_end,
        fillcolor=shade_color,
        layer="below",
        line=dict(color=border_color, width=1, dash="dash"),
        annotation_text="Anomaly Window",
        annotation_position="top left",
        annotation=dict(font=dict(color=border_color, size=12))
    )

    fig.update_layout(
        paper_bgcolor='#0d1117',
        plot_bgcolor='#161b22',
        font=dict(color='#8b949e'),
        xaxis=dict(
            title=x_label,
            gridcolor='#21262d',
            showgrid=True
        ),
        yaxis=dict(
            title=plot_col,
            gridcolor='#21262d',
            showgrid=True
        ),
        margin=dict(l=20, r=20, t=40, b=40),
        height=400,
        showlegend=False
    )

    return fig

uploaded_file = st.file_uploader("Choose a file (.csv or .xlsx)")
if uploaded_file is not None:
    df = load_data(uploaded_file)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.success(f"File uploaded: **{uploaded_file.name}**")
    with col2:
        st.caption(f"{len(df):,} rows · {len(df.columns)} columns")
    with st.expander("Preview data"):
        st.dataframe(df.head(10))
else:
    st.info("Please upload a telemetry file to proceed.")

model = joblib.load('model/model.pkl')

if st.button("🔍 Run Anomaly Detection", type="primary"):
    drop_cols = ['segment', 'anomaly', 'train', 'channel']
    existing_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=existing_drop)
    model_features = ['mean', 'var', 'std', 'kurtosis', 'skew']

    if not any(col in X.columns for col in model_features):
        progress = st.progress(0, text="Analyzing columns...")
        X = compute_features_per_column(df)
        if X.empty:
            st.error("No columns passed the quality filters. Your data may have too many constant or binary columns. Try uploading a file with more varied sensor readings.")
            st.stop()
        progress.progress(50, text="Running model...")
        meta = X[['window_start', 'window_end', 'column']].copy()
        X_model = X.drop(columns=['window_start', 'window_end', 'column'])
        X_model = X_model[model.feature_names_in_]
        proba = model.predict_proba(X_model)[:, 1]
        meta['anomaly_score'] = proba
        flagged_rows = []
        for col_name in meta['column'].unique():
            col_mask = meta['column'] == col_name
            col_data = meta[col_mask].copy()
            
            # Fixed threshold instead of relative percentile
            col_data['anomaly_detected'] = col_data['anomaly_score'] >= 0.65
            flagged_rows.append(col_data[col_data['anomaly_detected']])
        anomaly_rows = pd.concat(flagged_rows).reset_index(drop=True)
        anomaly_summary = (
            anomaly_rows.groupby(['window_start', 'window_end'])
            .agg(
                anomalous_columns=('column', lambda cols: ', '.join(cols)),
                avg_score=('anomaly_score', 'mean')
            )
            .reset_index()
            .sort_values('avg_score', ascending=False)
            .reset_index(drop=True)
        )
        progress.progress(100, text="Done!")
        st.session_state['anomaly_summary'] = anomaly_summary
        st.session_state['anomaly_count'] = len(anomaly_summary)
        st.session_state['total_windows'] = len(meta['window_start'].unique())
        st.session_state['df'] = df
        st.session_state['numeric_cols'] = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        predictions = model.predict(X)
        df['anomaly_detected'] = predictions
        anomaly_count = (predictions == 1).sum()
        st.write(f"Detection complete — {anomaly_count} anomalies found.")
        st.dataframe(df)

if 'anomaly_summary' in st.session_state:
    anomaly_summary = st.session_state['anomaly_summary']
    anomaly_count = st.session_state['anomaly_count']
    total_windows = st.session_state['total_windows']
    df = st.session_state['df']
    numeric_cols = st.session_state['numeric_cols']

    high_count = len(anomaly_summary[anomaly_summary['avg_score'] >= 0.85])
    medium_count = len(anomaly_summary[(anomaly_summary['avg_score'] >= 0.70) & (anomaly_summary['avg_score'] < 0.85)])

    st.markdown('<div class="section-header">Detection Results</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Windows Scanned", f"{total_windows:,}")
    with col2:
        st.metric("Anomalous Windows", f"{anomaly_count:,}")
    with col3:
        st.metric("🔴 High Severity", f"{high_count}")
    with col4:
        st.metric("🟡 Medium Severity", f"{medium_count}")

    st.markdown('<div class="section-header">Anomaly List — sorted by severity</div>', unsafe_allow_html=True)

    cards_html = ""
    for _, row in anomaly_summary.iterrows():
        severity_class, severity_text = severity_label(row['avg_score'])
        cols_list = [c.strip() for c in row['anomalous_columns'].split(',')]
        col_tags = ''.join([f'<span style="display:inline-block;background:#21262d;border:1px solid #30363d;border-radius:4px;padding:2px 8px;margin:2px;font-size:12px;color:#8b949e;">{c}</span>' for c in cols_list])
        
        border_color = {"high": "#f85149", "medium": "#d29922", "low": "#3fb950"}[severity_class]
        score_bg = {"high": "#3d1a1a", "medium": "#2d2208", "low": "#0d2818"}[severity_class]
        score_color = {"high": "#f85149", "medium": "#d29922", "low": "#3fb950"}[severity_class]

        cards_html += f"""
        <div style="background:linear-gradient(135deg,#1a1f2e,#16213e);border:1px solid #30363d;border-left:4px solid {border_color};border-radius:8px;padding:16px 20px;margin-bottom:12px;">
            <span style="display:inline-block;padding:2px 10px;border-radius:12px;font-size:13px;font-weight:bold;float:right;background:{score_bg};color:{score_color};">Score: {row['avg_score']:.3f}</span>
            <strong style="color:#e6edf3;">Rows {int(row['window_start']):,} – {int(row['window_end']):,}</strong>
            &nbsp;&nbsp;<span style="color:#8b949e;font-size:13px;">{severity_text}</span>
            <div style="margin-top:8px;">{col_tags}</div>
        </div>
        """

    components.html(f"""
    <div style="height:420px;overflow-y:auto;padding:12px;border:1px solid #21262d;border-radius:8px;background:#0d1117;font-family:sans-serif;">
        {cards_html}
    </div>
    """, height=440)

    st.markdown('<div class="section-header">Inspect a Window</div>', unsafe_allow_html=True)

    selected_idx = st.selectbox(
        "Select a window to investigate:",
        options=anomaly_summary.index.tolist(),
        format_func=lambda i: f"Rows {int(anomaly_summary.loc[i, 'window_start']):,}–{int(anomaly_summary.loc[i, 'window_end']):,}  |  Score {anomaly_summary.loc[i, 'avg_score']:.3f}  |  {', '.join(anomaly_summary.loc[i, 'anomalous_columns'].split(',')[:3])}{'...' if len(anomaly_summary.loc[i, 'anomalous_columns'].split(',')) > 3 else ''}"
    )

    row = anomaly_summary.loc[selected_idx]
    flagged_cols = [c.strip() for c in row['anomalous_columns'].split(',')]

    plot_col = st.selectbox(
        "Select a column to plot:",
        options=flagged_cols + [c for c in numeric_cols if c not in flagged_cols],
        help="Flagged columns are listed first"
    )

    severity_class, severity_text = severity_label(row['avg_score'])
    st.markdown(f"""
    <div class="anomaly-card {severity_class}" style="margin-top:16px">
        <span class="score-badge score-{severity_class}">Score: {row['avg_score']:.3f}</span>
        <strong>{plot_col}</strong> &nbsp; {severity_text}<br>
        <span style="color:#8b949e; font-size:13px">
            Rows {int(row['window_start']):,} – {int(row['window_end']):,} &nbsp;·&nbsp; 
            Shaded region marks the anomaly window · 250 rows of context shown on each side
        </span>
    </div>
    """, unsafe_allow_html=True)

    fig = plot_anomaly_window(
        df=df,
        plot_col=plot_col,
        window_start=int(row['window_start']),
        window_end=int(row['window_end']),
        score=row['avg_score']
    )
    st.plotly_chart(fig, use_container_width=True)