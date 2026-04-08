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
    .block-container { max-width: 1400px; padding-left: 3rem; padding-right: 3rem; }
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
    /* Constrain column filter expander to scrollable container */
    [data-testid="stExpander"] details > div[data-testid="stExpanderDetails"] {
        max-height: 400px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; font-size: 4rem;'>CubeSat Diagnostic Tool</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e; font-size: 1.2rem;'>Upload satellite telemetry data to detect anomalies.</p>", unsafe_allow_html=True)

df = pd.DataFrame({'Column 1': [1, 2, 3, 4]})

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.xlsx'):
        return pd.read_excel(uploaded_file)
    else:
        return pd.read_csv(uploaded_file)

# ---------------------------------------------------------------------------
# Dynamic Column Classifier
# ---------------------------------------------------------------------------
# Instead of a hardcoded exclusion list, we analyze each numeric column and
# classify it into one of several categories. Only "sensor" columns are fed
# to the anomaly detector. Users can override any classification.
# ---------------------------------------------------------------------------

_NAME_PATTERNS = {
    'position': ['x_ecef', 'y_ecef', 'z_ecef', 'x_eci', 'y_eci', 'z_eci',
                  'vx_eci', 'vy_eci', 'vz_eci', 'longitude', 'latitude', 'altitude',
                  'gyro_x', 'gyro_y', 'gyro_z',
                  'mag_x', 'mag_y', 'mag_z'],
    'timestamp': ['mes', 'dia', 'hora', 'year', 'segundo', 'minuto'],
    'counter':   ['package_counter'],
    'binary':    ['state_charge', 'state_health',
                  'adm_status', 'eps_status', 'heater_status',
                  'adcs_status', 'payload_status', 'camera_mode'],
}

def _classify_column(col_name, values):
    """Return (category, reason) for a single numeric column.

    Categories
    ----------
    sensor    – continuous health/telemetry signal  -> analyze
    binary    – on/off command or status flag       -> exclude
    counter   – monotonically increasing counter    -> exclude
    position  – orbital position / velocity         -> exclude
    timestamp – time component                      -> exclude
    constant  – no variation                        -> exclude
    """
    col_lower = col_name.lower()

    # --- Name-based checks (fast, high confidence) ---
    for cat, patterns in _NAME_PATTERNS.items():
        if col_lower in [p.lower() for p in patterns]:
            return cat, f"Known {cat} column"

    # --- Statistical checks ---
    clean = values[~np.isnan(values)]
    if len(clean) < 20:
        return 'constant', 'Too few non-null values'

    n_unique = len(np.unique(clean))
    std = np.std(clean)

    # Constant
    if std < 1e-8:
        return 'constant', 'Zero variance'

    # Binary / near-binary (<=5 distinct values)
    if n_unique <= 5:
        return 'binary', f'Only {n_unique} unique values — likely status/command flag'

    # Monotonic counter detection
    diffs = np.diff(clean)
    pct_nonneg = np.mean(diffs >= 0)
    pct_nonpos = np.mean(diffs <= 0)
    if pct_nonneg > 0.97 and n_unique > 50:
        return 'counter', f'{pct_nonneg:.0%} non-decreasing — monotonic counter'
    if pct_nonpos > 0.97 and n_unique > 50:
        return 'counter', f'{pct_nonpos:.0%} non-increasing — monotonic counter'

    # Low-entropy discrete signal (e.g. status codes with 6-20 levels)
    unique_ratio = n_unique / len(clean)
    if n_unique <= 20 and unique_ratio < 0.005:
        return 'binary', f'{n_unique} unique values across {len(clean):,} rows — discrete status'

    # Everything else is a sensor
    return 'sensor', 'Continuous telemetry signal'


def classify_columns(df):
    """Classify every numeric column. Returns DataFrame with:
       column, category, reason
    """
    numeric_df = df.select_dtypes(include=[np.number])
    rows = []
    for col in numeric_df.columns:
        cat, reason = _classify_column(col, numeric_df[col].values)
        rows.append({'column': col, 'category': cat, 'reason': reason})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

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


def compute_features_per_column(df, sensor_cols, window_size=500):
    """Compute windowed features only for the given sensor columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    results = []
    for col in sensor_cols:
        if col not in numeric_df.columns:
            continue
        col_values = numeric_df[col].dropna().values
        if col_values.std() < 1e-6:
            continue
        col_values = (col_values - col_values.mean()) / col_values.std()
        for i in range(0, len(numeric_df), window_size):
            window_values = col_values[i:i+window_size]
            if len(window_values) < 10:
                continue
            features = compute_features_for_column(window_values, i, window_size, col)
            results.append(features)
    return pd.DataFrame(results)


def add_derived_features(frame):
    f = frame.copy()
    f['cv'] = f['std'] / (f['mean'].abs() + 1e-8)
    f['diff_var_ratio'] = f['diff_var'] / (f['var'] + 1e-8)
    f['diff2_var_ratio'] = f['diff2_var'] / (f['var'] + 1e-8)
    f['peak_density'] = f['n_peaks'] / (f['len'] + 1e-8)
    f['smooth10_peak_density'] = f['smooth10_n_peaks'] / (f['len'] + 1e-8)
    f['smooth20_peak_density'] = f['smooth20_n_peaks'] / (f['len'] + 1e-8)
    f['smooth10_ratio'] = f['smooth10_n_peaks'] / (f['n_peaks'] + 1e-8)
    f['smooth20_ratio'] = f['smooth20_n_peaks'] / (f['n_peaks'] + 1e-8)
    f['kurtosis_skew'] = f['kurtosis'] * f['skew']
    f['mean_var_ratio'] = f['mean'] / (f['var'] + 1e-8)
    return f


def severity_label(score):
    if score >= 0.85:
        return "high", "High"
    elif score >= 0.70:
        return "medium", "Medium"
    else:
        return "low", "Low"


def plot_anomaly_window(df, plot_col, window_start, window_end, score):
    start = max(0, window_start - 250)
    end = min(len(df), window_end + 250)
    window_data = df[plot_col].iloc[start:end].reset_index(drop=True)

    timestamp_col = None
    for col in ['UTC_Timestamp', 'timestamp', 'time', 'Time', 'date', 'Date']:
        if col in df.columns:
            timestamp_col = col
            break

    if timestamp_col:
        try:
            raw = df[timestamp_col].iloc[start:end].reset_index(drop=True)

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
        xaxis=dict(title=x_label, gridcolor='#21262d', showgrid=True),
        yaxis=dict(title=plot_col, gridcolor='#21262d', showgrid=True),
        margin=dict(l=20, r=20, t=40, b=40),
        height=400,
        showlegend=False
    )

    return fig


# ---------------------------------------------------------------------------
# UI — File Upload
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# UI — Column Classification (shown after upload, before detection)
# ---------------------------------------------------------------------------

if uploaded_file is not None:
    col_classes = classify_columns(df)

    sensor_cols = col_classes[col_classes['category'] == 'sensor']['column'].tolist()
    excluded = col_classes[col_classes['category'] != 'sensor']

    if 'col_overrides' not in st.session_state:
        st.session_state['col_overrides'] = {}

    with st.expander(f"Column filter — {len(sensor_cols)} sensor channels, {len(excluded)} auto-excluded", expanded=False):
        st.caption("Columns are automatically classified by statistical analysis. Override any column below.")

        for cat in ['binary', 'counter', 'position', 'timestamp', 'constant']:
            cat_cols = excluded[excluded['category'] == cat]
            if cat_cols.empty:
                continue
            st.markdown(f"**{cat.title()}** — {len(cat_cols)} columns excluded")
            for _, r in cat_cols.iterrows():
                c1, c2, c3 = st.columns([3, 5, 2])
                with c1:
                    st.code(r['column'], language=None)
                with c2:
                    st.caption(r['reason'])
                with c3:
                    if st.checkbox("Include", key=f"inc_{r['column']}"):
                        st.session_state['col_overrides'][r['column']] = 'sensor'

        st.markdown(f"**Sensor** — {len(sensor_cols)} columns included")
        exclude_choices = st.multiselect(
            "Exclude any sensor columns:",
            options=sensor_cols,
            default=[],
            key="manual_exclude"
        )
        for col in exclude_choices:
            st.session_state['col_overrides'][col] = 'excluded'

    # Build final sensor list with overrides applied
    final_sensor_cols = []
    for _, r in col_classes.iterrows():
        override = st.session_state['col_overrides'].get(r['column'])
        if override == 'sensor':
            final_sensor_cols.append(r['column'])
        elif override == 'excluded':
            continue
        elif r['category'] == 'sensor':
            final_sensor_cols.append(r['column'])

    st.session_state['final_sensor_cols'] = final_sensor_cols

# ---------------------------------------------------------------------------
# UI — Run Detection
# ---------------------------------------------------------------------------

if st.button("Run Anomaly Detection", type="primary"):
    sensor_cols = st.session_state.get('final_sensor_cols', [])
    if not sensor_cols:
        st.error("No sensor columns to analyze. Check the column filter above.")
        st.stop()

    drop_cols = ['segment', 'anomaly', 'train', 'channel']
    existing_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=existing_drop)
    model_features = ['mean', 'var', 'std', 'kurtosis', 'skew']

    if not any(col in X.columns for col in model_features):
        progress = st.progress(0, text=f"Analyzing {len(sensor_cols)} sensor channels...")
        X = compute_features_per_column(df, sensor_cols)
        if X.empty:
            st.error("No columns produced valid features. Try including more columns in the filter above.")
            st.stop()
        progress.progress(50, text="Running model...")
        meta = X[['window_start', 'window_end', 'column']].copy()
        X_model = X.drop(columns=['window_start', 'window_end', 'column'])
        X_model = add_derived_features(X_model)
        X_model = X_model[model.feature_names_in_]
        proba = model.predict_proba(X_model)[:, 1]
        meta['anomaly_score'] = proba
        flagged_rows = []
        for col_name in meta['column'].unique():
            col_mask = meta['column'] == col_name
            col_data = meta[col_mask].copy()
            col_data['anomaly_detected'] = col_data['anomaly_score'] >= 0.70
            flagged_rows.append(col_data[col_data['anomaly_detected']])
        if flagged_rows:
            anomaly_rows = pd.concat(flagged_rows).reset_index(drop=True)
        else:
            anomaly_rows = pd.DataFrame(columns=meta.columns)
        if not anomaly_rows.empty:
            # Sort columns by their individual scores (highest first)
            anomaly_rows = anomaly_rows.sort_values('anomaly_score', ascending=False)
            anomaly_summary = (
                anomaly_rows.groupby(['window_start', 'window_end'])
                .agg(
                    anomalous_columns=('column', lambda cols: ', '.join(cols)),
                    column_scores=('anomaly_score', lambda scores: ', '.join(f'{s:.3f}' for s in scores)),
                    avg_score=('anomaly_score', 'mean')
                )
                .reset_index()
                .sort_values('avg_score', ascending=False)
                .reset_index(drop=True)
            )
        else:
            anomaly_summary = pd.DataFrame(columns=['window_start', 'window_end', 'anomalous_columns', 'column_scores', 'avg_score'])
        progress.progress(100, text="Done!")
        st.session_state['anomaly_summary'] = anomaly_summary
        st.session_state['anomaly_count'] = len(anomaly_summary)
        st.session_state['total_windows'] = len(meta['window_start'].unique())
        st.session_state['df'] = df
        st.session_state['numeric_cols'] = sensor_cols
    else:
        predictions = model.predict(X)
        df['anomaly_detected'] = predictions
        anomaly_count = (predictions == 1).sum()
        st.write(f"Detection complete — {anomaly_count} anomalies found.")
        st.dataframe(df)

# ---------------------------------------------------------------------------
# UI — Results Display
# ---------------------------------------------------------------------------

if 'anomaly_summary' in st.session_state:
    anomaly_summary = st.session_state['anomaly_summary']
    anomaly_count = st.session_state['anomaly_count']
    total_windows = st.session_state['total_windows']
    df = st.session_state['df']
    numeric_cols = st.session_state['numeric_cols']

    high_count = len(anomaly_summary[anomaly_summary['avg_score'] >= 0.85]) if not anomaly_summary.empty else 0
    medium_count = len(anomaly_summary[(anomaly_summary['avg_score'] >= 0.70) & (anomaly_summary['avg_score'] < 0.85)]) if not anomaly_summary.empty else 0

    st.markdown('<div class="section-header">Detection Results</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Windows Scanned", f"{total_windows:,}")
    with col2:
        st.metric("Anomalous Windows", f"{anomaly_count:,}")
    with col3:
        st.metric("High Severity", f"{high_count}")
    with col4:
        st.metric("Medium Severity", f"{medium_count}")

    if anomaly_summary.empty:
        st.success("No anomalies detected in this file.")
        st.stop()

    st.markdown('<div class="section-header">Anomaly List — sorted by severity</div>', unsafe_allow_html=True)

    cards_html = ""
    for _, row in anomaly_summary.iterrows():
        severity_class, severity_text = severity_label(row['avg_score'])
        cols_list = [c.strip() for c in row['anomalous_columns'].split(',')]
        scores_list = [float(s.strip()) for s in row.get('column_scores', '').split(',') if s.strip()] if 'column_scores' in row and pd.notna(row.get('column_scores')) else []

        # Build tags with opacity scaled by individual score
        tag_parts = []
        for idx, c in enumerate(cols_list):
            if idx < len(scores_list):
                score = scores_list[idx]
                opacity = 0.4 + 0.6 * min(score / 1.0, 1.0)
                score_text = f' ({score:.2f})'
            else:
                opacity = 0.6
                score_text = ''
            tag_parts.append(
                f'<span style="display:inline-block;background:#21262d;border:1px solid #30363d;'
                f'border-radius:4px;padding:2px 8px;margin:2px;font-size:12px;'
                f'color:rgba(139,148,158,{opacity:.2f});">{c}{score_text}</span>'
            )
        col_tags = ''.join(tag_parts)

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