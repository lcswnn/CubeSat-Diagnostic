"""
evaluate.py — Validate CubeSat Diagnostic anomaly detections

Usage:
    python evaluate.py <telemetry.csv> [--threshold 0.70] [--window 500]

What it does:
  1. Runs the same detection pipeline as app.py
  2. If ground-truth labels exist (e.g. OPSSAT-AD 'anomaly' column), computes
     precision, recall, F1 per channel and overall
  3. If no labels exist, runs statistical cross-validation:
     - Z-score spike detection as independent second opinion
     - Isolation Forest as unsupervised confirmation
     - Mahalanobis Distance as distribution-based validation
     - Cross-channel correlation analysis (do multiple channels agree?)
  4. Outputs a summary report to the terminal and saves detailed results to CSV
"""

import argparse
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import joblib

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Column classifier (same logic as app.py)
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


def classify_column(col_name, values):
    col_lower = col_name.lower()
    for cat, patterns in _NAME_PATTERNS.items():
        if col_lower in [p.lower() for p in patterns]:
            return cat
    clean = values[~np.isnan(values)]
    if len(clean) < 20:
        return 'constant'
    n_unique = len(np.unique(clean))
    if np.std(clean) < 1e-8:
        return 'constant'
    if n_unique <= 5:
        return 'binary'
    diffs = np.diff(clean)
    if np.mean(diffs >= 0) > 0.97 and n_unique > 50:
        return 'counter'
    if np.mean(diffs <= 0) > 0.97 and n_unique > 50:
        return 'counter'
    if n_unique <= 20 and (n_unique / len(clean)) < 0.005:
        return 'binary'
    return 'sensor'


def get_sensor_columns(df):
    numeric_df = df.select_dtypes(include=[np.number])
    sensor_cols = []
    excluded = {}
    for col in numeric_df.columns:
        cat = classify_column(col, numeric_df[col].values)
        if cat == 'sensor':
            sensor_cols.append(col)
        else:
            excluded[col] = cat
    return sensor_cols, excluded


# ---------------------------------------------------------------------------
# Feature computation (same as app.py)
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


def compute_features(df, sensor_cols, window_size=500):
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
            window_values = col_values[i:i + window_size]
            if len(window_values) < 10:
                continue
            features = compute_features_for_column(window_values, i, window_size, col)
            results.append(features)
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Statistical cross-validation (for unlabeled data)
# ---------------------------------------------------------------------------

def zscore_validate(df, col, window_start, window_end, context_rows=2000):
    """Check if the flagged window is statistically unusual vs surrounding data."""
    values = df[col].dropna().values
    ctx_start = max(0, window_start - context_rows)
    ctx_end = min(len(values), window_end + context_rows)

    context = np.concatenate([values[ctx_start:window_start], values[window_end:ctx_end]])
    window = values[window_start:window_end]

    if len(context) < 20 or len(window) < 10:
        return None, 'insufficient data'

    ctx_mean = np.mean(context)
    ctx_std = np.std(context)
    if ctx_std < 1e-8:
        return None, 'context has zero variance'

    window_mean = np.mean(window)
    window_std = np.std(window)

    # Mean shift z-score
    mean_z = abs(window_mean - ctx_mean) / (ctx_std / np.sqrt(len(window)))

    # Variance ratio (F-test style)
    var_ratio = window_std / (ctx_std + 1e-8)

    confirmed = mean_z > 3.0 or var_ratio > 2.0 or var_ratio < 0.5
    reason = f'mean_z={mean_z:.1f}, var_ratio={var_ratio:.2f}'
    return confirmed, reason


def isolation_forest_validate(df, sensor_cols, window_size=500):
    """Run Isolation Forest on the same full feature set the RF uses.
    Returns a dict of (window_start, col) -> anomaly_score.
    """
    feat_df = compute_features(df, sensor_cols, window_size)
    if feat_df.empty:
        return {}

    keys = list(zip(feat_df['window_start'].astype(int), feat_df['column']))
    X_if = feat_df.drop(columns=['window_start', 'window_end', 'column'])
    X_if = add_derived_features(X_if)

    iso = IsolationForest(contamination=0.02, random_state=42, n_estimators=200)
    iso.fit(X_if)
    scores = iso.decision_function(X_if)
    # Lower score = more anomalous; convert to 0-1 where 1 = most anomalous
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s > 0:
        norm_scores = 1 - (scores - min_s) / (max_s - min_s)
    else:
        norm_scores = np.zeros(len(scores))

    return {k: s for k, s in zip(keys, norm_scores)}


def mahalanobis_validate(df, sensor_cols, window_size=500):
    """Compute Mahalanobis distance for each window using training distribution.
    Returns a dict of (window_start, col) -> md_score (normalized 0-1).
    """
    feat_df = compute_features(df, sensor_cols, window_size)
    if feat_df.empty:
        return {}

    keys = list(zip(feat_df['window_start'].astype(int), feat_df['column']))
    X_md = feat_df.drop(columns=['window_start', 'window_end', 'column'])
    X_md = add_derived_features(X_md)

    # Compute mean and covariance from all windows (proxy for normal distribution)
    mean = X_md.mean().values
    cov = X_md.cov().values

    # Regularize covariance to avoid singular matrix
    cov += np.eye(cov.shape[0]) * 1e-6

    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    # Compute MD for each window
    diffs = X_md.values - mean
    md_scores = np.sqrt(np.sum(diffs @ cov_inv * diffs, axis=1))

    # Normalize to 0-1 range
    min_md, max_md = md_scores.min(), md_scores.max()
    if max_md - min_md > 0:
        norm_md = (md_scores - min_md) / (max_md - min_md)
    else:
        norm_md = np.zeros(len(md_scores))

    return {k: s for k, s in zip(keys, norm_md)}


def cross_channel_check(anomaly_summary):
    """Check how many channels flag the same window — multi-channel agreement
    is a strong signal of a real anomaly."""
    results = []
    for _, row in anomaly_summary.iterrows():
        cols = [c.strip() for c in row['anomalous_columns'].split(',')]
        results.append({
            'window_start': int(row['window_start']),
            'window_end': int(row['window_end']),
            'n_channels': len(cols),
            'channels': ', '.join(cols),
            'score': row['avg_score'],
            'multi_channel': len(cols) > 1
        })
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Evaluate CubeSat anomaly detections')
    parser.add_argument('csv_file', help='Path to telemetry CSV')
    parser.add_argument('--model', default='model/model.pkl', help='Path to model.pkl')
    parser.add_argument('--threshold', type=float, default=0.70, help='Anomaly score threshold')
    parser.add_argument('--window', type=int, default=500, help='Window size')
    parser.add_argument('--output', default='evaluation_report.csv', help='Output CSV path')
    args = parser.parse_args()

    print("=" * 70)
    print("  CubeSat Diagnostic — Anomaly Detection Evaluation")
    print("=" * 70)

    # Load data
    print(f"\n📂 Loading {args.csv_file}...")
    df = pd.read_csv(args.csv_file)
    print(f"   {len(df):,} rows × {len(df.columns)} columns")

    # Classify columns
    sensor_cols, excluded = get_sensor_columns(df)
    print(f"\n📡 Column classification:")
    print(f"   {len(sensor_cols)} sensor channels → will analyze")
    print(f"   {len(excluded)} excluded:")
    for col, cat in excluded.items():
        print(f"      {cat:>10}  {col}")

    # Load model
    print(f"\n🤖 Loading model from {args.model}...")
    model = joblib.load(args.model)

    # Compute features and run model
    print(f"\n⚙️  Computing features (window={args.window})...")
    X = compute_features(df, sensor_cols, args.window)
    if X.empty:
        print("   ❌ No features computed. Check your data.")
        sys.exit(1)
    print(f"   {len(X):,} windows across {X['column'].nunique()} channels")

    meta = X[['window_start', 'window_end', 'column']].copy()
    X_model = X.drop(columns=['window_start', 'window_end', 'column'])
    X_model = add_derived_features(X_model)
    X_model = X_model[model.feature_names_in_]

    proba = model.predict_proba(X_model)[:, 1]
    meta['rf_score'] = proba
    meta['rf_flagged'] = proba >= args.threshold

    n_flagged = meta['rf_flagged'].sum()
    n_total = len(meta)
    print(f"\n🔍 Random Forest results:")
    print(f"   {n_flagged}/{n_total} windows flagged ({n_flagged/n_total*100:.1f}%)")

    # Build anomaly summary
    flagged = meta[meta['rf_flagged']].copy()
    if not flagged.empty:
        anomaly_summary = (
            flagged.groupby(['window_start', 'window_end'])
            .agg(
                anomalous_columns=('column', lambda cols: ', '.join(cols)),
                avg_score=('rf_score', 'mean')
            )
            .reset_index()
            .sort_values('avg_score', ascending=False)
            .reset_index(drop=True)
        )
    else:
        anomaly_summary = pd.DataFrame(
            columns=['window_start', 'window_end', 'anomalous_columns', 'avg_score']
        )

    # -----------------------------------------------------------------------
    # Check for ground-truth labels
    # -----------------------------------------------------------------------
    has_labels = 'anomaly' in df.columns and 'train' in df.columns

    if has_labels:
        print("\n" + "=" * 70)
        print("  GROUND-TRUTH EVALUATION (labeled dataset detected)")
        print("=" * 70)

        # The OPSSAT-AD dataset has pre-computed features with anomaly labels
        drop_cols = ['segment', 'anomaly', 'train', 'channel']
        existing = [c for c in drop_cols if c in df.columns]
        X_gt = df.drop(columns=existing)
        y_gt = df['anomaly']

        X_gt = add_derived_features(X_gt)
        X_gt = X_gt[model.feature_names_in_]

        preds = model.predict(X_gt)
        pred_proba = model.predict_proba(X_gt)[:, 1]

        print(f"\n📊 Classification Report:")
        print(classification_report(y_gt, preds, target_names=['Normal', 'Anomaly']))

        cm = confusion_matrix(y_gt, preds)
        tn, fp, fn, tp = cm.ravel()
        print(f"   Confusion Matrix:")
        print(f"   TP={tp}  FP={fp}")
        print(f"   FN={fn}  TN={tn}")
        print(f"\n   False positive rate: {fp/(fp+tn)*100:.1f}%")
        print(f"   False negative rate: {fn/(fn+tp)*100:.1f}%")

        # Per-channel breakdown if channel info exists
        if 'channel' in df.columns:
            print(f"\n📡 Per-channel performance:")
            for ch in df['channel'].unique():
                mask = df['channel'] == ch
                ch_preds = preds[mask]
                ch_true = y_gt[mask]
                ch_tp = ((ch_preds == 1) & (ch_true == 1)).sum()
                ch_fp = ((ch_preds == 1) & (ch_true == 0)).sum()
                ch_fn = ((ch_preds == 0) & (ch_true == 1)).sum()
                ch_f1 = 2 * ch_tp / (2 * ch_tp + ch_fp + ch_fn + 1e-8)
                print(f"   {ch:>30}  F1={ch_f1:.3f}  TP={ch_tp} FP={ch_fp} FN={ch_fn}")

    else:
        print("\n" + "=" * 70)
        print("  STATISTICAL CROSS-VALIDATION (no ground-truth labels)")
        print("=" * 70)

        if anomaly_summary.empty:
            print("\n   ✅ No anomalies detected — nothing to validate.")
            return

        # 1. Z-score validation
        print(f"\n📈 Z-score validation (mean shift & variance change):")
        zscore_results = []
        for _, row in flagged.iterrows():
            confirmed, reason = zscore_validate(
                df, row['column'],
                int(row['window_start']), int(row['window_end'])
            )
            zscore_results.append({
                'window_start': int(row['window_start']),
                'column': row['column'],
                'rf_score': row['rf_score'],
                'zscore_confirmed': confirmed,
                'zscore_detail': reason
            })
        zdf = pd.DataFrame(zscore_results)
        if not zdf.empty and 'zscore_confirmed' in zdf.columns:
            confirmed_count = zdf['zscore_confirmed'].sum()
            total_count = len(zdf)
            print(f"   {confirmed_count}/{total_count} flagged windows confirmed by z-score")
            print(f"   Agreement rate: {confirmed_count/total_count*100:.1f}%")

        # 2. Isolation Forest cross-validation
        print(f"\n🌲 Isolation Forest cross-validation:")
        iso_scores = isolation_forest_validate(df, sensor_cols, args.window)
        if iso_scores:
            iso_agreements = 0
            iso_total = 0
            for _, row in flagged.iterrows():
                key = (int(row['window_start']), row['column'])
                if key in iso_scores:
                    iso_total += 1
                    if iso_scores[key] > 0.6:
                        iso_agreements += 1
            if iso_total > 0:
                print(f"   {iso_agreements}/{iso_total} RF-flagged windows also flagged by Isolation Forest")
                print(f"   Agreement rate: {iso_agreements/iso_total*100:.1f}%")

        # 3. Mahalanobis Distance cross-validation
        print(f"\n📐 Mahalanobis Distance cross-validation:")
        md_scores = mahalanobis_validate(df, sensor_cols, args.window)
        md_agreements = 0
        md_total = 0
        if md_scores:
            for _, row in flagged.iterrows():
                key = (int(row['window_start']), row['column'])
                if key in md_scores:
                    md_total += 1
                    if md_scores[key] > 0.6:
                        md_agreements += 1
            if md_total > 0:
                print(f"   {md_agreements}/{md_total} RF-flagged windows also flagged by Mahalanobis Distance")
                print(f"   Agreement rate: {md_agreements/md_total*100:.1f}%")
            else:
                print(f"   No overlapping windows to compare")
        else:
            print(f"   Could not compute MD scores")

        # 4. Cross-channel analysis
        print(f"\n🔗 Cross-channel analysis:")
        cc = cross_channel_check(anomaly_summary)
        if not cc.empty:
            multi = cc[cc['multi_channel']]
            print(f"   {len(cc)} anomalous windows total")
            print(f"   {len(multi)} windows flagged across multiple channels (stronger signal)")
            if not multi.empty:
                print(f"\n   Multi-channel anomalies:")
                for _, r in multi.iterrows():
                    print(f"      Rows {r['window_start']:>6,}–{r['window_end']:>6,}  "
                          f"Score {r['score']:.3f}  "
                          f"Channels: {r['channels']}")

        # 5. Build detailed report
        print(f"\n" + "=" * 70)
        print(f"  DETECTION SUMMARY")
        print(f"=" * 70)

        # Summary by severity
        high = anomaly_summary[anomaly_summary['avg_score'] >= 0.85]
        medium = anomaly_summary[(anomaly_summary['avg_score'] >= 0.70) & (anomaly_summary['avg_score'] < 0.85)]
        low = anomaly_summary[anomaly_summary['avg_score'] < 0.70]
        print(f"\n   🔴 High severity:   {len(high)}")
        print(f"   🟡 Medium severity: {len(medium)}")
        print(f"   🟢 Low severity:    {len(low)}")

        # Confidence assessment
        if not zdf.empty and 'zscore_confirmed' in zdf.columns:
            agreement = zdf['zscore_confirmed'].mean()
            if agreement > 0.8:
                verdict = "HIGH confidence — most detections confirmed by independent methods"
            elif agreement > 0.5:
                verdict = "MODERATE confidence — majority confirmed, some may be false positives"
            else:
                verdict = "LOW confidence — many detections not confirmed, consider raising threshold"
            print(f"\n   📋 Overall verdict: {verdict}")
            print(f"      Z-score agreement: {agreement*100:.0f}%")
            if iso_total > 0:
                print(f"      Isolation Forest agreement: {iso_agreements/iso_total*100:.0f}%")
            if not cc.empty:
                print(f"      Multi-channel rate: {len(multi)/len(cc)*100:.0f}%")
            if md_total > 0:
                print(f"      Mahalanobis Distance agreement: {md_agreements/md_total*100:.0f}%")

    # Save detailed results
    if 'zdf' not in locals():
        zdf = pd.DataFrame()
    if 'iso_scores' not in locals():
        iso_scores = {}
    if 'md_scores' not in locals():
        md_scores = {}
    detail_rows = []
    for _, row in meta.iterrows():
        r = {
            'window_start': int(row['window_start']),
            'window_end': int(row['window_end']),
            'column': row['column'],
            'rf_score': row['rf_score'],
            'rf_flagged': row['rf_flagged'],
        }
        # Add z-score if available
        if not has_labels and not zdf.empty:
            match = zdf[(zdf['window_start'] == r['window_start']) & (zdf['column'] == r['column'])]
            if not match.empty:
                r['zscore_confirmed'] = match.iloc[0].get('zscore_confirmed', None)
                r['zscore_detail'] = match.iloc[0].get('zscore_detail', '')
        # Add isolation forest if available
        if not has_labels and iso_scores:
            key = (r['window_start'], r['column'])
            r['iso_score'] = iso_scores.get(key, None)
        # Add mahalanobis distance if available
        if not has_labels and md_scores:
            key = (r['window_start'], r['column'])
            r['md_score'] = md_scores.get(key, None)
        detail_rows.append(r)

    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(args.output, index=False)
    print(f"\n💾 Detailed results saved to {args.output}")
    print(f"   {len(detail_df)} total windows evaluated")
    print()


if __name__ == '__main__':
    main() 