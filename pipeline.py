"""
AthleteLoadDSS - end-to-end pipeline
====================================
Runs the whole project in one command, from the raw CSV to DSS recommendations:

    python pipeline.py                       # uses sports_performance_data.csv
    python pipeline.py --input new_week.csv  # run on new athlete data

Stages
  1. Ingest    - load raw data and report data-quality issues
  2. Clean     - duplicates, missing values, inconsistent columns
  3. Model     - train & compare 4 ML models, keep the best non-baseline model
  4. Decide    - apply the DSS (risk rules + decision model) to every record
  5. Export    - save clean data, model, config, recommendations, run report

The Streamlit dashboard (app.py) reads the files written in stage 5.
"""
import argparse, json, logging, time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dss_engine import ACTIONS, CAT_COLS, NUM_COLS, decide

RANDOM_STATE = 42
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('pipeline')


# ---------------------------------------------------------------- 1. INGEST
def ingest(path):
    df = pd.read_csv(path)
    quality = {
        'rows': len(df), 'columns': df.shape[1],
        'duplicates': int(df.duplicated().sum()),
        'missing_pct': (df.isna().mean() * 100).round(1).to_dict(),
    }
    log.info(f'[1/5 INGEST] {len(df):,} rows x {df.shape[1]} cols, {quality["duplicates"]} duplicates')
    return df, quality


# ---------------------------------------------------------------- 2. CLEAN
def clean(df):
    df = df.drop_duplicates().copy()
    df = df.dropna(subset=['Performance_Metric'])                 # target must be known
    df['Injury_History'] = df['Injury_History'].fillna('None')     # blank = no injury
    df['Altitude_Training'] = df['Altitude_Training'].fillna('None')
    df = df.drop(columns=[c for c in ['Event', 'Athlete_Name'] if c in df])  # inconsistent / not predictive
    df['Competition_Date'] = pd.to_datetime(df['Competition_Date'], errors='coerce')
    df['Sport_Type'] = df['Sport_Type'].fillna('Unknown')
    for c in NUM_COLS:                                             # median within the same sport
        df[c] = df.groupby('Sport_Type')[c].transform(lambda s: s.fillna(s.median()))
    df['Training_Intensity'] = df['Training_Intensity'].fillna(df['Training_Intensity'].mode()[0])
    remaining = int(df[NUM_COLS + CAT_COLS].isna().sum().sum())
    log.info(f'[2/5 CLEAN]  {len(df):,} clean rows, {remaining} missing values left')
    return df


# ---------------------------------------------------------------- 3. MODEL
def build_models(only=None):
    models = {
        'Baseline (mean)': DummyRegressor(),
        'Ridge regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=RANDOM_STATE),
    }
    try:
        from xgboost import XGBRegressor
        models['XGBoost'] = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=RANDOM_STATE)
    except ImportError:
        log.info('             xgboost not installed - skipping XGBoost')
    if only:
        models = {k: v for k, v in models.items() if k in only or k == 'Baseline (mean)'}
    return models


def train(df, only=None):
    X, y = df[NUM_COLS + CAT_COLS], df['Performance_Metric']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    pre = ColumnTransformer([('num', StandardScaler(), NUM_COLS),
                             ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_COLS)])
    cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted = [], {}
    for name, m in build_models(only).items():
        pipe = Pipeline([('pre', pre), ('model', m)]).fit(X_tr, y_tr)
        p = pipe.predict(X_te)
        rows.append({'Model': name, 'MAE': mean_absolute_error(y_te, p),
                     'RMSE': mean_squared_error(y_te, p) ** 0.5, 'R2 (test)': r2_score(y_te, p),
                     'R2 (5-fold CV)': cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring='r2').mean()})
        fitted[name] = pipe
    results = pd.DataFrame(rows).set_index('Model').round(3)
    best = results.drop('Baseline (mean)')['R2 (5-fold CV)'].idxmax()
    model = fitted[best]
    # uncertainty: 80 % prediction interval from test residuals
    resid = y_te - model.predict(X_te)
    q_lo, q_hi = np.quantile(resid, [0.10, 0.90])
    preds = model.predict(X)
    cfg = {'model': best, 'perf_cut': float(np.quantile(preds, 0.60)),
           'interval_width': float(q_hi - q_lo), 'q_lo': float(q_lo), 'q_hi': float(q_hi)}
    beats = results.loc[best, 'R2 (5-fold CV)'] > results.loc['Baseline (mean)', 'R2 (5-fold CV)']
    log.info(f'[3/5 MODEL]  best = {best} (CV R2 {results.loc[best, "R2 (5-fold CV)"]:.3f}); '
             f'beats baseline: {"yes" if beats else "NO -> rules drive safety decisions"}')
    return model, results, cfg, preds


# ---------------------------------------------------------------- 4. DECIDE
def run_dss(df, preds, cfg):
    recs = [decide(r, p, cfg['perf_cut'], cfg['interval_width'])
            for r, p in zip(df.to_dict('records'), preds)]
    out = pd.DataFrame({
        'Athlete_ID': df['Athlete_ID'].values, 'Sport_Type': df['Sport_Type'].values,
        'Risk_Score': [r['risk'] for r in recs], 'Predicted_Performance': [r['pred'] for r in recs],
        'Recommendation': [r['action'] for r in recs],
        'Main_Drivers': [', '.join(d.lower() for d in r['drivers']) or 'no risk flags' for r in recs],
        'Confidence': ['Low model confidence - coach review' if r['low_confidence'] else 'OK' for r in recs],
    }, index=df.index)
    share = (out.Recommendation.value_counts(normalize=True).reindex(ACTIONS).fillna(0) * 100).round(1)
    log.info('[4/5 DECIDE] ' + ' | '.join(f'{a}: {v}%' for a, v in share.items()))
    return out, share


# ---------------------------------------------------------------- 5. EXPORT
def export(df, model, results, cfg, recs, share, quality, started):
    df.to_csv('sports_performance_clean.csv', index=False)
    recs.to_csv('dss_recommendations.csv', index=False)
    results.to_csv('model_comparison.csv')
    joblib.dump(model, 'performance_model.joblib')
    json.dump({k: v for k, v in cfg.items() if k != 'model'}, open('dss_config.json', 'w'), indent=1)
    report = {'run_at': datetime.now().isoformat(timespec='seconds'),
              'duration_s': round(time.time() - started, 1),
              'data_quality': quality, 'clean_rows': len(df),
              'model_selected': cfg['model'], 'model_comparison': results.to_dict(orient='index'),
              'recommendation_share_pct': share.to_dict()}
    json.dump(report, open('pipeline_run_report.json', 'w'), indent=1, default=str)
    log.info('[5/5 EXPORT] wrote sports_performance_clean.csv, dss_recommendations.csv, model_comparison.csv, '
             'performance_model.joblib, dss_config.json, pipeline_run_report.json')


def main():
    ap = argparse.ArgumentParser(description='AthleteLoadDSS end-to-end pipeline')
    ap.add_argument('--input', default='sports_performance_data.csv', help='raw athlete data CSV')
    args = ap.parse_args()
    started = time.time()
    raw, quality = ingest(args.input)
    df = clean(raw)
    model, results, cfg, preds = train(df)
    recs, share = run_dss(df, preds, cfg)
    export(df, model, results, cfg, recs, share, quality, started)
    log.info(f'Done in {time.time() - started:.1f}s. Start the dashboard with:  streamlit run app.py')


if __name__ == '__main__':
    main()
