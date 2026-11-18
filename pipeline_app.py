"""AthleteLoadDSS - Pipeline runner on localhost.

Run:   streamlit run pipeline_app.py --server.port 8502
Open:  http://localhost:8502
"""
import io, time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import inspect


def full(fn):
    """Full-width option that works on old and new Streamlit versions."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    w = params.get('width')
    if w is not None and isinstance(w.default, str):   # new API: width='stretch' / 'content'
        return {'width': 'stretch'}
    if 'use_container_width' in params:                 # old API
        return {'use_container_width': True}
    return {}


import pipeline as pl
from dss_engine import ACTIONS, NUM_COLS, CAT_COLS

st.set_page_config(page_title='AthleteLoadDSS Pipeline', page_icon='⚙️', layout='wide')
RED = '#E5533F'
COLORS = {'Increase load': '#2a9d8f', 'Maintain load': '#6c8f5e', 'Reduce load': '#e9a23b', 'Rest & medical check': '#e4513f'}

st.markdown(f"<h1 style='color:{RED};margin-bottom:0'>AthleteLoadDSS — Data Pipeline</h1>"
            "<p style='color:#666;margin-top:0'>Ingest → Clean → Model → Decide → Export, run on localhost. "
            "The coach dashboard (app.py) uses the exported results.</p>", unsafe_allow_html=True)

# ---------------- Sidebar: settings ----------------
with st.sidebar:
    st.header('1 · Data source')
    src = st.radio('Dataset', ['Project dataset (sports_performance_data.csv)', 'Upload a CSV'])
    upload = st.file_uploader('CSV with the same columns', type='csv') if src == 'Upload a CSV' else None
    st.header('2 · Models to train')
    all_models = ['Ridge regression', 'Random Forest', 'XGBoost']
    chosen = st.multiselect('Baseline (mean) is always included', all_models, default=all_models)
    st.caption('Tip: Ridge only runs in a few seconds; all models take about 1 minute.')
    st.header('3 · Output')
    publish = st.checkbox('Publish results to the coach dashboard', value=True,
                          help='Writes the clean data, model and recommendations that app.py reads.')
    run = st.button('▶  Run pipeline', type='primary', **full(st.button))

STAGES = ['1 Ingest', '2 Clean', '3 Model', '4 Decide', '5 Export']


def stage_bar(done, active=None):
    cols = st.columns(5)
    for i, (c, name) in enumerate(zip(cols, STAGES)):
        if i < done:
            bg, fg, icon = '#2a9d8f', 'white', '✓'
        elif i == active:
            bg, fg, icon = '#E8AD3E', 'black', '…'
        else:
            bg, fg, icon = '#eeeeee', '#888', ''
        c.markdown(f"<div style='background:{bg};color:{fg};padding:10px;border-radius:8px;text-align:center;"
                   f"font-weight:600'>{icon} {name}</div>", unsafe_allow_html=True)


if run:
    if src == 'Upload a CSV' and upload is None:
        st.error('Please upload a CSV first.'); st.stop()
    if not chosen:
        st.error('Choose at least one model.'); st.stop()
    t0 = time.time()
    bar = st.empty()
    res = {}
    try:
        with bar.container(): stage_bar(0, 0)
        raw, quality = pl.ingest(upload if upload is not None else 'sports_performance_data.csv')
        missing = [c for c in NUM_COLS + CAT_COLS + ['Performance_Metric', 'Athlete_ID'] if c not in raw.columns]
        if missing:
            st.error(f'The CSV is missing required columns: {missing}'); st.stop()
        res.update(raw=raw, quality=quality)

        with bar.container(): stage_bar(1, 1)
        clean = pl.clean(raw); res['clean'] = clean

        with bar.container(): stage_bar(2, 2)
        with st.spinner(f'Training {len(chosen) + 1} models with 5-fold cross-validation…'):
            model, results, cfg, preds = pl.train(clean, only=chosen)
        res.update(model=model, results=results, cfg=cfg, preds=preds)

        with bar.container(): stage_bar(3, 3)
        recs, share = pl.run_dss(clean, preds, cfg); res.update(recs=recs, share=share)

        with bar.container(): stage_bar(4, 4)
        if publish:
            pl.export(clean, model, results, cfg, recs, share, quality, t0)
            st.cache_resource.clear()
        with bar.container(): stage_bar(5)
        res['seconds'] = time.time() - t0
        res['published'] = publish
        st.session_state['res'] = res
    except Exception as e:
        st.exception(e); st.stop()

res = st.session_state.get('res')
if not res:
    stage_bar(0)
    st.info('Choose the data source and models in the sidebar, then click **Run pipeline**.')
    st.stop()

if not run:
    stage_bar(5)
st.success(f"Pipeline finished in {res['seconds']:.1f} s"
           + (' — results published to the dashboard (run: streamlit run app.py).' if res['published'] else '.'))

t1, t2, t3, t4, t5 = st.tabs(STAGES)

with t1:
    q = res['quality']
    k = st.columns(3)
    k[0].metric('Rows', f"{q['rows']:,}"); k[1].metric('Columns', q['columns']); k[2].metric('Duplicate rows', q['duplicates'])
    miss = pd.Series(q['missing_pct']).sort_values(ascending=True)
    f = px.bar(miss, orientation='h', labels={'value': '% missing', 'index': ''}, title='Missing values per column (raw)')
    f.update_traces(marker_color=[RED if v > 10 else '#9aa0a6' for v in miss.values]); f.update_layout(showlegend=False, height=520)
    st.plotly_chart(f, **full(st.plotly_chart))
    st.dataframe(res['raw'].head(20), **full(st.dataframe))

with t2:
    raw, clean = res['raw'], res['clean']
    k = st.columns(3)
    k[0].metric('Rows before', f'{len(raw):,}'); k[1].metric('Rows after', f'{len(clean):,}', f'{len(clean) - len(raw):,}')
    k[2].metric('Missing values left', int(clean[NUM_COLS + CAT_COLS].isna().sum().sum()))
    st.markdown("""**Cleaning steps applied**
1. Removed exact duplicate rows
2. Dropped rows without `Performance_Metric` (target)
3. `Injury_History` / `Altitude_Training` blanks → category **None**
4. Dropped `Event` (does not match sport) and `Athlete_Name`
5. Numeric gaps → median of the same sport; `Training_Intensity` gaps → most common value""")
    st.dataframe(clean.head(20), **full(st.dataframe))

with t3:
    r = res['results']
    st.markdown(f"**Selected model:** {res['cfg']['model']}  ·  80 % prediction interval width ≈ **{res['cfg']['interval_width']:.0f} points**")
    st.dataframe(r, **full(st.dataframe))
    f = px.bar(r.reset_index(), x='Model', y='R2 (5-fold CV)', title='R² (5-fold CV) — 0 means no better than the average',
               color_discrete_sequence=[RED]); f.add_hline(y=0, line_color='black')
    st.plotly_chart(f, **full(st.plotly_chart))
    best, base = r.loc[res['cfg']['model'], 'R2 (5-fold CV)'], r.loc['Baseline (mean)', 'R2 (5-fold CV)']
    if best <= base:
        st.warning('No model beats the mean baseline → predictions are uncertain, so the DSS lets the readiness rules drive safety decisions and shows a low-confidence flag.')
    else:
        st.success('The selected model beats the baseline.')

with t4:
    recs, share = res['recs'], res['share']
    k = st.columns(4)
    for i, a in enumerate(ACTIONS):
        k[i].metric(f'{a} ({share[a]:.1f} %)', f"{(recs.Recommendation == a).sum():,}")
    c1, c2 = st.columns(2)
    vc = recs.Recommendation.value_counts().reindex(ACTIONS).fillna(0)
    c1.plotly_chart(go.Figure(go.Bar(x=vc.index, y=vc.values, marker_color=[COLORS[a] for a in vc.index]))
                    .update_layout(title='Recommendations', height=380), **full(st.plotly_chart))
    f = px.histogram(recs, x='Risk_Score', nbins=40, title='Risk score distribution', color_discrete_sequence=['#5F6368'])
    for x, col in [(25, '#2a9d8f'), (45, '#E8AD3E'), (65, RED)]: f.add_vline(x=x, line_dash='dash', line_color=col)
    c2.plotly_chart(f.update_layout(height=380), **full(st.plotly_chart))
    st.subheader('Highest-risk athlete-weeks')
    st.dataframe(recs.sort_values('Risk_Score', ascending=False).head(50), **full(st.dataframe), hide_index=True)

with t5:
    st.markdown('Download the pipeline outputs:')
    d = st.columns(3)
    d[0].download_button('Recommendations (CSV)', res['recs'].to_csv(index=False), 'dss_recommendations.csv', 'text/csv')
    d[1].download_button('Clean data (CSV)', res['clean'].to_csv(index=False), 'sports_performance_clean.csv', 'text/csv')
    d[2].download_button('Model comparison (CSV)', res['results'].to_csv(), 'model_comparison.csv', 'text/csv')
    if res['published']:
        st.info('Files were also written to the project folder. Start the coach dashboard with:  `streamlit run app.py`  → http://localhost:8501')
