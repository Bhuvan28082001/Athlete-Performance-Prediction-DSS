"""AthleteLoadDSS — coach dashboard (Streamlit).  Run:  streamlit run app.py"""
import pandas as pd
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

from dss_engine import Engine, LABELS, WEIGHTS, ACTIONS, RISK_REST, RISK_REDUCE

st.set_page_config(page_title='AthleteLoadDSS', page_icon='🏃', layout='wide')
COLORS = {'Increase load': '#2a9d8f', 'Maintain load': '#6c8f5e',
          'Reduce load': '#e9a23b', 'Rest & medical check': '#e4513f'}


@st.cache_resource
def load():
    eng = Engine()
    df = pd.read_csv('sports_performance_clean.csv', keep_default_na=False, na_values=[''])
    recs = pd.read_csv('dss_recommendations.csv', keep_default_na=False, na_values=[''])
    return eng, df, recs


eng, df, recs = load()

st.markdown("<h1 style='color:#e4513f;margin-bottom:0'>AthleteLoadDSS</h1>"
            "<p style='color:#666;margin-top:0'>Weekly training-load decision support for coaches — "
            "the system recommends, the coach decides.</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(['Athlete assessment', 'Squad overview'])

# ---------------- Sidebar: athlete inputs ----------------
with st.sidebar:
    st.header('Athlete inputs')
    opts = ['Custom', 'A: Well-recovered runner', 'B: Busy soccer player',
            'C: Over-training swimmer', 'D: Injured, exhausted cyclist']
    qp = st.query_params.get('scenario', '')
    start = next((i for i, o in enumerate(opts) if o.startswith(qp + ':')), 0) if qp else 0
    preset = st.selectbox('Load a scenario', opts, index=start)
    P = {'Custom': dict(s='Running', ti='Medium', inj='None', alt='None', h=20.0, sl=7.5, rhr=65.0, hyd=80.0, mf=7.0),
         'A: Well-recovered runner': dict(s='Running', ti='Medium', inj='None', alt='None', h=18.0, sl=8.2, rhr=58.0, hyd=88.0, mf=8.5),
         'B: Busy soccer player': dict(s='Soccer', ti='High', inj='Minor', alt='None', h=26.0, sl=6.5, rhr=72.0, hyd=75.0, mf=6.5),
         'C: Over-training swimmer': dict(s='Swimming', ti='High', inj='None', alt='High', h=34.0, sl=5.8, rhr=80.0, hyd=66.0, mf=5.0),
         'D: Injured, exhausted cyclist': dict(s='Cycling', ti='High', inj='Major', alt='None', h=38.0, sl=5.0, rhr=92.0, hyd=55.0, mf=3.0)}[preset]
    sports = ['Basketball', 'Cycling', 'Running', 'Soccer', 'Swimming', 'Tennis']
    sport = st.selectbox('Sport', sports, index=sports.index(P['s']), key=f's{preset}')
    intensity = st.radio('Training intensity', ['Low', 'Medium', 'High'], index=['Low', 'Medium', 'High'].index(P['ti']), horizontal=True, key=f'ti{preset}')
    injury = st.radio('Injury history', ['None', 'Minor', 'Major'], index=['None', 'Minor', 'Major'].index(P['inj']), horizontal=True, key=f'inj{preset}')
    altitude = st.radio('Altitude training', ['None', 'Low', 'Medium', 'High'], index=['None', 'Low', 'Medium', 'High'].index(P['alt']), horizontal=True, key=f'alt{preset}')
    hours = st.slider('Training hours / week', 5.0, 40.0, P['h'], 0.5, key=f'h{preset}')
    sleep = st.slider('Sleep hours / night', 4.0, 10.0, P['sl'], 0.1, key=f'sl{preset}')
    rhr = st.slider('Resting heart rate (bpm)', 40.0, 100.0, P['rhr'], 1.0, key=f'rhr{preset}')
    hyd = st.slider('Hydration level (%)', 50.0, 100.0, P['hyd'], 1.0, key=f'hyd{preset}')
    focus = st.slider('Mental focus (1–10)', 1.0, 10.0, P['mf'], 0.1, key=f'mf{preset}')

med = df.median(numeric_only=True).to_dict()
athlete = {**med, 'Sport_Type': sport, 'Training_Intensity': intensity, 'Injury_History': injury,
           'Altitude_Training': altitude, 'Training_Hours_per_Week': hours, 'Sleep_Hours_per_Night': sleep,
           'Resting_Heart_Rate': rhr, 'Hydration_Level': hyd, 'Mental_Focus_Level': focus}
res = eng.run(athlete)

with tab1:
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        col = COLORS[res['action']]
        st.markdown(f"""<div style='background:{col};color:white;padding:22px;border-radius:10px'>
            <div style='font-size:14px;opacity:.85'>DSS RECOMMENDATION</div>
            <div style='font-size:30px;font-weight:700'>{res['action']}</div>
            <div style='font-size:14px;margin-top:6px'>Main drivers: {', '.join(res['drivers']) or 'no risk flags'}</div>
            </div>""", unsafe_allow_html=True)
        if res['low_confidence']:
            st.warning('Low model confidence — the performance prediction is uncertain. '
                       'Recommendation is driven mainly by the readiness rules; coach review required.')
    with c2:
        g = go.Figure(go.Indicator(mode='gauge+number', value=res['risk'], title={'text': 'Risk score'},
                                   gauge={'axis': {'range': [0, 100]}, 'bar': {'color': '#333'},
                                          'steps': [{'range': [0, RISK_REDUCE], 'color': '#d8eadf'},
                                                    {'range': [RISK_REDUCE, RISK_REST], 'color': '#f7e1b5'},
                                                    {'range': [RISK_REST, 100], 'color': '#f4c1b8'}]}))
        g.update_layout(height=230, margin=dict(t=40, b=0, l=30, r=40))
        st.plotly_chart(g, **full(st.plotly_chart))
    with c3:
        lo, hi = res['interval']
        st.metric('Predicted performance (0–100)', f"{res['pred']:.1f}")
        st.caption(f'80 % prediction interval: {lo:.0f} – {hi:.0f}')
        st.caption(f"'Increase' requires risk < 25 and prediction ≥ {eng.perf_cut:.1f} (top 40 % of squad)")

    st.subheader('Why? — contribution of each criterion to the risk score')
    cdf = pd.DataFrame({'Criterion': [LABELS[k] for k in WEIGHTS], 'Points': [res['contrib'][k] for k in WEIGHTS],
                        'Max': [100 * WEIGHTS[k] for k in WEIGHTS]})
    f = go.Figure()
    f.add_bar(y=cdf.Criterion, x=cdf.Max, orientation='h', marker_color='#eee', name='Max possible')
    f.add_bar(y=cdf.Criterion, x=cdf.Points, orientation='h', marker_color='#e4513f', name='This athlete')
    f.update_layout(barmode='overlay', height=280, margin=dict(t=10, b=10), xaxis_title='Risk points')
    st.plotly_chart(f, **full(st.plotly_chart))
    st.info('Coach decision: accept the recommendation or override it and record the reason.')

with tab2:
    st.subheader(f'Recommendations across {len(recs):,} athlete-weeks')
    vc = recs.Recommendation.value_counts().reindex(ACTIONS)
    k = st.columns(4)
    for i, a in enumerate(ACTIONS):
        k[i].metric(f'{a} ({vc[a] / len(recs) * 100:.1f} %)', f'{vc[a]:,}')
    sport_f = st.multiselect('Filter by sport', sorted(recs.Sport_Type.unique()))
    view = recs[recs.Sport_Type.isin(sport_f)] if sport_f else recs
    st.dataframe(view.sort_values('Risk_Score', ascending=False).head(200), **full(st.dataframe), height=380, hide_index=True)
