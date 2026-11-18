"""AthleteLoadDSS decision engine: knowledge-driven risk rules + model-driven performance prediction."""
import json, os
import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

NUM_COLS = ['Training_Hours_per_Week', 'Average_Heart_Rate', 'BMI', 'Sleep_Hours_per_Night',
            'Daily_Caloric_Intake', 'Hydration_Level', 'Previous_Competition_Performance',
            'Resting_Heart_Rate', 'Body_Fat_Percentage', 'VO2_Max', 'Event_Distance', 'Mental_Focus_Level']
CAT_COLS = ['Sport_Type', 'Training_Intensity', 'Injury_History', 'Altitude_Training']

WEIGHTS = {'injury': .25, 'sleep': .20, 'load': .20, 'rhr': .15, 'hydration': .10, 'focus': .10}
LABELS = {'injury': 'Injury history', 'sleep': 'Sleep', 'load': 'Training load',
          'rhr': 'Resting heart rate', 'hydration': 'Hydration', 'focus': 'Mental focus'}
RISK_REST, RISK_REDUCE, RISK_READY = 65, 45, 25
ACTIONS = ['Increase load', 'Maintain load', 'Reduce load', 'Rest & medical check']


def risk_components(r):
    """Each criterion scored 0 (no risk), 0.5 (caution) or 1 (high risk)."""
    hi_int, long = r['Training_Intensity'] == 'High', r['Training_Hours_per_Week'] > 30
    return {
        'injury': {'Major': 1, 'Minor': .5}.get(r['Injury_History'], 0),
        'sleep': 1 if r['Sleep_Hours_per_Night'] < 6 else .5 if r['Sleep_Hours_per_Night'] < 7 else 0,
        'load': 1 if (hi_int and long) else .5 if (hi_int or long) else 0,
        'rhr': 1 if r['Resting_Heart_Rate'] > 85 else .5 if r['Resting_Heart_Rate'] >= 75 else 0,
        'hydration': 1 if r['Hydration_Level'] < 60 else .5 if r['Hydration_Level'] < 70 else 0,
        'focus': 1 if r['Mental_Focus_Level'] < 4 else .5 if r['Mental_Focus_Level'] < 6 else 0,
    }


def decide(r, pred, perf_cut, interval_width):
    comp = risk_components(r)
    contrib = {k: 100 * WEIGHTS[k] * v for k, v in comp.items()}
    risk = sum(contrib.values())
    if risk >= RISK_REST:
        action = 'Rest & medical check'
    elif risk >= RISK_REDUCE:
        action = 'Reduce load'
    elif risk < RISK_READY and pred >= perf_cut:
        action = 'Increase load'
    else:
        action = 'Maintain load'
    drivers = [LABELS[k] for k, v in sorted(contrib.items(), key=lambda kv: -kv[1]) if v > 0][:3]
    return {'risk': round(risk, 1), 'pred': round(float(pred), 1), 'action': action,
            'contrib': contrib, 'drivers': drivers,
            'low_confidence': interval_width > 30}


class Engine:
    def __init__(self):
        self.model = joblib.load(os.path.join(HERE, 'performance_model.joblib'))
        with open(os.path.join(HERE, 'dss_config.json')) as f:
            cfg = json.load(f)
        self.perf_cut, self.width = cfg['perf_cut'], cfg['interval_width']
        self.q_lo, self.q_hi = cfg['q_lo'], cfg['q_hi']

    def run(self, athlete: dict):
        pred = self.model.predict(pd.DataFrame([athlete])[NUM_COLS + CAT_COLS])[0]
        out = decide(athlete, pred, self.perf_cut, self.width)
        out['interval'] = (max(0, pred + self.q_lo), min(100, pred + self.q_hi))
        return out
