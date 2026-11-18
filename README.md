# AthleteLoadDSS - source code

## Setup (once)
    python3 -m venv venv
    source venv/bin/activate          # Windows: venv\Scripts\activate
    pip install pandas scikit-learn xgboost streamlit plotly joblib matplotlib seaborn jupyter

## 1. Run the full pipeline (raw data -> recommendations)
    python pipeline.py
    python pipeline.py --input new_week.csv   # same pipeline on new athlete data
Stages: ingest -> clean -> model -> decide -> export. Takes about 1 minute.
Writes: sports_performance_clean.csv, dss_recommendations.csv, model_comparison.csv,
performance_model.joblib, dss_config.json, pipeline_run_report.json

## 1b. Run the pipeline in the browser (localhost)
    streamlit run pipeline_app.py --server.port 8502
Open http://localhost:8502 -> choose dataset (project CSV or upload) and models -> click "Run pipeline".
Each stage (Ingest, Clean, Model, Decide, Export) shows its results in its own tab; outputs can be downloaded
and are published to the coach dashboard.

## 2. Start the dashboard (demo)
    streamlit run app.py
Open a scenario directly: http://localhost:8501/?scenario=A  (A, B, C or D)

## 3. Explore the analysis step by step
    jupyter notebook Athlete_DSS.ipynb      (Kernel -> Restart & Run All)

## Files
- pipeline.py - end-to-end pipeline (one command)
- pipeline_app.py - the same pipeline as a localhost web app (port 8502)
- Athlete_DSS.ipynb - analysis with charts and explanations
- dss_engine.py - decision engine (risk rules + decision model)
- app.py - Streamlit coach dashboard
- performance_model.joblib / dss_config.json - trained model and cut-offs
