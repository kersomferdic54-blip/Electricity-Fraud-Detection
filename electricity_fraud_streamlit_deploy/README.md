# Electricity Fraud Detection System

Machine-learning decision-support tool that ranks electricity customers by how
unusual their daily consumption looks, so inspectors can prioritise field visits.
A *Potentially Fraudulent* result is a reason to inspect, not proof of theft.

- **Model:** Random Forest (scikit-learn) on 28 engineered consumption features
- **Data:** SGCC (State Grid Corporation of China) public dataset, 42,372 customers,
  daily kWh 2014-01-01 to 2016-10-31, labelled theft / normal
  (source: github.com/henryRDlab/ElectricityTheftDetection)
- **App:** Streamlit: batch scan, single-customer check, consumption charts,
  prediction history with inspection-outcome tracking
- **Staff guide:** `docs/Staff_User_Guide.docx`

## Run the app

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Open http://localhost:8501. Practice files are in `data/samples/`.

## Rebuild everything from the raw data

```bash
.venv/bin/python src/prepare_data.py   # clean + features -> data/processed/
.venv/bin/python src/train.py          # select features, tune, train, evaluate -> models/, reports/
.venv/bin/python src/make_guide.py     # regenerate docs/Staff_User_Guide.docx with current metrics
```

`data/raw/data.csv` must be a wide table: `CONS_NO, FLAG, <one column per date>`.

## Layout

```
app.py                  Streamlit web app
src/features.py         loading, cleaning, feature engineering (shared by training and app)
src/prepare_data.py     step 1 - build feature table
src/train.py            step 2 - feature selection, hyper-parameter search, thresholds, evaluation
src/make_guide.py       step 3 - staff guide (.docx)
models/                 trained model bundle
reports/                metrics.json and evaluation charts
data/samples/           held-out customers for practice uploads (+ answer key)
data/prediction_log.csv created by the app: every check + inspection outcomes
```
