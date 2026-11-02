"""Electricity Fraud Detection - staff web app.

Run:  streamlit run app.py
"""
import sys
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from features import ID_COL, MIN_DAYS, build_features, clean_series, read_any  # noqa: E402

MODEL_PATH = ROOT / "models" / "fraud_model.joblib"
LOG_PATH = ROOT / "data" / "prediction_log.csv"
SAMPLES = ROOT / "data" / "samples"
STATUS_OPTIONS = ["Not inspected", "Inspection scheduled", "Theft confirmed",
                  "Meter fault", "No issue found"]
LOG_COLS = ["checked_at", ID_COL, "risk_score", "risk_level", "prediction", "reasons",
            "period_start", "period_end", "source", "inspection_status", "notes"]

# Shown as percentages in explanations; everything else as a plain number.
PCT_FEATURES = {"missing_frac", "spike_frac", "zero_frac", "longest_zero_run", "low_frac",
                "max_month_drop", "min_month_ratio", "sustained_drop", "recent_ratio",
                "last_vs_first_year", "summer_winter_ratio", "weekend_ratio", "p10_ratio", "p90_ratio"}
KWH_FEATURES = {"mean", "median", "std"}
NOT_A_REASON = {"n_days"}

st.set_page_config(page_title="Electricity Fraud Detection", page_icon="⚡", layout="wide")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def level_of(score, b):
    if score >= b["threshold_high"]:
        return "High"
    if score >= b["threshold_medium"]:
        return "Medium"
    return "Low"


def explain(row, b):
    """Up to three measurements where this customer sits furthest towards the
    theft side of normal training customers (weighted by model importance)."""
    found = []
    for feat, ref in b["reference"].items():
        if feat in NOT_A_REASON or feat not in b["features"]:
            continue
        v = row[feat]
        pct = float(np.interp(v, ref["values"], ref["q"]))
        unusual = pct if ref["direction"] == "high" else 1 - pct
        if unusual < 0.80:
            continue
        if feat in PCT_FEATURES:
            shown = f"{v:.0%}" if v < 10 else "over 1000%"
        elif feat in KWH_FEATURES:
            shown = f"{v:.1f} kWh"
        else:
            shown = f"{v:.2f}"
        side = "higher" if ref["direction"] == "high" else "lower"
        label = b["descriptions"][feat].replace(" (kWh)", "")
        text = f"{label}: {shown} ({side} than {min(unusual, 0.99):.0%} of normal customers)"
        found.append((unusual * (0.5 + b["importances"].get(feat, 0) * 10), text))
    found.sort(reverse=True)
    return "; ".join(t for _, t in found[:3]) or "Several moderate signals together; no single measurement is extreme"


def score(wide, b):
    feats, skipped = build_features(wide)
    if feats.empty:
        return pd.DataFrame(), skipped, feats
    s = b["model"].predict_proba(feats[b["features"]])[:, 1]
    res = pd.DataFrame({
        ID_COL: feats.index,
        "risk_score": np.round(s * 100, 1),
        "risk_level": [level_of(v, b) for v in s],
        "prediction": np.where(s >= b["threshold_high"], "Potentially Fraudulent", "Normal"),
    })
    res["reasons"] = [explain(feats.loc[c], b) if lvl != "Low" else "Consumption pattern looks like normal customers"
                      for c, lvl in zip(feats.index, res["risk_level"])]
    first = wide.apply(lambda r: r.first_valid_index(), axis=1)
    last = wide.apply(lambda r: r.last_valid_index(), axis=1)
    res["period_start"] = pd.to_datetime(first.loc[res[ID_COL]].values).strftime("%Y-%m-%d")
    res["period_end"] = pd.to_datetime(last.loc[res[ID_COL]].values).strftime("%Y-%m-%d")
    order = {"High": 0, "Medium": 1, "Low": 2}
    res = res.sort_values(["risk_level", "risk_score"], key=lambda c: c.map(order) if c.name == "risk_level" else -c)
    res.insert(0, "priority", range(1, len(res) + 1))
    return res.reset_index(drop=True), skipped, feats


def read_log():
    if LOG_PATH.exists():
        log = pd.read_csv(LOG_PATH, dtype={ID_COL: str, "notes": str, "inspection_status": str})
        log = log.reindex(columns=LOG_COLS)
        log["notes"] = log["notes"].fillna("").astype(str)
        log["inspection_status"] = log["inspection_status"].fillna("Not inspected")
        return log
    return pd.DataFrame(columns=LOG_COLS)


def append_log(res, source):
    if res.empty:
        return
    new = res.drop(columns="priority").copy()
    new["checked_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    new["source"] = source
    new["inspection_status"] = "Not inspected"
    new["notes"] = ""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([read_log(), new[LOG_COLS]], ignore_index=True).to_csv(LOG_PATH, index=False)


def level_badge(level):
    color = {"High": "#dc2626", "Medium": "#d97706", "Low": "#16a34a"}[level]
    return f"<span style='background:{color};color:white;padding:2px 10px;border-radius:12px;font-weight:600'>{level} risk</span>"


def history_chart(series, title):
    clean, _ = clean_series(series)
    raw = series.reindex(clean.index) if len(clean) else series
    fig = go.Figure()
    fig.add_scatter(x=raw.index, y=raw.values, mode="lines", name="Daily kWh",
                    line=dict(width=1, color="#93c5fd"))
    fig.add_scatter(x=clean.index, y=clean.rolling(30, min_periods=7).mean(), mode="lines",
                    name="30-day average", line=dict(width=2.5, color="#1d4ed8"))
    missing = raw[raw.isna()]
    if len(missing):
        fig.add_scatter(x=missing.index, y=[0] * len(missing), mode="markers", name="No reading",
                        marker=dict(color="#dc2626", size=4, symbol="line-ns-open"))
    fig.update_layout(title=title, height=340, margin=dict(l=10, r=10, t=40, b=10),
                      yaxis_title="kWh per day", legend=dict(orientation="h", y=-0.2))
    return fig


def monthly_chart(series):
    clean, _ = clean_series(series)
    m = clean.resample("MS").sum()
    fig = go.Figure(go.Bar(x=m.index, y=m.values, marker_color="#2563eb"))
    fig.update_layout(title="Monthly consumption", height=280, yaxis_title="kWh per month",
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


def customer_detail(cust, wide, res, b):
    row = res.loc[res[ID_COL] == cust].iloc[0]
    c1, c2, c3 = st.columns([1.2, 1, 2.5])
    c1.markdown(f"**Customer** `{cust}`<br>{level_badge(row.risk_level)}", unsafe_allow_html=True)
    c2.metric("Risk score", f"{row.risk_score:.0f} / 100")
    c3.markdown(f"**Prediction:** {row.prediction}  \n**Why:** {row.reasons}")
    st.plotly_chart(history_chart(wide.loc[cust], "Daily consumption history"), use_container_width=True)
    st.plotly_chart(monthly_chart(wide.loc[cust]), use_container_width=True)
    past = read_log()
    past = past[past[ID_COL] == str(cust)]
    if len(past):
        st.markdown("**Previous checks for this customer**")
        st.dataframe(past[["checked_at", "risk_score", "risk_level", "prediction",
                           "period_start", "period_end", "inspection_status", "notes"]],
                     hide_index=True, use_container_width=True)


# ----------------------------------------------------------------------------
bundle = load_model()
st.sidebar.title("⚡ Fraud Detection")
page = st.sidebar.radio("Go to", ["Scan customers", "Check one customer",
                                  "Prediction history", "About the model"])
st.sidebar.info("Decision-support tool. A **Potentially Fraudulent** result means "
                "*inspect this customer*; it does **not** prove electricity theft.")

if bundle is None:
    st.error("No trained model found. Run `python src/prepare_data.py` then `python src/train.py`.")
    st.stop()

# ---- Scan customers --------------------------------------------------------
if page == "Scan customers":
    st.title("Scan customers for unusual consumption")
    st.write("Upload a CSV or Excel file with daily readings for one or many customers. "
             "The system ranks customers so inspectors can start with the most suspicious ones.")
    with st.expander("Accepted file layouts and sample files"):
        st.markdown(
            "**Layout A (long)** - one row per customer per day: `CONS_NO, DATE, KWH`  \n"
            "**Layout B (wide)** - one row per customer: `CONS_NO`, then one column per date "
            "(e.g. `2016-01-01`, `2016-01-02`, ...)  \n"
            f"Each customer needs at least **{MIN_DAYS} days** of readings; 12 months or more gives the best results.")
        for f in ["sample_upload_long.csv", "sample_upload_wide.csv", "sample_answer_key.csv"]:
            if (SAMPLES / f).exists():
                st.download_button(f"Download {f}", (SAMPLES / f).read_bytes(), f, "text/csv", key=f)

    up = st.file_uploader("Customer consumption file", type=["csv", "xlsx", "xls"])
    if up is not None:
        try:
            df = pd.read_excel(up) if up.name.endswith(("xlsx", "xls")) else pd.read_csv(up)
            wide = read_any(df)
        except Exception as e:  # noqa: BLE001 - show any parsing problem to the user
            st.error(f"Could not read the file: {e}")
            st.stop()
        with st.spinner(f"Analysing {len(wide):,} customers..."):
            res, skipped, _ = score(wide, bundle)
        st.session_state.update(scan_wide=wide, scan_res=res, scan_skipped=skipped, scan_name=up.name)
        if st.session_state.get("logged_file") != (up.name, up.size):
            append_log(res, up.name)
            st.session_state["logged_file"] = (up.name, up.size)

    if "scan_res" in st.session_state:
        res, wide = st.session_state.scan_res, st.session_state.scan_wide
        skipped = st.session_state.scan_skipped
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Customers analysed", f"{len(res):,}")
        k2.metric("High risk (Potentially Fraudulent)", f"{(res.risk_level == 'High').sum():,}")
        k3.metric("Medium risk (watch list)", f"{(res.risk_level == 'Medium').sum():,}")
        k4.metric("Could not be analysed", f"{len(skipped):,}")

        show = st.multiselect("Show risk levels", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
        view = res[res.risk_level.isin(show)]
        st.dataframe(
            view, hide_index=True, use_container_width=True, height=380,
            column_config={"risk_score": st.column_config.ProgressColumn(
                "Risk score", min_value=0, max_value=100, format="%.0f")})
        st.download_button("Download results (CSV)", res.to_csv(index=False).encode(),
                           f"fraud_scan_{date.today()}.csv", "text/csv")
        if len(skipped):
            with st.expander(f"{len(skipped)} customers could not be analysed"):
                st.dataframe(skipped, hide_index=True, use_container_width=True)

        if len(res):
            st.subheader("Customer details")
            cust = st.selectbox("Select a customer", res[ID_COL],
                                format_func=lambda c: f"{c}  ({res.set_index(ID_COL).loc[c, 'risk_level']} risk)")
            customer_detail(cust, wide, res, bundle)

# ---- Check one customer ----------------------------------------------------
elif page == "Check one customer":
    st.title("Check one customer")
    tab0, tab1, tab2 = st.tabs(["Demo customers", "Type or paste readings", "Upload one customer's file"])
    wide = None
    with tab0:
        st.write("Real customers from the test data (never used in training), for demonstrations.")
        d1, d2 = st.columns(2)
        demo = None
        if d1.button("Show a theft case (real label: theft)", type="primary"):
            demo = "demo_customer_high_risk.csv"
        if d2.button("Show a normal customer (real label: normal)"):
            demo = "demo_customer_normal.csv"
        if demo and (SAMPLES / demo).exists():
            wide = read_any(pd.read_csv(SAMPLES / demo, dtype={ID_COL: str}))
            source = f"demo: {demo}"
    with tab1:
        cid = st.text_input("Customer number (CONS_NO)", "CUSTOMER-001")
        start = st.date_input("Date of the first reading", date(2016, 1, 1))
        txt = st.text_area(f"Daily readings in kWh: one number per day, at least {MIN_DAYS} days "
                           "(one per line or separated by commas; leave a blank between commas "
                           "for a missing day)", height=180, placeholder="12.4, 11.9, 13.1, ...")
        parts = [p.strip() for p in txt.replace("\n", ",").split(",")]
        while parts and parts[-1] == "":
            parts.pop()
        vals = pd.to_numeric(pd.Series(parts, dtype=object).replace("", np.nan), errors="coerce")
        n_vals = int(vals.notna().sum())
        bad = [p for p, v in zip(parts, vals) if p and pd.isna(v)]
        if parts:
            end = pd.Timestamp(start) + pd.Timedelta(days=len(parts) - 1)
            st.caption(f"{n_vals} of {MIN_DAYS} required daily readings entered "
                       f"({pd.Timestamp(start):%d %b %Y} to {end:%d %b %Y}).")
        if st.button("Check customer", type="primary") and parts:
            if bad:
                st.error(f"These entries are not numbers: {', '.join(bad[:5])}"
                         f"{' ...' if len(bad) > 5 else ''}. Use digits only, e.g. 12.4")
            elif n_vals < MIN_DAYS:
                st.session_state.pop("one_res", None)
                st.warning(
                    f"Only {n_vals} daily reading{'s' if n_vals != 1 else ''} entered. "
                    f"The check needs one kWh value **per day** for at least **{MIN_DAYS} days** "
                    "(12 months gives the best results).  \n"
                    "If you have a monthly bill total instead, it cannot be used here: export the "
                    "daily readings from the meter data system, or use the 'Upload one customer's "
                    "file' tab.")
            else:
                idx = pd.date_range(start, periods=len(vals), freq="D")
                wide = pd.DataFrame([vals.values], index=[cid.strip() or "CUSTOMER"], columns=idx)
                source = "manual entry"
    with tab2:
        up = st.file_uploader("File with columns CONS_NO, DATE, KWH", type=["csv", "xlsx"], key="one")
        if up is not None and st.button("Check uploaded customer", type="primary"):
            try:
                df = pd.read_excel(up) if up.name.endswith("xlsx") else pd.read_csv(up)
                wide = read_any(df)
                source = up.name
            except Exception as e:  # noqa: BLE001 - show any parsing problem to the user
                st.error(f"Could not read the file: {e}")
    if wide is not None:
        res, skipped, _ = score(wide, bundle)
        if res.empty:
            st.session_state.pop("one_res", None)
            st.warning("This customer could not be analysed: " + "; ".join(skipped["reason"]))
            if wide.iloc[0].notna().sum() > 1:
                st.plotly_chart(history_chart(wide.iloc[0], "Readings entered"), use_container_width=True)
        else:
            append_log(res, source)
            st.session_state.update(one_wide=wide, one_res=res)
    if "one_res" in st.session_state:
        res = st.session_state.one_res
        for cust in res[ID_COL]:
            customer_detail(cust, st.session_state.one_wide, res, bundle)

# ---- Prediction history ----------------------------------------------------
elif page == "Prediction history":
    st.title("Prediction history and inspection tracking")
    log = read_log()
    if log.empty:
        st.info("No predictions yet. Results are saved here automatically after every scan.")
        st.stop()
    c1, c2 = st.columns(2)
    q = c1.text_input("Search customer number")
    lv = c2.multiselect("Risk level", ["High", "Medium", "Low"], default=["High", "Medium"])
    mask = log.risk_level.isin(lv) & log[ID_COL].astype(str).str.contains(q.strip(), case=False, regex=False)
    st.caption("Update **inspection_status** and **notes** after a field visit, then click Save.")
    edited = st.data_editor(
        log[mask], hide_index=True, use_container_width=True, height=420,
        disabled=[c for c in LOG_COLS if c not in ("inspection_status", "notes")],
        column_config={"inspection_status": st.column_config.SelectboxColumn(options=STATUS_OPTIONS),
                       "risk_score": st.column_config.NumberColumn(format="%.0f")})
    b1, b2 = st.columns([1, 4])
    if b1.button("Save changes", type="primary"):
        log.loc[edited.index, ["inspection_status", "notes"]] = edited[["inspection_status", "notes"]]
        log.to_csv(LOG_PATH, index=False)
        st.success("Saved.")
    b2.download_button("Download full history (CSV)", log.to_csv(index=False).encode(),
                       "prediction_history.csv", "text/csv")
    done = log[log.inspection_status.isin(["Theft confirmed", "Meter fault", "No issue found"])]
    if len(done):
        st.subheader("Inspection outcomes for flagged customers")
        st.dataframe(done.groupby(["risk_level", "inspection_status"]).size().unstack(fill_value=0),
                     use_container_width=True)

# ---- About the model -------------------------------------------------------
else:
    m = bundle["metrics"]
    st.title("About the model")
    st.markdown(
        f"**Algorithm:** Random Forest classifier ({m['best_params']['n_estimators']} decision trees)  \n"
        f"**Trained on:** {bundle['trained_on']} - {m['train_customers']:,} customers, "
        f"tested on {m['test_customers']:,} customers it had never seen  \n"
        f"**Trained:** {bundle['trained_at']}")
    c = st.columns(4)
    c[0].metric("ROC-AUC", f"{m['roc_auc']:.2f}")
    c[1].metric("Precision (High risk)", f"{m['precision']:.0%}")
    c[2].metric("Recall (High risk)", f"{m['recall']:.0%}")
    c[3].metric("Hit rate in top 1%", f"{m['precision_top_1pct']:.0%}",
                help=f"Normal fraud rate is {m['base_fraud_rate']:.1%}")
    st.markdown(
        f"- **Risk score** (0-100) ranks customers; it is not a probability.\n"
        f"- **High risk** = score >= {m['threshold_high'] * 100:.0f} -> *Potentially Fraudulent*, send for inspection.\n"
        f"- **Medium risk** = score {m['threshold_medium'] * 100:.0f}-{m['threshold_high'] * 100:.0f} -> watch list; "
        f"together High + Medium catch about {m['recall_at_medium']:.0%} of known theft cases.\n"
        f"- **Low risk** = below {m['threshold_medium'] * 100:.0f}.")
    for img in ["roc_pr_curves.png", "feature_importance.png", "confusion_matrix.png"]:
        p = ROOT / "reports" / img
        if p.exists():
            st.image(str(p), width=700)
    st.warning("The model learns from historical labels. It can miss theft (false negatives) and "
               "flag honest customers (false positives), e.g. vacant houses, seasonal homes, "
               "solar panels or a broken meter. Always confirm with a field inspection.")
