
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Campaign Orders Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = [
    "limit_infor",
    "campaign_type",
    "campaign_level",
    "product_level",
    "resource_amount",
    "email_rate",
    "price",
    "discount_rate",
    "hour_resouces",
    "campaign_fee",
]
TARGET = "orders"

MODEL_PARAMS = dict(
    n_estimators=400,
    max_depth=24,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features=0.8,
    random_state=42,
    n_jobs=-1,
)

DATA_CANDIDATES = [
    Path(__file__).parent / "products_campaign_sales.csv",
    Path("products_campaign_sales.csv"),
]

@st.cache_data
def load_data():
    for path in DATA_CANDIDATES:
        if path.exists():
            df = pd.read_csv(path)
            return df, str(path)
    return None, None

@st.cache_resource
def train_model(df):
    work = df.copy()
    if work["price"].isna().any():
        work["price"] = work["price"].fillna(work["price"].median())

    X = work[FEATURES].copy()
    y = work[TARGET].copy()

    # Deterministic 70/15/15 split using target-quantile stratification.
    bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=bins
    )
    temp_bins = pd.qcut(y_temp, q=5, labels=False, duplicates="drop")
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=temp_bins
    )

    X_dev = pd.concat([X_train, X_val])
    y_dev = pd.concat([y_train, y_val])

    model = ExtraTreesRegressor(**MODEL_PARAMS)
    model.fit(X_dev, y_dev)

    pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred),
    }

    return model, X_test, y_test, pred, metrics

df, data_path = load_data()

if df is None:
    st.error("Dataset not found. Put products_campaign_sales.csv in the same folder as app.py.")
    st.stop()

model, X_test, y_test, test_pred, metrics = train_model(df)

# ---------- Sidebar ----------
with st.sidebar:
    st.title("📈 Campaign AI")
    st.caption("Product AD Campaign Performance Prediction")
    st.divider()
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔮 Predict Orders", "📊 Explore Data", "🧠 Model Insights", "ℹ️ Methodology"],
    )
    st.divider()
    st.caption(f"Dataset: {len(df):,} campaigns")
    st.caption(f"Predictors: {len(FEATURES)}")
    st.caption("Model: Tuned Extra Trees Regressor")

# ---------- Header ----------
st.title("Product AD Campaign Performance")
st.markdown(
    "Predict expected **orders** from a product advertising campaign and explore "
    "the factors associated with campaign performance."
)

# ---------- Dashboard ----------
if page == "🏠 Dashboard":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Campaigns", f"{len(df):,}")
    c2.metric("Avg. Orders", f"{df[TARGET].mean():,.0f}")
    c3.metric("Test RMSE", f"{metrics['RMSE']:,.1f}")
    c4.metric("Test R²", f"{metrics['R2']:.3f}")

    st.subheader("Actual vs Predicted Orders")
    chart_df = pd.DataFrame({"Actual": y_test.values, "Predicted": test_pred})
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(chart_df["Actual"], chart_df["Predicted"], alpha=0.65)
    lims = [
        min(chart_df["Actual"].min(), chart_df["Predicted"].min()),
        max(chart_df["Actual"].max(), chart_df["Predicted"].max()),
    ]
    ax.plot(lims, lims, linestyle="--")
    ax.set_xlabel("Actual Orders")
    ax.set_ylabel("Predicted Orders")
    ax.set_title("Final Test Set: Actual vs Predicted")
    st.pyplot(fig, clear_figure=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Orders Distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df[TARGET], bins=30)
        ax.set_xlabel("Orders")
        ax.set_ylabel("Campaign Count")
        st.pyplot(fig, clear_figure=True)

    with right:
        st.subheader("Top Predictive Features")
        imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        imp.plot.barh(ax=ax)
        ax.set_xlabel("Importance")
        st.pyplot(fig, clear_figure=True)

    st.info(
        "Deployment note: campaign_fee, hour_resouces and email_rate should only be used "
        "for genuine pre-campaign forecasting if those values are known before launch."
    )

# ---------- Prediction ----------
elif page == "🔮 Predict Orders":
    st.subheader("Campaign Order Prediction")
    st.write("Enter the campaign inputs below. The model returns an estimated number of orders.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            limit_infor = st.number_input("limit_infor", min_value=0, max_value=10, value=0, step=1)
            campaign_type = st.number_input("campaign_type", min_value=0, max_value=6, value=3, step=1)
            campaign_level = st.number_input("campaign_level", min_value=0, max_value=1, value=1, step=1)
            product_level = st.number_input("product_level", min_value=1, max_value=3, value=1, step=1)
            resource_amount = st.number_input("resource_amount", min_value=1, max_value=9, value=5, step=1)

        with col2:
            email_rate = st.number_input("email_rate", min_value=0.08, max_value=0.84, value=0.47, step=0.01, format="%.2f")
            price = st.number_input("price", min_value=100.0, max_value=197.0, value=163.0, step=1.0)
            discount_rate = st.number_input("discount_rate", min_value=0.49, max_value=0.98, value=0.81, step=0.01, format="%.2f")
            hour_resouces = st.number_input("hour_resouces", min_value=2, max_value=3410, value=850, step=1)
            campaign_fee = st.number_input("campaign_fee", min_value=20, max_value=33380, value=3700, step=10)

        submitted = st.form_submit_button("🚀 Predict Orders", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([{
            "limit_infor": limit_infor,
            "campaign_type": campaign_type,
            "campaign_level": campaign_level,
            "product_level": product_level,
            "resource_amount": resource_amount,
            "email_rate": email_rate,
            "price": price,
            "discount_rate": discount_rate,
            "hour_resouces": hour_resouces,
            "campaign_fee": campaign_fee,
        }])
        prediction = float(model.predict(input_df)[0])

        st.success(f"### Estimated Orders: {prediction:,.0f}")

        a, b = st.columns(2)
        with a:
            st.metric("Predicted orders", f"{prediction:,.0f}")
        with b:
            avg = df[TARGET].mean()
            delta = prediction - avg
            st.metric("vs dataset average", f"{delta:+,.0f}")

        st.caption(
            "This is a model estimate, not a guarantee. Actual campaign results can differ."
        )

# ---------- Data exploration ----------
elif page == "📊 Explore Data":
    st.subheader("Dataset Explorer")

    a, b, c = st.columns(3)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Columns", f"{df.shape[1]}")
    c.metric("Missing Values", f"{int(df.isna().sum().sum())}")

    st.dataframe(df, use_container_width=True, height=420)

    st.subheader("Correlation with Orders")
    corr = df[FEATURES + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    corr.plot.barh(ax=ax)
    ax.set_xlabel("Correlation")
    st.pyplot(fig, clear_figure=True)

    selected = st.selectbox("Choose a feature", FEATURES, index=FEATURES.index("campaign_fee"))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(df[selected], df[TARGET], alpha=0.55)
    ax.set_xlabel(selected)
    ax.set_ylabel("Orders")
    ax.set_title(f"{selected} vs Orders")
    st.pyplot(fig, clear_figure=True)

# ---------- Model insights ----------
elif page == "🧠 Model Insights":
    st.subheader("Model Performance")

    perf = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R²"],
        "Test value": [metrics["MAE"], metrics["RMSE"], metrics["R2"]],
    })
    st.dataframe(perf, use_container_width=True, hide_index=True)

    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    st.dataframe(importance, use_container_width=True, hide_index=True)

    st.subheader("Prediction Error")
    residuals = y_test.values - test_pred
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(test_pred, residuals, alpha=0.65)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Predicted Orders")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title("Residual Diagnostic")
    st.pyplot(fig, clear_figure=True)

    st.subheader("Largest Test Errors")
    errors = X_test.copy()
    errors["Actual Orders"] = y_test.values
    errors["Predicted Orders"] = test_pred
    errors["Absolute Error"] = np.abs(errors["Actual Orders"] - errors["Predicted Orders"])
    st.dataframe(
        errors.sort_values("Absolute Error", ascending=False).head(15),
        use_container_width=True,
    )

    st.warning(
        "Feature importance describes how useful variables were to this fitted model. "
        "It does not establish that changing a feature will causally increase orders."
    )

# ---------- Methodology ----------
else:
    st.subheader("Project Methodology")
    st.markdown("""
### Objective
Build a machine-learning regression model to predict the number of orders generated by a product advertising campaign.

### Data
The application uses the supplied `products_campaign_sales.csv` dataset with 731 observations and 11 columns.

### Cleaning
- Duplicate rows are checked.
- Missing `price` values are filled with the median.
- Predictors are kept in their original numeric form.
- No undocumented engineered features are used in the primary model.

### Train / validation / test strategy
The project uses a deterministic 70% / 15% / 15% split with target-quantile stratification. 
The final test set is kept separate from model fitting.

### Final model
The application uses a tuned **Extra Trees Regressor** with:
- 400 trees
- maximum depth 24
- minimum samples split 4
- minimum samples leaf 2
- maximum features 0.8
- random state 42

### Important deployment consideration
The project report identifies `campaign_fee`, `hour_resouces`, and `email_rate` as variables whose timing should be verified. 
If any of these are only known after or during a campaign, using them for pre-launch forecasting would create a leakage problem.

### Responsible interpretation
Predictions are estimates based on historical data. Feature importance is predictive, not causal.
""")

st.divider()
st.caption("Campaign AI • Built from the project's supplied dataset and documented modeling workflow.")
