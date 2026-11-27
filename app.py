"""RevenueGuard AI — Streamlit application.

Classification model: trained Logistic Regression pipeline from the notebook.
Optional regression model: loaded when a compatible artifact is placed in models/.

Run from the project root:
    streamlit run app.py
"""

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except ImportError:
    px = None


# -----------------------------------------------------------------------------
# Application configuration
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
CLASSIFIER_CANDIDATES = [
    MODEL_DIR / "revenueguard_logistic_model.joblib",
    MODEL_DIR / "revenueguard_logistic_pipeline.joblib",
    BASE_DIR / "revenueguard_logistic_model.joblib",
    BASE_DIR / "revenueguard_logistic_pipeline.joblib",
]
REGRESSOR_CANDIDATES = [
    MODEL_DIR / "revenueguard_regression_model.joblib",
    MODEL_DIR / "revenueguard_leakage_amount_model.joblib",
    BASE_DIR / "revenueguard_regression_model.joblib",
]

# These are the exact predictor names used in the uploaded notebook.
FEATURE_COLUMNS = [
    "transaction_value",
    "customer_tenure_months",
    "customer_type",
    "sales_channel",
    "payment_method",
    "payment_terms_days",
    "discount_rate_pct",
    "manual_discount_flag",
    "approval_delay_hours",
    "invoice_delay_days",
    "order_complexity_score",
    "service_completion_pct",
    "delivery_delay_days",
    "prior_customer_disputes",
    "prior_payment_defaults",
    "billing_adjustment_count",
    "reconciliation_frequency_days",
    "system_integration_score",
    "operational_exception_count",
    "contract_compliance_score",
]

CATEGORICAL_OPTIONS = {
    "customer_type": ["Individual", "SME", "Enterprise"],
    "sales_channel": ["Direct", "Online", "Partner", "Mobile"],
    "payment_method": ["Bank Transfer", "Card", "Cash", "Direct Debit", "Wallet"],
}

NUMERIC_DEFAULTS = {
    "transaction_value": 25000.0,
    "customer_tenure_months": 24,
    "payment_terms_days": 30,
    "discount_rate_pct": 5.0,
    "manual_discount_flag": 0,
    "approval_delay_hours": 4.0,
    "invoice_delay_days": 1.0,
    "order_complexity_score": 5,
    "service_completion_pct": 95.0,
    "delivery_delay_days": 0.0,
    "prior_customer_disputes": 0,
    "prior_payment_defaults": 0,
    "billing_adjustment_count": 0,
    "reconciliation_frequency_days": 7,
    "system_integration_score": 8,
    "operational_exception_count": 0,
    "contract_compliance_score": 90.0,
}

st.set_page_config(
    page_title="RevenueGuard AI | Revenue Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #f5f8fc; }
    [data-testid="stSidebar"] { background-color: #10243a; }
    [data-testid="stSidebar"] * { color: #f4f7fb !important; }
    .rg-header {
        background: linear-gradient(115deg, #10243a 0%, #174b62 100%);
        padding: 1.45rem 1.7rem; border-radius: 16px; color: white;
        margin-bottom: 1.2rem;
    }
    .rg-header h1 { color: white; margin: 0; font-size: 2rem; }
    .rg-header p { color: #d6e7ef; margin: .35rem 0 0 0; }
    .rg-note { background: #eaf3f7; padding: .85rem 1rem; border-radius: 10px;
               border-left: 4px solid #1596a6; color: #183447; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def first_existing(paths: list[Path]) -> Optional[Path]:
    return next((p for p in paths if p.exists()), None)


@st.cache_resource(show_spinner="Loading RevenueGuard AI models...")
def load_artifact(path_string: str) -> Any:
    # Only load Joblib files created by you or from a trusted source.
    return joblib.load(path_string)


def unpack_classifier(artifact: Any) -> tuple[Any, Optional[float]]:
    """Accept either a Pipeline or the model-artifact dictionary."""
    if isinstance(artifact, dict):
        model = artifact.get("model") or artifact.get("pipeline") or artifact.get("classifier")
        threshold = artifact.get("threshold")
        return model, float(threshold) if threshold is not None else None
    return artifact, None


def unpack_regressor(artifact: Any) -> Any:
    if isinstance(artifact, dict):
        return artifact.get("model") or artifact.get("pipeline") or artifact.get("regressor")
    return artifact


classifier_path = first_existing(CLASSIFIER_CANDIDATES)
regressor_path = first_existing(REGRESSOR_CANDIDATES)

classifier = None
saved_threshold = None
regressor = None
model_error = None

if classifier_path:
    try:
        classifier_artifact = load_artifact(str(classifier_path))
        classifier, saved_threshold = unpack_classifier(classifier_artifact)
        if classifier is None or not hasattr(classifier, "predict_proba"):
            raise ValueError("The saved classifier must expose predict_proba().")
    except Exception as exc:
        model_error = str(exc)
else:
    model_error = "No Logistic Regression model file was found in the expected models/ folder."

if regressor_path:
    try:
        regressor_artifact = load_artifact(str(regressor_path))
        regressor = unpack_regressor(regressor_artifact)
        if regressor is None or not hasattr(regressor, "predict"):
            regressor = None
    except Exception:
        regressor = None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def validate_feature_frame(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate and order uploaded data to match the notebook's model inputs."""
    missing = [c for c in FEATURE_COLUMNS if c not in data.columns]
    if missing:
        return pd.DataFrame(), missing

    clean = data[FEATURE_COLUMNS].copy()
    for col in FEATURE_COLUMNS:
        if col in CATEGORICAL_OPTIONS:
            clean[col] = clean[col].astype("string").fillna(CATEGORICAL_OPTIONS[col][0]).astype(str)
        else:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")
    return clean, []


def make_predictions(data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    probabilities = classifier.predict_proba(data)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results = data.copy()
    results["leakage_probability"] = probabilities
    results["leakage_prediction"] = predictions
    results["leakage_status"] = np.where(predictions == 1, "Leakage risk", "No leakage predicted")
    results["risk_level"] = pd.cut(
        probabilities,
        bins=[-np.inf, 0.25, 0.50, 0.75, np.inf],
        labels=["Low", "Moderate", "High", "Critical"],
        right=False,
    ).astype(str)

    # The regression model is conditional severity: estimate amount only for
    # records flagged by the classifier. The expected exposure multiplies this
    # conditional amount by the leakage probability.
    if regressor is not None:
        estimated_amount = np.full(len(results), np.nan, dtype=float)
        flagged = predictions == 1
        if flagged.any():
            amount_predictions = np.asarray(regressor.predict(data.loc[flagged]), dtype=float)
            estimated_amount[flagged] = np.maximum(amount_predictions, 0)
        results["estimated_leakage_amount_if_leakage"] = estimated_amount
        results["expected_revenue_exposure"] = probabilities * np.nan_to_num(estimated_amount, nan=0.0)

    return results


def build_manual_input() -> pd.DataFrame:
    st.subheader("Transaction details")
    st.caption("Enter the transaction attributes that are available at the time you want the prediction.")

    with st.form("single_transaction_form"):
        tab1, tab2, tab3 = st.tabs(["Transaction & customer", "Billing & payment", "Operations & controls"])

        with tab1:
            c1, c2, c3 = st.columns(3)
            with c1:
                transaction_value = st.number_input("Transaction value", min_value=0.0, value=NUMERIC_DEFAULTS["transaction_value"], step=1000.0)
                customer_type = st.selectbox("Customer type", CATEGORICAL_OPTIONS["customer_type"])
            with c2:
                customer_tenure_months = st.number_input("Customer tenure (months)", min_value=0, max_value=1200, value=NUMERIC_DEFAULTS["customer_tenure_months"])
                sales_channel = st.selectbox("Sales channel", CATEGORICAL_OPTIONS["sales_channel"])
            with c3:
                order_complexity_score = st.slider("Order complexity score", 1, 10, NUMERIC_DEFAULTS["order_complexity_score"])
                service_completion_pct = st.slider("Service completion (%)", 0.0, 100.0, NUMERIC_DEFAULTS["service_completion_pct"])

        with tab2:
            c1, c2, c3 = st.columns(3)
            with c1:
                payment_method = st.selectbox("Payment method", CATEGORICAL_OPTIONS["payment_method"])
                payment_terms_days = st.selectbox("Payment terms (days)", [0, 7, 14, 30, 45, 60], index=3)
            with c2:
                discount_rate_pct = st.number_input("Discount rate (%)", min_value=0.0, max_value=100.0, value=NUMERIC_DEFAULTS["discount_rate_pct"])
                manual_discount_flag = st.selectbox("Manual discount applied?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            with c3:
                approval_delay_hours = st.number_input("Approval delay (hours)", min_value=0.0, value=NUMERIC_DEFAULTS["approval_delay_hours"])
                invoice_delay_days = st.number_input("Invoice delay (days)", min_value=0.0, value=NUMERIC_DEFAULTS["invoice_delay_days"])

        with tab3:
            c1, c2, c3 = st.columns(3)
            with c1:
                delivery_delay_days = st.number_input("Delivery delay (days)", min_value=0.0, value=NUMERIC_DEFAULTS["delivery_delay_days"])
                prior_customer_disputes = st.number_input("Previous customer disputes", min_value=0, value=0, step=1)
                prior_payment_defaults = st.number_input("Previous payment defaults", min_value=0, value=0, step=1)
            with c2:
                billing_adjustment_count = st.number_input("Billing adjustments", min_value=0, value=0, step=1)
                reconciliation_frequency_days = st.selectbox("Reconciliation frequency (days)", [1, 3, 7, 14, 30], index=2)
                operational_exception_count = st.number_input("Operational exceptions", min_value=0, value=0, step=1)
            with c3:
                system_integration_score = st.slider("System integration score", 1, 10, NUMERIC_DEFAULTS["system_integration_score"])
                contract_compliance_score = st.slider("Contract compliance score", 0.0, 100.0, NUMERIC_DEFAULTS["contract_compliance_score"])

        submitted = st.form_submit_button("Predict revenue leakage", type="primary", use_container_width=True)

    if not submitted:
        return pd.DataFrame()

    row = {
        "transaction_value": transaction_value,
        "customer_tenure_months": customer_tenure_months,
        "customer_type": customer_type,
        "sales_channel": sales_channel,
        "payment_method": payment_method,
        "payment_terms_days": payment_terms_days,
        "discount_rate_pct": discount_rate_pct,
        "manual_discount_flag": manual_discount_flag,
        "approval_delay_hours": approval_delay_hours,
        "invoice_delay_days": invoice_delay_days,
        "order_complexity_score": order_complexity_score,
        "service_completion_pct": service_completion_pct,
        "delivery_delay_days": delivery_delay_days,
        "prior_customer_disputes": prior_customer_disputes,
        "prior_payment_defaults": prior_payment_defaults,
        "billing_adjustment_count": billing_adjustment_count,
        "reconciliation_frequency_days": reconciliation_frequency_days,
        "system_integration_score": system_integration_score,
        "operational_exception_count": operational_exception_count,
        "contract_compliance_score": contract_compliance_score,
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.title("🛡️ RevenueGuard AI")
st.sidebar.caption("Predictive Revenue Leakage Intelligence")
page = st.sidebar.radio("Navigation", ["Overview", "Single prediction", "Batch predictions", "About the model"])

st.sidebar.divider()
st.sidebar.subheader("Prediction settings")
default_threshold = saved_threshold if saved_threshold is not None else 0.10
# The notebook evaluated 0.10; this is a starting setting, not a validated
# production threshold. Users can adjust it for their operational tolerance.
threshold = st.sidebar.slider(
    "Class 1 threshold",
    min_value=0.05,
    max_value=0.95,
    value=float(np.clip(default_threshold, 0.05, 0.95)),
    step=0.01,
    help="A lower threshold flags more transactions as leakage risk, usually increasing recall while potentially increasing false positives.",
)
st.sidebar.caption(f"Current threshold: {threshold:.2f}")


# -----------------------------------------------------------------------------
# Header and model status
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="rg-header">
      <h1>🛡️ RevenueGuard AI</h1>
      <p>Predict revenue leakage. Prioritise financial risk. Support better decisions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if classifier is None:
    st.error("The Logistic Regression model could not be loaded.")
    st.code(
        "models/revenueguard_logistic_model.joblib\n"
        "or\n"
        "models/revenueguard_logistic_pipeline.joblib"
    )
    if model_error:
        st.caption(f"Details: {model_error}")
    st.info("Save the fitted `lr_pipeline` from your notebook into the project's models/ folder, then restart the Streamlit app.")
    st.stop()

st.success(f"Classification model loaded: `{classifier_path.name}`")
if regressor is None:
    st.info("Leakage amount estimation is not active yet. The Logistic Regression classifier is available; add the trained regression artifact to models/ when it is ready.")
else:
    st.success("Conditional leakage amount regression model loaded.")


# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------
if page == "Overview":
    st.subheader("Business intelligence at a glance")
    st.markdown(
        '<div class="rg-note">RevenueGuard AI evaluates transaction attributes and estimates the probability that a transaction will experience revenue leakage. Predictions are decision-support signals and should be reviewed by authorised finance or operations personnel.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction objective", "Leakage / No leakage")
    c2.metric("Positive class", "Class 1 — Leakage")
    c3.metric("Current threshold", f"{threshold:.2f}")

    st.subheader("How to use the application")
    st.markdown(
        """
        1. Choose **Single prediction** to assess one transaction.
        2. Choose **Batch predictions** to upload a CSV containing multiple transactions.
        3. Adjust the Class 1 threshold in the sidebar to change how aggressively the model flags leakage risk.
        4. Review the predicted probability and risk category, then investigate flagged transactions.
        """
    )
    st.subheader("Prediction workflow")
    st.write("Transaction data → Saved preprocessing and Logistic Regression pipeline → Leakage probability → Threshold decision → Risk review")

elif page == "Single prediction":
    st.subheader("Assess an individual transaction")
    manual_data = build_manual_input()
    if not manual_data.empty:
        try:
            result = make_predictions(manual_data, threshold)
            row = result.iloc[0]
            st.divider()
            st.subheader("Prediction result")
            left, middle, right = st.columns(3)
            left.metric("Leakage probability", f"{row['leakage_probability']:.1%}")
            middle.metric("Predicted outcome", "Leakage" if row["leakage_prediction"] == 1 else "No leakage")
            right.metric("Risk category", row["risk_level"])

            if row["leakage_prediction"] == 1:
                st.error("Revenue leakage risk detected. Prioritise this transaction for review.")
            else:
                st.success("The model did not flag this transaction as leakage risk at the selected threshold.")

            if regressor is not None and pd.notna(row.get("estimated_leakage_amount_if_leakage", np.nan)):
                a, b = st.columns(2)
                a.metric("Estimated leakage if it occurs", f"{row['estimated_leakage_amount_if_leakage']:,.2f}")
                b.metric("Expected revenue exposure", f"{row['expected_revenue_exposure']:,.2f}")
            else:
                st.caption("The monetary leakage estimate will appear here after the regression model has been trained and saved.")

            with st.expander("View submitted transaction data"):
                st.dataframe(manual_data, use_container_width=True)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.caption("Check that the saved pipeline was trained with the same 20 input features and compatible preprocessing.")

elif page == "Batch predictions":
    st.subheader("Upload transactions for batch scoring")
    st.write("Upload a CSV containing all 20 predictor columns used to train the Logistic Regression pipeline.")
    st.download_button(
        "Download CSV template",
        data=pd.DataFrame(columns=FEATURE_COLUMNS).to_csv(index=False).encode("utf-8"),
        file_name="revenueguard_batch_template.csv",
        mime="text/csv",
    )
    uploaded_file = st.file_uploader("Choose transaction CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded_data = pd.read_csv(uploaded_file)
            clean_data, missing_columns = validate_feature_frame(uploaded_data)
            if missing_columns:
                st.error("The uploaded file is missing required columns:")
                st.write(missing_columns)
            elif clean_data.empty:
                st.warning("The uploaded file contains no transaction rows.")
            else:
                st.write(f"Transactions ready for scoring: **{len(clean_data):,}**")
                with st.spinner("Scoring transactions..."):
                    batch_results = make_predictions(clean_data, threshold)

                total = len(batch_results)
                flagged = int(batch_results["leakage_prediction"].sum())
                avg_prob = float(batch_results["leakage_probability"].mean())
                c1, c2, c3 = st.columns(3)
                c1.metric("Transactions scored", f"{total:,}")
                c2.metric("Flagged as leakage risk", f"{flagged:,}")
                c3.metric("Average leakage probability", f"{avg_prob:.1%}")

                if px is not None:
                    fig = px.histogram(
                        batch_results,
                        x="leakage_probability",
                        nbins=25,
                        title="Distribution of predicted leakage probabilities",
                        labels={"leakage_probability": "Leakage probability"},
                    )
                    fig.add_vline(x=threshold, line_dash="dash", annotation_text="Selected threshold")
                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("Transaction-level predictions")
                display_columns = [
                    "leakage_probability", "leakage_prediction", "leakage_status", "risk_level"
                ]
                if regressor is not None:
                    display_columns += ["estimated_leakage_amount_if_leakage", "expected_revenue_exposure"]
                display = pd.concat([uploaded_data.reset_index(drop=True), batch_results[display_columns].reset_index(drop=True)], axis=1)
                st.dataframe(display, use_container_width=True, height=420)

                st.download_button(
                    "Download prediction results",
                    data=display.to_csv(index=False).encode("utf-8"),
                    file_name="revenueguard_ai_predictions.csv",
                    mime="text/csv",
                    type="primary",
                )
        except Exception as exc:
            st.error(f"Unable to process the uploaded CSV: {exc}")

elif page == "About the model":
    st.subheader("Model information")
    st.markdown(
        """
        **Classification model:** Logistic Regression pipeline from the RevenueGuard AI notebook.

        **Positive class:** `1` — revenue leakage.

        **Negative class:** `0` — no revenue leakage.

        **Input predictors:** The 20 transaction, customer, payment, billing, operational and control variables defined in the notebook.

        **Preprocessing:** The saved pipeline includes the fitted imputation, scaling and categorical encoding transformations. The app sends raw input columns to the complete pipeline.
        """
    )
    st.write(f"**Classifier artifact:** `{classifier_path}`")
    st.write(f"**Regression artifact:** `{regressor_path if regressor_path else 'Not available yet'}`")
    st.write(f"**Current decision threshold:** `{threshold:.2f}`")
    st.warning(
        "The 0.10 threshold reflects the threshold used in the notebook's example evaluation, not a confirmed production operating point. Select the deployment threshold using validation data and business costs, then assess it on an untouched test set."
    )
    st.info(
        "Model predictions are not proof that leakage has occurred. They are risk indicators to support investigation. Validate the application with representative real business data before operational use."
    )

st.divider()
st.caption("RevenueGuard AI | Predictive Revenue Leakage Intelligence | Decision-support prototype")
