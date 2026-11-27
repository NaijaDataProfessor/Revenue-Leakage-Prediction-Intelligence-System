"""
RevenueGuard AI — Predictive Revenue Leakage Intelligence
Run:
    streamlit run app.py

Expected model:
    models/revenueguard_logistic_model.joblib
or:
    models/revenueguard_logistic_pipeline.joblib

The saved classifier must be the complete fitted preprocessing + Logistic
Regression pipeline. An optional conditional leakage-amount regressor can be
placed in models/ when available.
"""

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    px = None
    go = None


# =============================================================================
# CONFIGURATION
# =============================================================================
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

# Remote images are decorative only. The app's prediction workflow does not
# depend on these images loading.
# Decorative images are loaded from stable Unsplash image URLs below.


# =============================================================================
# PAGE SETUP AND THEME
# =============================================================================
st.set_page_config(
    page_title="RevenueGuard AI | Revenue Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --rg-bg: #08111F;
        --rg-panel: #111E30;
        --rg-panel-2: #17283D;
        --rg-border: #2A3B50;
        --rg-text: #F3F7FC;
        --rg-muted: #A9B8C9;
        --rg-teal: #19B6B0;
        --rg-cyan: #38D6E8;
        --rg-orange: #F5A623;
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: var(--rg-bg) !important;
        color: var(--rg-text) !important;
    }

    [data-testid="stHeader"] {
        background: rgba(8, 17, 31, 0.92) !important;
    }

    .main .block-container {
        max-width: 1500px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }

    /* Ensure readable text across Streamlit components */
    .stApp p, .stApp label, .stApp li, .stApp span,
    .stApp [data-testid="stMarkdownContainer"] {
        color: var(--rg-text);
    }

    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        letter-spacing: -0.02em;
    }

    [data-testid="stSidebar"] {
        background: #0D1A2B !important;
        border-right: 1px solid var(--rg-border);
    }

    [data-testid="stSidebar"] * {
        color: #EAF2FA !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--rg-muted) !important;
    }

    .rg-hero {
        background:
          radial-gradient(circle at 85% 20%, rgba(56,214,232,.20), transparent 28%),
          linear-gradient(115deg, #10243A 0%, #0D3B4A 55%, #075B65 100%);
        border: 1px solid #246477;
        border-radius: 20px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 16px 40px rgba(0,0,0,.20);
    }

    .rg-eyebrow {
        color: #8DE9E5 !important;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .14em;
        text-transform: uppercase;
        margin-bottom: .5rem;
    }

    .rg-hero h1 {
        color: #FFFFFF !important;
        font-size: clamp(2rem, 4vw, 3rem);
        margin: 0;
        line-height: 1.08;
    }

    .rg-hero p {
        color: #E0F7FA !important;
        font-size: 1.02rem;
        margin: .75rem 0 0 0;
        max-width: 760px;
    }

    .rg-section-label {
        color: #75E1DF !important;
        font-size: .78rem;
        text-transform: uppercase;
        letter-spacing: .12em;
        font-weight: 800;
        margin-bottom: .3rem;
    }

    .rg-card {
        background: linear-gradient(145deg, #15263A, #101D2E);
        border: 1px solid var(--rg-border);
        border-radius: 15px;
        padding: 1.05rem 1.15rem;
        min-height: 112px;
    }

    .rg-card-label {
        color: #AFC0D1 !important;
        font-size: .82rem;
        font-weight: 600;
    }

    .rg-card-value {
        color: #FFFFFF !important;
        font-size: 1.45rem;
        font-weight: 800;
        margin-top: .35rem;
    }

    .rg-card-foot {
        color: #91A5BA !important;
        font-size: .75rem;
        margin-top: .25rem;
    }

    .rg-note {
        background: #132B3D;
        border: 1px solid #22546A;
        border-left: 4px solid var(--rg-cyan);
        color: #E7F5FC !important;
        border-radius: 12px;
        padding: 1rem 1.1rem;
    }

    .rg-risk-high {
        background: rgba(239, 68, 68, .12);
        border: 1px solid rgba(248, 113, 113, .55);
        border-radius: 12px;
        padding: 1rem;
    }

    .rg-risk-low {
        background: rgba(20, 184, 166, .12);
        border: 1px solid rgba(45, 212, 191, .45);
        border-radius: 12px;
        padding: 1rem;
    }

    /* Inputs */
    .stTextInput input, .stNumberInput input, .stTextArea textarea {
        background: #17283D !important;
        color: #FFFFFF !important;
        border: 1px solid #43566D !important;
        border-radius: 9px !important;
    }

    [data-baseweb="select"] > div {
        background: #17283D !important;
        color: #FFFFFF !important;
        border-color: #43566D !important;
    }

    [data-baseweb="popover"], [data-baseweb="menu"],
    [role="option"] {
        background: #17283D !important;
        color: #FFFFFF !important;
    }

    .stSlider label, .stCheckbox label {
        color: #EAF2FA !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: .4rem;
        border-bottom: 1px solid #2A3B50;
    }

    .stTabs [data-baseweb="tab"] {
        color: #C2D0DE !important;
        background: #122238;
        border-radius: 8px 8px 0 0;
    }

    .stTabs [aria-selected="true"] {
        color: #52E0DB !important;
        border-bottom: 3px solid #19B6B0 !important;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: #14243A;
        border: 1px solid #2A3B50;
        border-radius: 13px;
        padding: .9rem 1rem;
    }

    [data-testid="stMetricLabel"] {
        color: #B9C8D8 !important;
    }

    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }

    /* Buttons */
    .stButton button, .stDownloadButton button,
    .stFormSubmitButton button {
        background: linear-gradient(100deg, #087E83, #0B9D9B) !important;
        color: #FFFFFF !important;
        border: 1px solid #35D4CE !important;
        border-radius: 9px !important;
        font-weight: 700 !important;
        min-height: 2.65rem;
    }

    .stButton button:hover, .stDownloadButton button:hover,
    .stFormSubmitButton button:hover {
        background: #0E7490 !important;
        border-color: #8AF3ED !important;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        border-radius: 11px;
    }

    /* Dataframes */
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid #2A3B50;
        border-radius: 10px;
        overflow: hidden;
    }

    .stCaption, [data-testid="stCaptionContainer"] {
        color: #9FB0C3 !important;
    }

    hr {
        border-color: #2A3B50 !important;
    }

    a {
        color: #62DDE0 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# MODEL LOADING
# =============================================================================
def first_existing(paths: list[Path]) -> Optional[Path]:
    return next((path for path in paths if path.exists()), None)


@st.cache_resource(show_spinner="Loading RevenueGuard AI model...")
def load_artifact(path_string: str) -> Any:
    # Only load model files created by you or from a trusted source.
    return joblib.load(path_string)


def unpack_classifier(artifact: Any) -> tuple[Any, Optional[float]]:
    if isinstance(artifact, dict):
        model = (
            artifact.get("model")
            or artifact.get("pipeline")
            or artifact.get("classifier")
        )
        threshold = artifact.get("threshold")
        return model, float(threshold) if threshold is not None else None
    return artifact, None


def unpack_regressor(artifact: Any) -> Any:
    if isinstance(artifact, dict):
        return (
            artifact.get("model")
            or artifact.get("pipeline")
            or artifact.get("regressor")
        )
    return artifact


classifier_path = first_existing(CLASSIFIER_CANDIDATES)
regressor_path = first_existing(REGRESSOR_CANDIDATES)

classifier = None
saved_threshold = None
regressor = None
model_error = None

if classifier_path:
    try:
        artifact = load_artifact(str(classifier_path))
        classifier, saved_threshold = unpack_classifier(artifact)
        if classifier is None or not hasattr(classifier, "predict_proba"):
            raise ValueError("The saved classifier must expose predict_proba().")
    except Exception as exc:
        model_error = str(exc)
else:
    model_error = (
        "No classifier was found. Add your fitted pipeline to the models/ folder."
    )

if regressor_path:
    try:
        regressor_artifact = load_artifact(str(regressor_path))
        regressor = unpack_regressor(regressor_artifact)
        if regressor is None or not hasattr(regressor, "predict"):
            regressor = None
    except Exception:
        regressor = None


# =============================================================================
# HELPERS
# =============================================================================
def validate_feature_frame(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    missing = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if missing:
        return pd.DataFrame(), missing

    clean = data[FEATURE_COLUMNS].copy()

    for column in FEATURE_COLUMNS:
        if column in CATEGORICAL_OPTIONS:
            clean[column] = (
                clean[column]
                .astype("string")
                .fillna(CATEGORICAL_OPTIONS[column][0])
                .astype(str)
            )
        else:
            clean[column] = pd.to_numeric(clean[column], errors="coerce")

    return clean, []


def make_predictions(data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    probabilities = classifier.predict_proba(data)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results = data.copy()
    results["leakage_probability"] = probabilities
    results["leakage_prediction"] = predictions
    results["leakage_status"] = np.where(
        predictions == 1, "Leakage risk", "No leakage predicted"
    )
    results["risk_level"] = pd.cut(
        probabilities,
        bins=[-np.inf, 0.25, 0.50, 0.75, np.inf],
        labels=["Low", "Moderate", "High", "Critical"],
        right=False,
    ).astype(str)

    if regressor is not None:
        estimated_amount = np.full(len(results), np.nan, dtype=float)
        flagged = predictions == 1

        if flagged.any():
            amount_predictions = np.asarray(
                regressor.predict(data.loc[flagged]), dtype=float
            )
            estimated_amount[flagged] = np.maximum(amount_predictions, 0)

        results["estimated_leakage_amount_if_leakage"] = estimated_amount
        results["expected_revenue_exposure"] = (
            probabilities * np.nan_to_num(estimated_amount, nan=0.0)
        )

    return results


def money(value: float) -> str:
    return f"{value:,.2f}"


def section_heading(icon: str, title: str, subtitle: Optional[str] = None) -> None:
    st.markdown(f"### {icon} {title}")
    if subtitle:
        st.caption(subtitle)


def metric_card(icon: str, label: str, value: str, foot: str = "") -> None:
    st.markdown(
        f"""
        <div class="rg-card">
            <div class="rg-card-label">{icon}&nbsp; {label}</div>
            <div class="rg-card-value">{value}</div>
            <div class="rg-card-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_manual_input() -> pd.DataFrame:
    section_heading(
        "🧾",
        "Transaction details",
        "Enter information available at the point you want the transaction assessed.",
    )

    with st.form("single_transaction_form"):
        tab1, tab2, tab3 = st.tabs(
            [
                "💼 Transaction & customer",
                "💳 Billing & payment",
                "⚙️ Operations & controls",
            ]
        )

        with tab1:
            c1, c2, c3 = st.columns(3)
            with c1:
                transaction_value = st.number_input(
                    "Transaction value",
                    min_value=0.0,
                    value=float(NUMERIC_DEFAULTS["transaction_value"]),
                    step=1000.0,
                )
                customer_type = st.selectbox(
                    "Customer type", CATEGORICAL_OPTIONS["customer_type"]
                )
            with c2:
                customer_tenure_months = st.number_input(
                    "Customer tenure (months)",
                    min_value=0,
                    max_value=1200,
                    value=int(NUMERIC_DEFAULTS["customer_tenure_months"]),
                )
                sales_channel = st.selectbox(
                    "Sales channel", CATEGORICAL_OPTIONS["sales_channel"]
                )
            with c3:
                order_complexity_score = st.slider(
                    "Order complexity score", 1, 10,
                    int(NUMERIC_DEFAULTS["order_complexity_score"])
                )
                service_completion_pct = st.slider(
                    "Service completion (%)", 0.0, 100.0,
                    float(NUMERIC_DEFAULTS["service_completion_pct"])
                )

        with tab2:
            c1, c2, c3 = st.columns(3)
            with c1:
                payment_method = st.selectbox(
                    "Payment method", CATEGORICAL_OPTIONS["payment_method"]
                )
                payment_terms_days = st.selectbox(
                    "Payment terms (days)", [0, 7, 14, 30, 45, 60], index=3
                )
            with c2:
                discount_rate_pct = st.number_input(
                    "Discount rate (%)", min_value=0.0, max_value=100.0,
                    value=float(NUMERIC_DEFAULTS["discount_rate_pct"])
                )
                manual_discount_flag = st.selectbox(
                    "Manual discount applied?",
                    [0, 1],
                    format_func=lambda value: "Yes" if value == 1 else "No",
                )
            with c3:
                approval_delay_hours = st.number_input(
                    "Approval delay (hours)", min_value=0.0,
                    value=float(NUMERIC_DEFAULTS["approval_delay_hours"])
                )
                invoice_delay_days = st.number_input(
                    "Invoice delay (days)", min_value=0.0,
                    value=float(NUMERIC_DEFAULTS["invoice_delay_days"])
                )

        with tab3:
            c1, c2, c3 = st.columns(3)
            with c1:
                delivery_delay_days = st.number_input(
                    "Delivery delay (days)", min_value=0.0,
                    value=float(NUMERIC_DEFAULTS["delivery_delay_days"])
                )
                prior_customer_disputes = st.number_input(
                    "Previous customer disputes", min_value=0, value=0, step=1
                )
                prior_payment_defaults = st.number_input(
                    "Previous payment defaults", min_value=0, value=0, step=1
                )
            with c2:
                billing_adjustment_count = st.number_input(
                    "Billing adjustments", min_value=0, value=0, step=1
                )
                reconciliation_frequency_days = st.selectbox(
                    "Reconciliation frequency (days)", [1, 3, 7, 14, 30], index=2
                )
                operational_exception_count = st.number_input(
                    "Operational exceptions", min_value=0, value=0, step=1
                )
            with c3:
                system_integration_score = st.slider(
                    "System integration score", 1, 10,
                    int(NUMERIC_DEFAULTS["system_integration_score"])
                )
                contract_compliance_score = st.slider(
                    "Contract compliance score", 0.0, 100.0,
                    float(NUMERIC_DEFAULTS["contract_compliance_score"])
                )

        submitted = st.form_submit_button(
            "🔍 Predict revenue leakage",
            type="primary",
            use_container_width=True,
        )

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


def show_prediction_result(result: pd.DataFrame) -> None:
    row = result.iloc[0]
    probability = float(row["leakage_probability"])
    flagged = int(row["leakage_prediction"]) == 1

    st.divider()
    section_heading("📊", "Prediction result", "Model assessment for the submitted transaction.")

    a, b, c = st.columns(3)
    a.metric("🎯 Leakage probability", f"{probability:.1%}")
    b.metric("🚦 Predicted outcome", "Leakage risk" if flagged else "No leakage")
    c.metric("⚠️ Risk category", str(row["risk_level"]))

    if flagged:
        st.markdown(
            """
            <div class="rg-risk-high">
                <strong>🚨 Revenue leakage risk detected</strong><br>
                This transaction has been flagged for further review. Prioritise
                verification of billing, payment, approvals and supporting records.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="rg-risk-low">
                <strong>✅ No leakage risk flagged at the selected threshold</strong><br>
                The model did not classify this transaction as leakage risk.
                This is not a guarantee that leakage has not occurred.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if regressor is not None and pd.notna(
        row.get("estimated_leakage_amount_if_leakage", np.nan)
    ):
        x, y = st.columns(2)
        x.metric(
            "💸 Estimated leakage if leakage occurs",
            money(float(row["estimated_leakage_amount_if_leakage"])),
        )
        y.metric(
            "📉 Expected revenue exposure",
            money(float(row["expected_revenue_exposure"])),
        )
    else:
        st.info(
            "The monetary leakage estimate will become available after the "
            "regression model has been trained and saved."
        )

    with st.expander("🧾 View submitted transaction data"):
        st.dataframe(result[FEATURE_COLUMNS], use_container_width=True)


# =============================================================================
# SIDEBAR NAVIGATION AND SETTINGS
# =============================================================================
st.sidebar.markdown(
    """
    <div style="padding: .4rem 0 1rem 0;">
      <div style="font-size:2rem;">🛡️</div>
      <div style="font-size:1.35rem;font-weight:850;color:#FFFFFF;">
        RevenueGuard AI
      </div>
      <div style="font-size:.78rem;color:#9FB5C9;">
        PREDICTIVE REVENUE INTELLIGENCE
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "WORKSPACE",
    [
        "🏠 Overview",
        "🔎 Single prediction",
        "📂 Batch predictions",
        "🧠 About the model",
    ],
    label_visibility="visible",
)

st.sidebar.divider()
st.sidebar.markdown("### 🎛️ Prediction settings")

default_threshold = (
    saved_threshold if saved_threshold is not None else 0.10
)
threshold = st.sidebar.slider(
    "Class 1 threshold",
    min_value=0.05,
    max_value=0.95,
    value=float(np.clip(default_threshold, 0.05, 0.95)),
    step=0.01,
    help=(
        "A lower threshold generally flags more transactions as leakage risk. "
        "It may increase recall but can also increase false positives."
    ),
)
st.sidebar.caption(f"Current threshold: **{threshold:.2f}**")
st.sidebar.divider()
st.sidebar.caption("🔐 Decision-support prototype")
st.sidebar.caption("Predictions should be reviewed by authorised personnel.")


# =============================================================================
# GLOBAL HEADER
# =============================================================================
st.markdown(
    """
    <div class="rg-hero">
      <div class="rg-eyebrow">🛡️ Revenue protection • Risk intelligence • Decision support</div>
      <h1>RevenueGuard AI</h1>
      <p>
        Identify transactions at risk of revenue leakage, prioritise financial
        reviews and support stronger revenue protection decisions.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Small visual strip using remote images; prediction functions work if images
# are blocked or unavailable.
with st.container():
    image_cols = st.columns([1, 1, 1])
    image_specs = [
        (
            "https://images.unsplash.com/photo-1554224155-6726b3ff858f?auto=format&fit=crop&w=900&q=80",
            "💼 Financial control",
        ),
        (
            "https://images.unsplash.com/photo-1460925895917-afdab827c52f?auto=format&fit=crop&w=900&q=80",
            "📊 Business analytics",
        ),
        (
            "https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=900&q=80",
            "📈 Data intelligence",
        ),
    ]
    for col, (image_url, caption) in zip(image_cols, image_specs):
        with col:
            try:
                st.image(image_url, caption=caption, use_container_width=True)
            except Exception:
                st.caption(caption)


# =============================================================================
# MODEL STATUS
# =============================================================================
if classifier is None:
    st.error("❌ The classification model could not be loaded.")
    st.warning(
        "Place the fitted Logistic Regression pipeline in the project's "
        "`models/` folder. The file must be the complete pipeline, including "
        "preprocessing."
    )
    if model_error:
        st.code(model_error)
    st.stop()

status_cols = st.columns([1, 1, 2])
with status_cols[0]:
    st.success("🟢 Classification model loaded")
with status_cols[1]:
    if regressor is not None:
        st.success("🟢 Regression model loaded")
    else:
        st.info("🟡 Regression model not connected")
with status_cols[2]:
    st.caption(f"Classifier artifact: `{classifier_path.name}`")


# =============================================================================
# PAGES
# =============================================================================
if page == "🏠 Overview":
    section_heading(
        "📌",
        "Business intelligence at a glance",
        "A workspace for identifying and reviewing potential revenue leakage.",
    )

    st.markdown(
        """
        <div class="rg-note">
          <strong>How RevenueGuard AI helps</strong><br>
          The application evaluates transaction, customer, payment, billing,
          operational and control information to estimate the likelihood of
          revenue leakage. Treat each prediction as a risk signal for review,
          not as proof that leakage has occurred.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    m1, m2, m3 = st.columns(3)
    with m1:
        metric_card("🎯", "Prediction objective", "Leakage / No leakage", "Binary classification")
    with m2:
        metric_card("🚩", "Positive class", "Class 1", "Revenue leakage")
    with m3:
        metric_card("⚖️", "Current threshold", f"{threshold:.2f}", "Adjustable in the sidebar")

    st.write("")
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        section_heading("🧭", "How to use RevenueGuard AI")
        st.markdown(
            """
            **1. 🔎 Single prediction**  
            Enter the details of one transaction and assess its leakage risk.

            **2. 📂 Batch predictions**  
            Upload a CSV file containing the 20 required predictor columns.

            **3. 🎛️ Adjust the threshold**  
            Use the sidebar to change how readily the model flags Class 1.

            **4. 📋 Review and investigate**  
            Prioritise flagged transactions and verify the underlying records.
            """
        )

    with right:
        section_heading("🔄", "Prediction workflow")
        st.markdown(
            """
            <div class="rg-card">
              <div style="font-size:1.05rem;font-weight:800;color:#FFFFFF;">🧾 Transaction data</div>
              <div style="color:#9FB5C9;text-align:center;padding:.25rem;">↓</div>
              <div style="font-size:1.05rem;font-weight:800;color:#FFFFFF;">🧠 Machine learning model</div>
              <div style="color:#9FB5C9;text-align:center;padding:.25rem;">↓</div>
              <div style="font-size:1.05rem;font-weight:800;color:#FFFFFF;">📊 Leakage probability</div>
              <div style="color:#9FB5C9;text-align:center;padding:.25rem;">↓</div>
              <div style="font-size:1.05rem;font-weight:800;color:#55DED8;">🚦 Risk decision</div>
              <div style="color:#9FB5C9;text-align:center;padding:.25rem;">↓</div>
              <div style="font-size:1.05rem;font-weight:800;color:#FFFFFF;">🔍 Finance / operations review</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    section_heading("🧩", "What the model evaluates")
    feature_groups = st.columns(4)
    group_data = [
        ("💼", "Transaction", "Value, complexity and customer tenure"),
        ("💳", "Payment", "Payment method, terms and discounts"),
        ("🧾", "Billing", "Invoice delays and billing adjustments"),
        ("⚙️", "Operations", "Exceptions, reconciliation and compliance"),
    ]
    for col, (icon, title, detail) in zip(feature_groups, group_data):
        with col:
            st.markdown(
                f"""
                <div class="rg-card">
                  <div style="font-size:1.6rem;">{icon}</div>
                  <div style="font-weight:800;color:#FFFFFF;margin:.35rem 0;">{title}</div>
                  <div style="font-size:.82rem;color:#AFC0D1;">{detail}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

elif page == "🔎 Single prediction":
    section_heading(
        "🔎",
        "Assess an individual transaction",
        "Complete the fields below and submit the transaction for assessment.",
    )
    manual_data = build_manual_input()

    if not manual_data.empty:
        try:
            prediction = make_predictions(manual_data, threshold)
            show_prediction_result(prediction)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.caption(
                "Confirm that the saved pipeline expects the same 20 input "
                "features and includes its fitted preprocessing steps."
            )

elif page == "📂 Batch predictions":
    section_heading(
        "📂",
        "Batch transaction assessment",
        "Score multiple transactions from a single CSV upload.",
    )

    st.markdown(
        """
        <div class="rg-note">
          <strong>Before uploading</strong><br>
          Your CSV must contain all 20 predictor columns used during model
          training. Do not include the target variable or the reference
          probability field as model inputs.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    template = pd.DataFrame(columns=FEATURE_COLUMNS)
    st.download_button(
        "⬇️ Download CSV template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="revenueguard_ai_batch_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader(
        "📤 Upload transaction CSV",
        type=["csv"],
        help="Upload a CSV with all 20 required predictor columns.",
    )

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
                st.success(f"✅ {len(clean_data):,} transactions ready for scoring.")

                with st.spinner("Assessing transactions..."):
                    batch_results = make_predictions(clean_data, threshold)

                total = len(batch_results)
                flagged = int(batch_results["leakage_prediction"].sum())
                average_probability = float(
                    batch_results["leakage_probability"].mean()
                )
                flag_rate = flagged / total if total else 0.0

                st.write("")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("📦 Transactions scored", f"{total:,}")
                k2.metric("🚩 Flagged transactions", f"{flagged:,}")
                k3.metric("📊 Average risk probability", f"{average_probability:.1%}")
                k4.metric("⚠️ Flag rate", f"{flag_rate:.1%}")

                if px is not None:
                    section_heading("📈", "Risk distribution")
                    chart_col1, chart_col2 = st.columns(2)

                    with chart_col1:
                        risk_counts = (
                            batch_results["risk_level"]
                            .value_counts()
                            .reindex(
                                ["Low", "Moderate", "High", "Critical"],
                                fill_value=0,
                            )
                            .rename_axis("Risk level")
                            .reset_index(name="Transactions")
                        )
                        fig_risk = px.bar(
                            risk_counts,
                            x="Risk level",
                            y="Transactions",
                            color="Risk level",
                            title="Transactions by risk category",
                            color_discrete_map={
                                "Low": "#19B6B0",
                                "Moderate": "#F5A623",
                                "High": "#F97316",
                                "Critical": "#EF4444",
                            },
                        )
                        fig_risk.update_layout(
                            paper_bgcolor="#08111F",
                            plot_bgcolor="#111E30",
                            font_color="#EAF2FA",
                            legend_title_text="",
                        )
                        st.plotly_chart(fig_risk, use_container_width=True)

                    with chart_col2:
                        fig_hist = px.histogram(
                            batch_results,
                            x="leakage_probability",
                            nbins=25,
                            title="Predicted leakage probability",
                            labels={"leakage_probability": "Leakage probability"},
                        )
                        fig_hist.add_vline(
                            x=threshold,
                            line_dash="dash",
                            line_color="#F5A623",
                            annotation_text="Selected threshold",
                        )
                        fig_hist.update_layout(
                            paper_bgcolor="#08111F",
                            plot_bgcolor="#111E30",
                            font_color="#EAF2FA",
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                section_heading("📋", "Transaction-level results")
                display_columns = [
                    "leakage_probability",
                    "leakage_prediction",
                    "leakage_status",
                    "risk_level",
                ]
                if regressor is not None:
                    display_columns += [
                        "estimated_leakage_amount_if_leakage",
                        "expected_revenue_exposure",
                    ]

                display = pd.concat(
                    [
                        uploaded_data.reset_index(drop=True),
                        batch_results[display_columns].reset_index(drop=True),
                    ],
                    axis=1,
                )

                st.dataframe(display, use_container_width=True, height=430)

                st.download_button(
                    "⬇️ Download prediction results",
                    data=display.to_csv(index=False).encode("utf-8"),
                    file_name="revenueguard_ai_predictions.csv",
                    mime="text/csv",
                    type="primary",
                )

        except Exception as exc:
            st.error(f"Unable to process the uploaded CSV: {exc}")
            st.caption(
                "Check that the file is a valid CSV and that its predictor "
                "columns contain appropriate values."
            )

elif page == "🧠 About the model":
    section_heading(
        "🧠",
        "Model information",
        "Understand the model's purpose, inputs and current deployment status.",
    )

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown(
            """
            <div class="rg-card">
              <div class="rg-section-label">CLASSIFICATION ENGINE</div>
              <h3>Logistic Regression</h3>
              <p style="color:#B9C8D8;">
                The classification pipeline estimates the probability that a
                transaction belongs to the revenue leakage class.
              </p>
              <hr>
              <p><strong>Class 1:</strong> Revenue leakage</p>
              <p><strong>Class 0:</strong> No revenue leakage</p>
              <p><strong>Decision threshold:</strong> Adjustable from the sidebar</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="rg-card">
              <div class="rg-section-label">PREDICTOR INPUTS</div>
              <h3>20 transaction features</h3>
              <p style="color:#B9C8D8;">
                Inputs cover transaction value, customer history, sales
                channels, payment terms, discounts, billing, operations and
                contract controls.
              </p>
              <hr>
              <p>🧾 Transaction & customer</p>
              <p>💳 Payment & billing</p>
              <p>⚙️ Operational controls</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("### 🔧 Deployment details")
    st.write(f"**Classifier artifact:** `{classifier_path}`")
    st.write(
        f"**Regression artifact:** "
        f"`{regressor_path if regressor_path else 'Not available yet'}`"
    )
    st.write(f"**Current decision threshold:** `{threshold:.2f}`")
    st.write(f"**Required predictor count:** `{len(FEATURE_COLUMNS)}`")

    with st.expander("View all 20 required predictor columns"):
        st.code("\n".join(FEATURE_COLUMNS), language="text")

    st.warning(
        "The 0.10 threshold is only a starting value if it was saved from the "
        "notebook example. Select the deployment threshold using validation "
        "data and business costs, then evaluate it on an untouched test set."
    )

    st.info(
        "Model predictions are risk indicators, not proof that leakage has "
        "occurred. Validate the application with representative real business "
        "data before using it for operational decisions."
    )


# =============================================================================
# FOOTER
# =============================================================================
st.divider()
footer_left, footer_right = st.columns([2, 1])
with footer_left:
    st.caption(
        "🛡️ RevenueGuard AI | Predictive Revenue Leakage Intelligence"
    )
with footer_right:
    st.caption("Built for better revenue protection decisions.")
