"""
RevenueGuard AI — Predictive Revenue Leakage Intelligence
Run:
    streamlit run app.py

Required primary model:
    models/revenueguard_logistic_model.joblib
or:
    models/revenueguard_logistic_pipeline.joblib

Optional extended model:
    models/revenueguard_linear_regression.joblib

Both artifacts must be complete fitted pipelines, including preprocessing.
The classifier is the primary predictor; the regressor provides an optional
monetary leakage estimate.
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
    MODEL_DIR / "revenueguard_linear_regression.joblib",
    MODEL_DIR / "revenueguard_regression_model.joblib",
    MODEL_DIR / "revenueguard_leakage_amount_model.joblib",
    BASE_DIR / "revenueguard_linear_regression.joblib",
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
    model_error = "No classifier found. Regression-only operation is supported."

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


def make_predictions(
    data: pd.DataFrame,
    threshold: float,
    run_classifier: bool = True,
    run_regressor: bool = False,
) -> pd.DataFrame:
    """Run the selected prediction engine(s) and return their outputs."""
    results = data.copy()

    if run_classifier:
        if classifier is None:
            raise RuntimeError(
                "The primary classification model is unavailable. "
                "Add the fitted classifier pipeline to the models/ folder."
            )
        probabilities = classifier.predict_proba(data)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
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

    if run_regressor:
        if regressor is None:
            raise RuntimeError(
                "The extended regression model is unavailable. "
                "Add revenueguard_linear_regression.joblib to models/."
            )
        amount_predictions = np.asarray(regressor.predict(data), dtype=float)
        results["predicted_leakage_amount"] = np.maximum(amount_predictions, 0)

    return results


def money(value: float) -> str:
    """Format monetary predictions in Nigerian naira."""
    return f"₦{value:,.2f}"


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
                    "Approval delay (hours)",
                    min_value=0,
                    value=int(NUMERIC_DEFAULTS["approval_delay_hours"]),
                    step=1,
                    format="%d",
                    help="Enter a whole number of hours.",
                )
                invoice_delay_days = st.number_input(
                    "Invoice delay (days)",
                    min_value=0,
                    value=int(NUMERIC_DEFAULTS["invoice_delay_days"]),
                    step=1,
                    format="%d",
                    help="Enter a whole number of days.",
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


def show_prediction_result(
    result: pd.DataFrame,
    primary_mode: str,
    show_other_model: bool,
) -> None:
    row = result.iloc[0]
    st.divider()
    section_heading(
        "📊",
        "Prediction result",
        "The selected model is shown first. The companion model is displayed only when requested.",
    )

    if primary_mode == "Classification":
        probability = float(row["leakage_probability"])
        flagged = int(row["leakage_prediction"]) == 1
        a, b, c = st.columns(3)
        a.metric("🎯 Leakage probability", f"{probability:.1%}")
        b.metric("🚦 Predicted outcome", "Leakage risk" if flagged else "No leakage")
        c.metric("⚠️ Risk category", str(row["risk_level"]))

        if flagged:
            st.warning(
                "The primary classifier flagged this transaction for further review. "
                "This is a risk signal, not confirmation of a realised loss."
            )
        else:
            st.success(
                "The primary classifier did not flag this transaction at the "
                "selected threshold. This does not guarantee that leakage is absent."
            )

        if show_other_model and "predicted_leakage_amount" in result.columns:
            st.markdown("#### Extended prediction · Estimated leakage amount")
            st.metric(
                "💸 Predicted revenue leakage amount",
                money(float(row["predicted_leakage_amount"])),
            )
            st.caption(
                "This is the regressor's direct estimate of leakage amount. "
                "It is not multiplied by the classifier probability."
            )

    else:
        st.markdown("#### Extended prediction · Regression estimate")
        st.metric(
            "💸 Predicted revenue leakage amount",
            money(float(row["predicted_leakage_amount"])),
        )
        st.caption(
            "The regression model estimates the leakage amount directly, "
            "including zero-valued outcomes. This is not a confirmed financial loss."
        )

        if show_other_model and "leakage_probability" in result.columns:
            probability = float(row["leakage_probability"])
            flagged = int(row["leakage_prediction"]) == 1
            a, b, c = st.columns(3)
            a.metric("🎯 Companion leakage probability", f"{probability:.1%}")
            b.metric("🚦 Classifier outcome", "Leakage risk" if flagged else "No leakage")
            c.metric("⚠️ Classifier risk category", str(row["risk_level"]))

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

st.sidebar.markdown("**Choose prediction outputs**")

main_prediction_enabled = classifier is not None and st.sidebar.checkbox(
    "🎯 Main prediction · Leakage risk",
    value=classifier is not None,
    key="main_prediction_enabled",
    help="Run the primary classification model to estimate leakage risk.",
)

extended_prediction_enabled = regressor is not None and st.sidebar.checkbox(
    "💸 Extended prediction · Leakage amount",
    value=False,
    key="extended_prediction_enabled",
    help="Run the regression model to estimate the potential leakage amount in Nigerian naira.",
)

if classifier is None:
    st.sidebar.caption("Main classifier is not connected.")
if regressor is None:
    st.sidebar.caption("Extended regressor is not connected.")

if not main_prediction_enabled and not extended_prediction_enabled:
    st.sidebar.warning("Select at least one prediction output.")

# Preserve a primary display order when both outputs are selected.
primary_mode = (
    "Classification" if main_prediction_enabled
    else "Regression"
)
show_other_model = main_prediction_enabled and extended_prediction_enabled

default_threshold = saved_threshold if saved_threshold is not None else 0.10
if classifier is not None and main_prediction_enabled:
    threshold = st.sidebar.slider(
        "Classifier decision threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(np.clip(default_threshold, 0.05, 0.95)),
        step=0.01,
        help=(
            "Applies only to the classification model. A lower threshold generally "
            "flags more transactions and may increase false positives."
        ),
    )
    st.sidebar.caption(f"Classifier threshold: **{threshold:.2f}**")
else:
    threshold = float(default_threshold)
    st.sidebar.caption("Threshold is not used for regression.")

st.sidebar.divider()
st.sidebar.caption("🔐 Decision-support prototype")
st.sidebar.caption("By NaijaDataProfessor | Waterside Tree Limited")
st.sidebar.markdown("[watersidetree.com](https://watersidetree.com)")


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
if classifier is None and regressor is None:
    st.error("❌ No compatible prediction model could be loaded.")
    st.warning(
        "Add the fitted classifier pipeline and/or the extended regression "
        "pipeline to the project's models/ folder."
    )
    if model_error:
        st.code(model_error)
    st.stop()

status_cols = st.columns([1, 1, 2])
with status_cols[0]:
    if classifier is not None:
        st.success("🟢 Main classifier connected")
    else:
        st.warning("🟡 Main classifier not connected")
with status_cols[1]:
    if regressor is not None:
        st.success("🟢 Extended regressor connected")
    else:
        st.info("ℹ️ Extended regressor not connected")
with status_cols[2]:
    st.caption(
        f"Classifier: `{classifier_path.name if classifier_path else 'Not available'}`  |  "
        f"Regressor: `{regressor_path.name if regressor_path else 'Not available'}`"
    )


# =============================================================================
# PAGES
# =============================================================================
if page == "🏠 Overview":
    section_heading(
        "📌",
        "Revenue intelligence at a glance",
        "Classify leakage risk first, then optionally estimate the potential leakage amount."
    )

    st.markdown(
        """
        <div class="rg-note">
          <strong>How RevenueGuard AI helps</strong><br>
          RevenueGuard AI's main prediction classifies whether a transaction
          presents revenue leakage risk. Its extended prediction estimates the
          potential leakage amount. Use both outputs to prioritise investigation;
          neither output confirms an accounting loss.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    m1, m2, m3 = st.columns(3)
    with m1:
        metric_card("🎯", "Main prediction", "Leakage risk", "Classification")
    with m2:
        metric_card("🧠", "Main model", "Logistic Regression", "Primary classification pipeline")
    with m3:
        metric_card("💸", "Extended prediction", "Leakage amount", "Optional regression estimate")

    st.write("")
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        section_heading("🧭", "How to use RevenueGuard AI")
        st.markdown(
            """
            **1. 🎯 Main prediction**  
            Classify whether a transaction presents revenue leakage risk.

            **2. 💸 Extended prediction**  
            Optionally estimate the potential leakage amount for the same transaction.

            **3. 📂 Batch predictions**  
            Upload a CSV containing the 20 required predictor columns.

            **4. 📋 Validate the records**  
            Review model outputs against invoices, payments, contracts and operational records.
            """
        )
    with right:
        section_heading("🔄", "Prediction workflow")
        st.markdown(
            """
            <div class="rg-card">
              <div style="font-size:1.05rem;font-weight:800;color:#FFFFFF;">🧾 Transaction data</div>
              <div style="color:#9FB5C9;text-align:center;padding:.25rem;">↓</div>
              <div style="font-size:1.05rem;font-weight:800;color:#FFFFFF;">🎯 Classification engine</div>
              <div style="color:#9FB5C9;text-align:center;padding:.25rem;">↓</div>
              <div style="font-size:1.05rem;font-weight:800;color:#55DED8;">🚦 Leakage risk result</div>
              <div style="color:#9FB5C9;text-align:center;padding:.25rem;">↓ optional</div>
              <div style="font-size:1.05rem;font-weight:800;color:#FFFFFF;">💸 Regression amount estimate</div>
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
            run_classifier = main_prediction_enabled
            run_regressor = extended_prediction_enabled
            prediction = make_predictions(
                manual_data,
                threshold,
                run_classifier=run_classifier,
                run_regressor=run_regressor,
            )
            show_prediction_result(
                prediction,
                primary_mode=primary_mode,
                show_other_model=show_other_model,
            )
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
        "Run your selected prediction engine on multiple transactions, with an optional companion-model result."
    )

    st.markdown(
        """
        <div class="rg-note">
          <strong>Before uploading</strong><br>
          Your CSV must contain all 20 predictor columns used during model
          training. Do not include target variables or reference probability
          fields as model inputs.
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

                run_classifier = main_prediction_enabled
                run_regressor = extended_prediction_enabled

                with st.spinner("Running selected RevenueGuard AI prediction..."):
                    batch_results = make_predictions(
                        clean_data,
                        threshold,
                        run_classifier=run_classifier,
                        run_regressor=run_regressor,
                    )

                total = len(batch_results)
                if "leakage_probability" in batch_results.columns:
                    flagged = int(batch_results["leakage_prediction"].sum())
                    average_probability = float(batch_results["leakage_probability"].mean())
                    k1, k2, k3 = st.columns(3)
                    k1.metric("📦 Transactions scored", f"{total:,}")
                    k2.metric("🚩 Classifier-flagged", f"{flagged:,}")
                    k3.metric("🎯 Average leakage probability", f"{average_probability:.1%}")

                if "predicted_leakage_amount" in batch_results.columns:
                    predicted_total = float(batch_results["predicted_leakage_amount"].sum())
                    average_amount = float(batch_results["predicted_leakage_amount"].mean())
                    a1, a2 = st.columns(2)
                    a1.metric("💸 Total estimated leakage (NGN)", money(predicted_total))
                    a2.metric("📊 Average estimated leakage (NGN)", money(average_amount))

                section_heading("📋", "Transaction-level results")
                display_columns = []
                if "leakage_probability" in batch_results.columns:
                    display_columns += [
                        "leakage_probability",
                        "leakage_prediction",
                        "leakage_status",
                        "risk_level",
                    ]
                if "predicted_leakage_amount" in batch_results.columns:
                    display_columns += ["predicted_leakage_amount"]

                display = pd.concat(
                    [
                        uploaded_data.reset_index(drop=True),
                        batch_results[display_columns].reset_index(drop=True),
                    ],
                    axis=1,
                )

                if "predicted_leakage_amount" in display.columns:
                    display = display.rename(columns={
                        "predicted_leakage_amount": "predicted_leakage_amount_ngn"
                    })
                    st.dataframe(
                        display.style.format({
                            "predicted_leakage_amount_ngn": "₦{:,.2f}"
                        }),
                        use_container_width=True,
                        height=430,
                    )
                else:
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
        "Understand the primary classification model and optional extended regression model."
    )

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.markdown(
            """
            <div class="rg-card">
              <div class="rg-section-label">MAIN PREDICTION ENGINE</div>
              <h3>Classification · Logistic Regression</h3>
              <p style="color:#B9C8D8;">
                The primary classifier estimates the probability that a
                transaction presents revenue leakage risk and assigns a
                classification using the selected decision threshold.
              </p>
              <hr>
              <p><strong>Target:</strong> revenue_leakage</p>
              <p><strong>Output:</strong> Leakage risk / no leakage risk</p>
              <p><strong>Threshold:</strong> Adjustable in the sidebar</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="rg-card">
              <div class="rg-section-label">EXTENDED PREDICTION ENGINE</div>
              <h3>Regression · Linear Regression</h3>
              <p style="color:#B9C8D8;">
                When enabled, the regressor estimates a continuous leakage
                amount using the same 20 transaction predictors. It can be
                selected as the primary view or displayed alongside the
                classification result.
              </p>
              <hr>
              <p><strong>Target:</strong> leakage_amount</p>
              <p><strong>Output:</strong> Estimated monetary amount</p>
              <p><strong>Threshold:</strong> Not applicable to regression</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("### 🔧 Deployment details")
    st.write(
        f"**Main classifier artifact:** "
        f"`{classifier_path if classifier_path else 'Not available'}`"
    )
    st.write(
        f"**Extended regressor artifact:** "
        f"`{regressor_path if regressor_path else 'Not connected'}`"
    )
    st.write(f"**Required predictor count:** `{len(FEATURE_COLUMNS)}`")

    with st.expander("View all 20 required predictor columns"):
        st.code("\n".join(FEATURE_COLUMNS), language="text")

    st.info(
        "The classifier is the primary predictor. The regression model is an "
        "optional extended estimate. Validate both models on representative "
        "real business data before using predictions for operational or "
        "financial decisions."
    )

    st.markdown("### 👨🏽‍💻 About the creator")
    st.markdown(
        """
        **NaijaDataProfessor**  
        Data Scientist | Analytics & AI Solutions Consultant

        RevenueGuard AI is presented by **Waterside Tree Limited**, a data,
        analytics and AI consultancy helping organisations turn data into
        actionable business intelligence through analytics engineering,
        machine learning, predictive modelling, business intelligence and
        decision-support solutions.

        **NaijaDataProfessor** has trained 280+ analytics enthusiasts,
        mentored 200+ professionals and delivered 150+ real-world data projects.

        **Connect and collaborate**
        - [LinkedIn](https://www.linkedin.com/in/tolulope-emuleomo-77a231270/)
        - [X / Twitter](https://x.com/naijadataprof)
        - [Medium](https://medium.com/@dataprofessor_)
        - [Waterside Tree Limited](https://watersidetree.com)
        - Email: [dataprofessor290@gmail.com](mailto:dataprofessor290@gmail.com)
        """
    )


# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.markdown(
    """
    <div style="background:linear-gradient(110deg,#10243A,#075B65);
                border:1px solid #246477;border-radius:16px;padding:1.2rem 1.4rem;">
      <div style="font-size:1.15rem;font-weight:850;color:#FFFFFF;">
        🛡️ RevenueGuard AI
      </div>
      <div style="color:#D8F5F5;margin-top:.25rem;">
        A predictive revenue intelligence solution by
        <strong>NaijaDataProfessor</strong> | <strong>Waterside Tree Limited</strong>
      </div>
      <div style="color:#BFE8EA;font-size:.9rem;margin-top:.65rem;">
        Data Science • Machine Learning • Business Intelligence • AI Solutions
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    **Connect with NaijaDataProfessor**

    [LinkedIn](https://www.linkedin.com/in/tolulope-emuleomo-77a231270/) ·
    [X / Twitter](https://x.com/naijadataprof) ·
    [Medium](https://medium.com/@dataprofessor_) ·
    [Waterside Tree Limited](https://watersidetree.com) ·
    [Email](mailto:dataprofessor290@gmail.com)
    """
)
st.caption(
    "© 2026 Waterside Tree Limited. RevenueGuard AI is a decision-support prototype. "
    "Predictions are estimates, not proof of realised revenue loss."
)
