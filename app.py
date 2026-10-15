
from pathlib import Path

from datetime import datetime
import joblib
import streamlit as st

import csv
import html
import io



# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "abusive_text_model.joblib"

st.set_page_config(
    page_title="TextGuard | NLP Research Lab",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------
# Custom styling
# --------------------------------------------------

st.markdown(
    """
    
    .score-track {
        position: relative;
        height: 12px;
        border-radius: 999px;
        background: #263448;
        margin: 18px 0 12px;
    }

    .score-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #0E7490, #5EEAD4);
    }

    .threshold-marker {
        position: absolute;
        top: -6px;
        bottom: -6px;
        width: 2px;
        background: #F8FAFC;
        z-index: 2;
    }

    .score-legend {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        color: #94A3B8;
        font-size: 11px;
    }

    .history-card {
        background: #111D2D;
        border: 1px solid #263448;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 10px;
    }

    .history-snippet {
        color: #CBD5E1;
        font-size: 13px;
        line-height: 1.6;
        overflow-wrap: anywhere;
    }

    .history-meta {
        color: #94A3B8;
        font-size: 11px;
        margin-top: 8px;
    }

    .analytics-card {
        background: #111D2D;
        border: 1px solid #263448;
        border-radius: 12px;
        padding: 16px;
    }

    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap'
    );

    :root {
        --bg: #0B1220;
        --panel: #111D2D;
        --border: #263448;
        --text: #F1F5F9;
        --muted: #94A3B8;
        --teal: #5EEAD4;
        --teal-dark: #0E7490;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 18, 32, 0.95);
    }

    [data-testid="stToolbar"] {
        right: 1rem;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text) !important;
        letter-spacing: -0.035em;
    }

    p, label, .stMarkdown {
        color: #CBD5E1;
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 13px;
        margin-bottom: 2.2rem;
    }

    .brand-icon {
        width: 46px;
        height: 46px;
        border-radius: 14px;
        background: #123A40;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
    }

    .brand-name {
        font-family: 'Space Grotesk', sans-serif;
        color: #F1F5F9;
        font-size: 21px;
        font-weight: 700;
        letter-spacing: 0.06em;
    }

    .brand-subtitle {
        color: var(--muted);
        font-size: 10px;
        letter-spacing: 0.15em;
        margin-top: 3px;
    }

    .status-pill {
        margin-left: auto;
        border: 1px solid #245C55;
        color: #5EEAD4;
        background: #102C2D;
        padding: 7px 12px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.06em;
        white-space: nowrap;
    }

    .eyebrow {
        color: var(--teal);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.17em;
        margin-bottom: 12px;
    }

    .hero-title {
        color: #F8FAFC;
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(36px, 5vw, 58px);
        line-height: 1.08;
        font-weight: 700;
        letter-spacing: -0.055em;
        margin: 0 0 18px 0;
    }

    .hero-title span {
        color: var(--teal);
    }

    .hero-copy {
        max-width: 660px;
        color: #A8B7C9;
        font-size: 16px;
        line-height: 1.8;
        margin-bottom: 28px;
    }

    .section-label {
        color: #E2E8F0;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 17px;
        font-weight: 600;
        margin-bottom: 5px;
    }

    .section-help {
        color: var(--muted);
        font-size: 13px;
        margin-bottom: 15px;
    }

    [data-testid="stTextArea"] textarea {
        background: #0E1929 !important;
        color: #F1F5F9 !important;
        border: 1px solid #304158 !important;
        border-radius: 12px !important;
        padding: 16px !important;
        font-size: 15px !important;
        line-height: 1.7 !important;
        caret-color: var(--teal);
    }

    [data-testid="stTextArea"] textarea:focus {
        border-color: #2DD4BF !important;
        box-shadow: 0 0 0 1px #2DD4BF !important;
    }

    [data-testid="stTextArea"] textarea::placeholder {
        color: #71849B !important;
    }

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #304158;
        background: #142236;
        color: #DCE6F2;
        font-weight: 600;
        min-height: 42px;
        transition: all 0.18s ease;
    }

    .stButton > button:hover {
        border-color: #2DD4BF;
        color: #5EEAD4;
        background: #172C3D;
    }

    [data-testid="stFormSubmitButton"] button,
    .stButton button[kind="primary"] {
        background: #0E7490 !important;
        border: 1px solid #0E7490 !important;
        color: #ECFEFF !important;
        font-weight: 700 !important;
        min-height: 48px !important;
    }

    [data-testid="stFormSubmitButton"] button:hover,
    .stButton button[kind="primary"]:hover {
        background: #0F8BA8 !important;
        border-color: #0F8BA8 !important;
    }

    .metric-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 19px;
        min-height: 105px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 11px;
        letter-spacing: 0.1em;
        font-weight: 600;
        margin-bottom: 12px;
    }

    .metric-value {
        color: #F1F5F9;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 23px;
        font-weight: 600;
    }

    .metric-accent {
        color: var(--teal);
    }

    .result-card {
        background: #111D2D;
        border: 1px solid #304158;
        border-radius: 16px;
        padding: 24px;
        margin-top: 10px;
    }

    .result-positive {
        color: #FDA4AF;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 26px;
        font-weight: 700;
    }

    .result-negative {
        color: #5EEAD4;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 26px;
        font-weight: 700;
    }

    .result-description {
        color: #A8B7C9;
        font-size: 13px;
        line-height: 1.7;
        margin-top: 10px;
    }

    .small-note {
        color: #8192A8;
        font-size: 12px;
        line-height: 1.7;
    }

    .footer {
        border-top: 1px solid #263448;
        margin-top: 42px;
        padding-top: 20px;
        color: #8192A8;
        font-size: 12px;
        line-height: 1.8;
    }

    hr {
        border-color: #263448 !important;
        margin-top: 25px !important;
        margin-bottom: 25px !important;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-top: 1.4rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .brand-subtitle {
            letter-spacing: 0.08em;
        }

        .status-pill {
            font-size: 9px;
            padding: 6px 8px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Load the saved model
# --------------------------------------------------

@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. Run `python src/save_model.py` first."
    )
    st.stop()

artifact = load_artifact()
vectorizer = artifact["vectorizer"]
model = artifact["model"]
threshold = artifact["threshold"]

if "analysis_history" not in st.session_state:
    st.session_state["analysis_history"] = []



# --------------------------------------------------
# Header and introduction
# --------------------------------------------------

st.markdown(
    """
    <div class="brand-row">
        <div class="brand-icon">🛡️</div>
        <div>
            <div class="brand-name">TEXTGUARD</div>
            <div class="brand-subtitle">
                NLP RESEARCH LAB · TEXT CLASSIFICATION
            </div>
        </div>
        <div class="status-pill">● MODEL READY</div>
    </div>

    <div class="eyebrow">MACHINE LEARNING · TEXT ANALYSIS</div>
    <div class="hero-title">
        Understand the patterns<br>behind <span>online language.</span>
    </div>
    <div class="hero-copy">
        Analyze text using a character-level machine-learning classifier
        trained to recognize patterns associated with abusive comments.
        Explore the prediction and its score in seconds.
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Model information cards
# --------------------------------------------------

metric_columns = st.columns(3)

metrics = [
    ("CLASSIFIER", "Character TF-IDF", False),
    ("ALGORITHM", "Logistic Regression", False),
    ("HELD-OUT TEST F1", "79.65%", True),
]

for column, (label, value, accent) in zip(metric_columns, metrics):
    value_class = "metric-value metric-accent" if accent else "metric-value"

    with column:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="{value_class}">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.divider()


# --------------------------------------------------
# Example comments
# --------------------------------------------------

st.markdown(
    '<div class="section-label">01 / Choose an example</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-help">'
    'Try a sample or enter your own text below.'
    '</div>',
    unsafe_allow_html=True,
)

examples = {
    "Neutral example": (
        "I enjoyed reading this article and learned something new."
    ),
    "Insulting example": (
        "You are an idiot and nobody wants to hear from you."
    ),
    "Ambiguous example": (
        "Wow, that was a brilliant idea."
    ),
}

example_columns = st.columns(3)

for column, (label, example_text) in zip(
    example_columns,
    examples.items(),
):
    with column:
        if st.button(label, use_container_width=True):
            st.session_state["comment_input"] = example_text


# --------------------------------------------------
# Text input and analysis
# --------------------------------------------------

st.markdown(
    '<div class="section-label">02 / Analyze text</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-help">'
    'Paste an online comment. Up to 10,000 characters.'
    '</div>',
    unsafe_allow_html=True,
)

with st.form("analysis_form"):
    comment = st.text_area(
        "Comment text",
        key="comment_input",
        placeholder=(
            "Paste a comment here to explore its predicted classification..."
        ),
        height=175,
        max_chars=10000,
        label_visibility="collapsed",
    )

    submitted = st.form_submit_button(
        "Analyze comment  →",
        use_container_width=True,
    )


# --------------------------------------------------
# Prediction results
# --------------------------------------------------

if submitted:
    if not comment.strip():
        st.warning("Enter some text before running the analysis.")

    else:
        features = vectorizer.transform([comment])
        score = float(model.predict_proba(features)[0, 1])
        prediction = int(score >= threshold)

        st.session_state["analysis_history"].insert(
            0,
            {
                "comment": comment.strip()[:180],
                "score": score,
                "prediction": prediction,
                "time": datetime.now().strftime("%H:%M:%S"),
            },
        )

        # Keep only the 10 most recent analyses.
        st.session_state["analysis_history"] = (
            st.session_state["analysis_history"][:10]
        )


        st.divider()

        st.markdown(
            '<div class="section-label">03 / Analysis results</div>',
            unsafe_allow_html=True,
        )

        result_columns = st.columns([1.15, 0.85], gap="large")

        with result_columns[0]:
            if prediction == 1:
                result_title = "Abusive pattern detected"
                result_class = "result-positive"
                result_symbol = "⚑"
                result_description = (
                    "The model score meets or exceeds the configured "
                    "decision threshold. Review the text and its context "
                    "before drawing conclusions."
                )
            else:
                result_title = "No abusive pattern detected"
                result_class = "result-negative"
                result_symbol = "✓"
                result_description = (
                    "The model score is below the configured decision "
                    "threshold. This does not guarantee that the text "
                    "is harmless or appropriate in every context."
                )

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="metric-label">PREDICTED CLASS</div>
                    <div class="{result_class}">
                        {result_symbol} {result_title}
                    </div>
                    <div class="result-description">
                        {result_description}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with result_columns[1]:
            st.markdown(
                """
                <div class="result-card">
                    <div class="metric-label">
                        MODEL SCORE
                    </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="metric-value metric-accent">'
                f'{score:.1%}</div>',
                unsafe_allow_html=True,
            )

            
            st.markdown(
                f"""
                <div class="score-track">
                    <div class="score-fill"
                         style="width: {score * 100:.2f}%;">
                    </div>
                    <div class="threshold-marker"
                         style="left: {threshold * 100:.2f}%;">
                    </div>
                </div>
                <div class="score-legend">
                    <span>0% · Lower score</span>
                    <span>Threshold: {threshold:.0%}</span>
                    <span>100%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="small-note">
                The score is the classifier's estimated probability for
                the abusive class and may not be perfectly calibrated.
                The prediction threshold is {threshold:.2f}.
                This model was trained on labelled Wikipedia comments;
                results can be affected by context, sarcasm, dialect,
                and dataset bias.
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------
# Session analytics and recent history
# --------------------------------------------------

history = st.session_state["analysis_history"]

st.divider()

st.markdown(
    '<div class="section-label">04 / Session analytics</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-help">'
    'A temporary overview of analyses in this browser session.'
    '</div>',
    unsafe_allow_html=True,
)

total_analyses = len(history)
flagged_count = sum(item["prediction"] == 1 for item in history)
clear_count = total_analyses - flagged_count

analytics_columns = st.columns(3)

analytics_data = [
    ("COMMENTS ANALYZED", total_analyses),
    ("FLAGGED ABUSIVE", flagged_count),
    ("PREDICTED NON-ABUSIVE", clear_count),
]

for column, (label, value) in zip(
    analytics_columns,
    analytics_data,
):
    with column:
        st.markdown(
            f"""
            <div class="analytics-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("")
history_heading, clear_column = st.columns([2, 1])

with history_heading:
    st.markdown(
        '<div class="section-label">Recent analyses</div>',
        unsafe_allow_html=True,
    )

with clear_column:
    if st.button(
        "Clear history",
        use_container_width=True,
        disabled=(total_analyses == 0),
    ):
        st.session_state["analysis_history"] = []
        st.rerun()

if not history:
    st.markdown(
        """
        <div class="history-card">
            <div class="history-snippet">
                No analyses yet. Submit a comment above to start building
                your session history.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for index, item in enumerate(history, start=1):
        result_label = (
            "Abusive"
            if item["prediction"] == 1
            else "Non-abusive"
        )

        st.markdown(
            f"""
            <div class="history-card">
                <div class="metric-label">
                    ANALYSIS {index:02d} · {item["time"]}
                </div>
                <div class="history-snippet">
                    {item["comment"]}{"…" if len(item["comment"]) == 180 else ""}
                </div>
                <div class="history-meta">
                    Prediction: {result_label}
                    &nbsp; · &nbsp;
                    Model score: {item["score"]:.1%}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# Download recent analyses as CSV
if history:
    csv_buffer = io.StringIO()
    fieldnames = ["time", "comment", "prediction", "score"]

    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=fieldnames,
    )
    writer.writeheader()
    writer.writerows(history)

    st.download_button(
        label="Download analysis history (CSV)",
        data=csv_buffer.getvalue().encode("utf-8"),
        file_name="textguard_analysis_history.csv",
        mime="text/csv",
        use_container_width=True,
    )


st.caption(
    "Privacy note: recent analyses are kept in Streamlit session memory. "
    "They are not saved to a database or written to a history file. "
    "Avoid entering sensitive personal information."
)


# --------------------------------------------------
# Model performance and insights
# --------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-label">05 / Model insights</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-help">'
    'Performance measured on the held-out test dataset.'
    '</div>',
    unsafe_allow_html=True,
)

metric_columns = st.columns(4)

test_metrics = [
    ("ACCURACY", "95.87%"),
    ("PRECISION", "79.79%"),
    ("RECALL", "79.50%"),
    ("F1 SCORE", "79.65%"),
]

for column, (label, value) in zip(metric_columns, test_metrics):
    with column:
        st.metric(label, value)

st.caption(
    "Test-set results from the selected character-level TF-IDF "
    "and Logistic Regression model at a 0.70 decision threshold. "
    "These results are from the final evaluation, not the current session."
)

st.markdown("#### Model comparison")

st.image(
    "reports/figures/model_comparison.png",
    caption=(
        "Validation F1-score comparison: word-level, "
        "character-level, and combined TF-IDF models."
    ),
    use_container_width=True,
)


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        <b style="color:#CBD5E1;">TEXTGUARD</b> · Academic NLP project<br>
        Character-level TF-IDF + Logistic Regression
        &nbsp;·&nbsp; Python / scikit-learn / Streamlit<br><br>
        This tool supports text analysis and research. It is not a
        definitive judgment of a person's intent and should not be used
        as the sole basis for moderation decisions.
    </div>
    """,
    unsafe_allow_html=True,
)
