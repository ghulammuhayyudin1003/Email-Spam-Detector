

import sys
from pathlib import Path

import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
# Must be done BEFORE any local src.* imports.
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ocr_extractor import extract_and_clean, is_ocr_available
from src.trainer import RF_MODEL_PATH, LR_MODEL_PATH, load_pipeline
# ── Auto-setup: download dataset + train if models don't exist ────────────────

def auto_train_if_needed():
    """
    On Streamlit Cloud, model .pkl files don't exist on first launch
    (they are excluded from GitHub). This function:
      1. Downloads the SMS Spam dataset automatically (free, public, ~500KB)
      2. Runs train.py to fit and save both models
      3. Only runs once — skipped if models already exist
    """
    if RF_MODEL_PATH.exists() and LR_MODEL_PATH.exists():
        return   # models already trained — nothing to do

    import subprocess
    import urllib.request
    import zipfile
    from pathlib import Path

    # ── Step A: Download dataset ──────────────────────────────────────────────
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    dataset_path = data_dir / "dataset.csv"

    if not dataset_path.exists():
        st.info("⏳ First launch: downloading dataset (~500KB)...")

        # UCI SMS Spam Collection — public domain, no login required
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

        zip_path = data_dir / "spam.zip"
        urllib.request.urlretrieve(url, zip_path)

        # Extract and convert to label,text CSV format
        with zipfile.ZipFile(zip_path, "r") as z:
            raw = z.read("SMSSpamCollection").decode("utf-8")

        # Convert tab-separated format → label,text CSV
        import csv
        import io
        lines = raw.strip().split("\n")
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["label", "text"])
        for line in lines:
            parts = line.split("\t", 1)
            if len(parts) == 2:
                writer.writerow([parts[0].strip(), parts[1].strip()])

        dataset_path.write_text(output.getvalue(), encoding="utf-8")
        zip_path.unlink()   # clean up zip file
        st.info("✅ Dataset downloaded (5,572 emails).")

    # ── Step B: Train models ──────────────────────────────────────────────────
    st.info("⏳ Training models... this takes 2–3 minutes on first launch.")

    progress_bar = st.progress(0, text="Starting training...")

 result = subprocess.run(
    [sys.executable, "train.py", "--no-cv"],  # sys.executable = correct Python
    capture_output=True,
    text=True,
    cwd=str(PROJECT_ROOT),
)

    progress_bar.progress(100, text="Training complete!")

    if result.returncode != 0:
        st.error(f"Training failed:\n{result.stderr}")
        st.stop()
    else:
        st.success("✅ Models trained successfully! Reloading app...")
        st.rerun()


# Call it immediately — runs only on first launch
auto_train_if_needed()

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .spam-box {
        background-color: #ffe4e4;
        border: 2px solid #e53935;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 1.4em;
        font-weight: bold;
        color: #b71c1c;
    }
    .ham-box {
        background-color: #e8f5e9;
        border: 2px solid #43a047;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 1.4em;
        font-weight: bold;
        color: #1b5e20;
    }
    .metric-card {
        background-color: #f5f5f5;
        border-radius: 8px;
        padding: 12px;
        margin: 4px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Model loading (cached — runs only once per Streamlit session) ─────────────

@st.cache_resource(show_spinner="Loading models…")
def load_models() -> dict:
    """
    Load both trained pipelines from disk.

    Streamlit's @cache_resource ensures this runs once and reuses the result
    on every user interaction — the model is never reloaded mid-session.

    Returns:
        dict mapping display name → fitted Pipeline (or None if not trained yet).
    """
    models = {}
    for name, path in [
        ("🌲 Random Forest (Most Stable)",          RF_MODEL_PATH),
        ("📐 Logistic Regression CV (Best Accuracy)", LR_MODEL_PATH),
    ]:
        models[name] = load_pipeline(path) if path.exists() else None
    return models


# ── Prediction helpers ────────────────────────────────────────────────────────

def predict(pipeline, text: str) -> tuple:
    """
    Run inference and return (label, spam_probability, ham_probability).

    Args:
        pipeline: Fitted sklearn Pipeline (TF-IDF + classifier).
        text:     Pre-cleaned email text string.

    Returns:
        (label: str, spam_prob: float, ham_prob: float)
    """
    proba   = pipeline.predict_proba([text])[0]   # shape: (2,)
    classes = list(pipeline.classes_)

    spam_prob = proba[classes.index("spam")]
    ham_prob  = proba[classes.index("ham")]
    label     = "spam" if spam_prob >= 0.5 else "ham"

    return label, spam_prob, ham_prob


def confidence_label(prob: float) -> str:
    """Convert a probability value to a human-readable confidence string."""
    if prob >= 0.95:
        return "Very High"
    elif prob >= 0.80:
        return "High"
    elif prob >= 0.65:
        return "Moderate"
    else:
        return "Low"


# ── Result renderer ───────────────────────────────────────────────────────────

def _render_result(
    label: str,
    spam_prob: float,
    ham_prob: float,
    original_text: str,
    cleaned_text: str,
    chosen_model_name: str,       # BUG FIX: now passed explicitly, not read from global scope
) -> None:
    """
    Render the classification result card with verdict, probabilities, and
    an expandable explanation.

    Args:
        label:             'spam' or 'ham'.
        spam_prob:         Model's spam confidence (0–1).
        ham_prob:          Model's ham confidence (0–1).
        original_text:     Raw input text (for char-count display).
        cleaned_text:      Text after preprocessing (for preview).
        chosen_model_name: Name of the active model (passed from the caller
                           to avoid reading a stale module-level variable).
    """
    st.divider()

    # ── Verdict banner ────────────────────────────────────────────────────────
    if label == "spam":
        st.markdown(
            f'<div class="spam-box">'
            f'🚫 SPAM &nbsp;—&nbsp; {confidence_label(spam_prob)} Confidence '
            f'({spam_prob:.1%})'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="ham-box">'
            f'✅ HAM (Legitimate) &nbsp;—&nbsp; {confidence_label(ham_prob)} Confidence '
            f'({ham_prob:.1%})'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")   # spacing

    # ── Probability bars ──────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🚫 Spam probability", f"{spam_prob:.1%}")
        st.progress(float(spam_prob))
    with col2:
        st.metric("✅ Ham probability", f"{ham_prob:.1%}")
        st.progress(float(ham_prob))

    # ── Explanation expander ──────────────────────────────────────────────────
    # Extract a clean short model name for display (strip the emoji prefix)
    short_model_name = chosen_model_name.split("(")[0].strip().lstrip("🌲📐 ")

    with st.expander("🔍 How was this classified?", expanded=False):
        st.markdown(
            f"""
**Preprocessing** reduced the text from **{len(original_text):,} chars → {len(cleaned_text):,} chars**
(HTML, URLs, stop words, punctuation and numbers removed; text lemmatised).

The cleaned text was then:
1. Vectorised by **TF-IDF** (5,000 features, unigrams + bigrams)
2. Classified by **{short_model_name}**

**Why these models?**
Both were validated on the Kaggle spam dataset (7,673 emails, 70/30 split) in
*"Machine Learning-Based Email Spam Detection"* (EJASET 2025, Table 3):

| Model | Accuracy | Std Dev |
|---|---|---|
| Logistic Regression CV | 0.9845 | 0.0100 |
| Random Forest | 0.9818 | **0.0058** ← most stable |

**Cleaned text preview:**
```
{cleaned_text[:300]}{'…' if len(cleaned_text) > 300 else ''}
```
            """
        )


# ── Main UI ───────────────────────────────────────────────────────────────────

def main() -> None:

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("📧 Email Spam Detector")
    st.markdown(
        """
        **Research-backed** spam detection using TF-IDF features with
        **Random Forest** and **Logistic Regression CV** — the two most
        stable models from the paper
        *"Machine Learning-Based Email Spam Detection: Accuracy, Overfitting
        and Robustness Analysis"* (EJASET, 2025).
        """
    )
    st.divider()

    # ── Load models ───────────────────────────────────────────────────────────
    all_models      = load_models()
    available_models = {k: v for k, v in all_models.items() if v is not None}

    if not available_models:
        st.error(
            "⚠️  No trained models found in `models/`. "
            "Run `python train.py` first.",
            icon="🚨",
        )
        st.code("python train.py", language="bash")
        return

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        # chosen_model_name is a local variable inside main()
        # It is passed explicitly to _render_result() — no global needed.
        chosen_model_name = st.selectbox(
            "Select model",
            options=list(available_models.keys()),
            help=(
                "Random Forest: most stable (std=0.0058). "
                "Logistic Regression CV: highest accuracy (98.45%)."
            ),
        )

        st.divider()
        st.subheader("📊 Paper Results")
        st.markdown("*Table 3 — EJASET 2025:*")

        for model_label, stats in {
            "Random Forest":         {"acc": 0.9818, "std": 0.0058},
            "Logistic Regression CV": {"acc": 0.9845, "std": 0.0100},
        }.items():
            st.markdown(
                f'<div class="metric-card"><b>{model_label}</b><br>'
                f'Accuracy: {stats["acc"]:.4f} &nbsp; Std: {stats["std"]:.4f}</div>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.caption("Retrain anytime: `python train.py`")

    # ── Active pipeline ───────────────────────────────────────────────────────
    pipeline = available_models[chosen_model_name]

    # ── Tabs: text email vs image attachment ──────────────────────────────────
    tab_text, tab_image = st.tabs(["✉️ Text Email", "🖼️ Image Attachment (OCR)"])

    # ── Tab 1: Text email ─────────────────────────────────────────────────────
    with tab_text:
        st.subheader("Paste or type the email content")
        email_text = st.text_area(
            "Email body",
            height=200,
            placeholder=(
                "e.g. Congratulations! You have been selected to receive "
                "a FREE iPhone. Click here to claim your prize NOW!"
            ),
        )

        if st.button("🔍 Classify Email", type="primary", key="btn_text"):
            if not email_text.strip():
                st.warning("Please enter some email text first.")
            else:
                with st.spinner("Analysing…"):
                    from src.preprocessor import clean_text
                    cleaned = clean_text(email_text)

                if not cleaned:
                    st.error(
                        "Text became empty after cleaning. "
                        "Try a longer or more content-rich email."
                    )
                else:
                    label, spam_prob, ham_prob = predict(pipeline, cleaned)
                    _render_result(
                        label, spam_prob, ham_prob,
                        email_text, cleaned,
                        chosen_model_name,          # ← passed explicitly
                    )

    # ── Tab 2: Image OCR ──────────────────────────────────────────────────────
    with tab_image:
        st.subheader("Upload an image email attachment")
        st.markdown(
            "Detects **image-based spam** — text hidden inside images to evade "
            "keyword filters. Tesseract OCR extracts the text, then the same "
            "model classifies it."
        )

        if not is_ocr_available():
            st.warning(
                "⚠️  OCR is not available on this machine.\n\n"
                "**Linux/WSL:** `sudo apt-get install tesseract-ocr`  \n"
                "**macOS:** `brew install tesseract`  \n"
                "**Python:** `pip install pytesseract`  \n"
                "**Streamlit Cloud:** add `tesseract-ocr` to `packages.txt` "
                "(already included in this project).",
                icon="ℹ️",
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload image (.jpg / .png / .gif / .bmp / .webp)",
                type=["jpg", "jpeg", "png", "gif", "bmp", "webp"],
            )

            if uploaded_file is not None:
                col_img, col_info = st.columns([1, 1])

                with col_img:
                    # BUG FIX: use_container_width replaces deprecated use_column_width
                    st.image(
                        uploaded_file,
                        caption="Uploaded image",
                        use_container_width=True,
                    )

                with col_info:
                    with st.spinner("Running OCR…"):
                        uploaded_file.seek(0)         # reset after st.image() read it
                        image_bytes = uploaded_file.read()
                        ocr_text = extract_and_clean(image_bytes)

                    if not ocr_text:
                        st.error(
                            "No text extracted.\n\n"
                            "Tips: ensure the image has visible text, "
                            "is at least 200×200 px, and is not heavily compressed."
                        )
                    else:
                        st.info(
                            f"**OCR text ({len(ocr_text)} chars):**\n\n"
                            f"{ocr_text[:400]}{'…' if len(ocr_text) > 400 else ''}"
                        )
                        label, spam_prob, ham_prob = predict(pipeline, ocr_text)
                        _render_result(
                            label, spam_prob, ham_prob,
                            ocr_text, ocr_text,
                            chosen_model_name,      # ← passed explicitly
                        )


if __name__ == "__main__":
    main()
