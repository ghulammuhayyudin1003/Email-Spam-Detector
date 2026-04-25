
import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import (
    BOLD, CYAN, GREEN, RESET,
    compare_models,
    cross_validate_model,
    evaluate_on_test_set,
    print_overfitting_check,
    print_section,
)
from src.preprocessor import clean_series
from src.trainer import (
    RF_MODEL_PATH,
    LR_MODEL_PATH,
    build_random_forest_pipeline,
    build_logistic_regression_cv_pipeline,
    train_and_save,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# ── CLI arguments ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train spam detection models (Random Forest + Logistic Regression CV)."
    )
    parser.add_argument(
        "--data",
        type=Path,
        # BUG FIX: was PROJECT_ROOT / "data" / "dataset.csv" (already correct)
        # Your file was dataset.CSV.csv at root — it has been moved to data/dataset.csv
        default=PROJECT_ROOT / "data" / "dataset.csv",
        help="Path to dataset CSV. Default: data/dataset.csv",
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Skip 5-fold cross-validation (faster for debugging).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.30,
        help="Fraction of data for test set (default: 0.30 — matches paper).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return parser.parse_args()


# ── Step 1: Load & validate ───────────────────────────────────────────────────

def load_and_validate(data_path: Path) -> pd.DataFrame:
    """
    Load the CSV dataset and validate its structure.

    Handles both column naming conventions:
      • Kaggle raw format: v1 (label), v2 (text)
      • Clean format:      label, text

    Normalises all labels to lowercase so 'Spam'/'Ham' → 'spam'/'ham'.
    """
    print_section("Step 1 — Loading Dataset")

    if not data_path.exists():
        logger.error(
            "Dataset not found: %s\n"
            "  Make sure data/dataset.csv exists in the project folder.\n"
            "  Or pass a custom path: python train.py --data /path/to/file.csv",
            data_path,
        )
        sys.exit(1)

    logger.info("Reading: %s", data_path)
    df = pd.read_csv(data_path, encoding="latin-1")

    # Normalise column names (handles v1/v2, label/text, category/message, etc.)
    col_map = {}
    for col in df.columns:
        if col.lower() in ("v1", "label", "category", "class"):
            col_map[col] = "label"
        elif col.lower() in ("v2", "text", "message", "email", "body"):
            col_map[col] = "text"
    df.rename(columns=col_map, inplace=True)

    if "label" not in df.columns or "text" not in df.columns:
        logger.error(
            "CSV must have 'label' and 'text' columns (or v1/v2). Found: %s",
            list(df.columns),
        )
        sys.exit(1)

    df = df[["label", "text"]].copy()

    # Drop nulls and normalise types
    before = len(df)
    df.dropna(subset=["label", "text"], inplace=True)
    df["text"]  = df["text"].astype(str)
    df["label"] = df["label"].str.strip().str.lower()   # 'Spam' → 'spam'
    after = len(df)

    if before != after:
        logger.warning("Dropped %d rows with null values.", before - after)

    # Validate labels
    unique_labels = set(df["label"].unique())
    if not unique_labels.issubset({"spam", "ham"}):
        logger.error("Unexpected labels found: %s. Expected only 'spam' and 'ham'.", unique_labels)
        sys.exit(1)

    # Print dataset statistics
    spam_count = (df["label"] == "spam").sum()
    ham_count  = (df["label"] == "ham").sum()
    spam_pct   = spam_count / len(df) * 100

    print(f"\n  {'Total samples':<25} {len(df):>8,}")
    print(f"  {'Spam':<25} {spam_count:>8,}  ({spam_pct:.1f}%)")
    print(f"  {'Ham':<25} {ham_count:>8,}  ({100 - spam_pct:.1f}%)")

    if spam_pct < 15 or spam_pct > 85:
        logger.warning(
            "Severe class imbalance (%.1f%% spam). "
            "Both models use class_weight='balanced' to compensate.",
            spam_pct,
        )

    return df


# ── Step 2: Preprocessing ─────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full text cleaning pipeline to the 'text' column.

    Expected runtime: ~30–60 seconds for 7,673 rows on a standard laptop.
    """
    print_section("Step 2 — Text Preprocessing")
    logger.info("Cleaning %d email texts (may take ~30–60 seconds) …", len(df))

    t0 = time.time()
    df["clean_text"] = clean_series(df["text"])
    elapsed = time.time() - t0

    avg_raw   = df["text"].str.len().mean()
    avg_clean = df["clean_text"].str.len().mean()
    reduction = (1 - avg_clean / avg_raw) * 100

    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Avg raw length   : {avg_raw:>7.0f} chars")
    print(f"  Avg clean length : {avg_clean:>7.0f} chars  ({reduction:.0f}% reduction)")

    empty_mask = df["clean_text"].str.strip() == ""
    if empty_mask.any():
        logger.warning("%d emails became empty after cleaning — dropping them.", empty_mask.sum())
        df = df[~empty_mask].copy()

    return df


# ── Step 3: Train/test split ──────────────────────────────────────────────────

def split_data(df: pd.DataFrame, test_size: float, seed: int):
    """
    Stratified split that preserves the spam/ham ratio in both sets.

    Stratification is critical when classes are imbalanced so that neither
    set ends up with a disproportionate share of the minority class.
    """
    print_section("Step 3 — Train / Test Split")

    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed,
    )

    tr_spam_pct = (y_train == "spam").mean() * 100
    te_spam_pct = (y_test  == "spam").mean() * 100

    print(f"\n  {'Split':<12} {'Samples':>8}   {'Spam%':>7}")
    print(f"  {'─'*30}")
    print(f"  {'Training':<12} {len(X_train):>8,}   {tr_spam_pct:>6.1f}%")
    print(f"  {'Test':<12} {len(X_test):>8,}   {te_spam_pct:>6.1f}%")
    print(f"\n  {GREEN}✓ Stratification preserved the spam ratio in both splits.{RESET}")

    return X_train, X_test, y_train, y_test


# ── Step 4: Train + evaluate ──────────────────────────────────────────────────

def train_evaluate(
    pipeline,
    X_train, X_test,
    y_train, y_test,
    model_name: str,
    save_path: Path,
    run_cv: bool,
    seed: int,
) -> dict:
    """
    Full cycle: cross-validate → fit on full train set → evaluate → save.

    Cross-validation is run on the training set ONLY (X_test is never seen
    during CV). A fresh pipeline is used for CV so the final pipeline can be
    fitted on the entire training set without data leakage.
    """
    print_section(f"Training: {model_name}")

    # ── Optional 5-fold CV ────────────────────────────────────────────────────
    if run_cv:
        cv_pipeline = (
            build_random_forest_pipeline()
            if model_name == "Random Forest"
            else build_logistic_regression_cv_pipeline()
        )
        cross_validate_model(
            model=cv_pipeline,
            X=X_train,
            y=y_train,
            model_name=model_name,
            n_splits=5,
        )

    # ── Fit final model on full training set ──────────────────────────────────
    logger.info("Fitting final %s …", model_name)
    t0 = time.time()
    fitted = train_and_save(
        pipeline=pipeline,
        X_train=X_train,
        y_train=y_train,
        save_path=save_path,
        model_name=model_name,
    )
    logger.info("Fitted in %.1fs", time.time() - t0)

    # ── Overfitting check ─────────────────────────────────────────────────────
    print_overfitting_check(
        train_accuracy=fitted.score(X_train, y_train),
        test_accuracy=fitted.score(X_test, y_test),
        model_name=model_name,
    )

    # ── Full evaluation ───────────────────────────────────────────────────────
    return evaluate_on_test_set(
        model=fitted,
        X_test=X_test,
        y_test=y_test,
        model_name=model_name,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print(f"\n{BOLD}{CYAN}")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       Email Spam Detection — Training Pipeline               ║")
    print("║       Paper: ML-Based Email Spam Detection (EJASET 2025)     ║")
    print("║       Models: Random Forest  +  Logistic Regression CV       ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(RESET)

    t0 = time.time()

    df                           = load_and_validate(args.data)
    df                           = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(df, args.test_size, args.seed)

    all_metrics = {}

    all_metrics["Random Forest"] = train_evaluate(
        pipeline   = build_random_forest_pipeline(),
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        model_name = "Random Forest",
        save_path  = RF_MODEL_PATH,
        run_cv     = not args.no_cv,
        seed       = args.seed,
    )

    all_metrics["Logistic Regression CV"] = train_evaluate(
        pipeline   = build_logistic_regression_cv_pipeline(),
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        model_name = "Logistic Regression CV",
        save_path  = LR_MODEL_PATH,
        run_cv     = not args.no_cv,
        seed       = args.seed,
    )

    compare_models(all_metrics)

    print_section("Training Complete")
    print(f"\n  Total time     : {time.time() - t0:.1f}s")
    print(f"  RF  model saved: {RF_MODEL_PATH}")
    print(f"  LR  model saved: {LR_MODEL_PATH}")
    print(f"\n  {GREEN}Next step → run:  streamlit run app.py{RESET}\n")


if __name__ == "__main__":
    main()
