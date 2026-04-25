"""
trainer.py
───────────
Defines, fits, and persists the two champion models from your paper:

  1. RandomForestClassifier — best robustness (std: 0.0058, Table 3)
  2. LogisticRegressionCV   — best raw accuracy (0.9845, Table 3)

Both are wrapped in sklearn Pipelines that include the TF-IDF vectoriser,
making each saved artefact a fully self-contained predictor.

BUG FIX (v2):
  - MODEL_DIR now correctly resolves relative to this file's location
    (src/ → parent → spam_detector/models/)
  - LogisticRegressionCV is now version-safe across sklearn 1.4 – 1.9+
    (the penalty/solver API changed in sklearn 1.8)
"""

import logging
import warnings
from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# ── sklearn version detection ─────────────────────────────────────────────────
# The LogisticRegressionCV penalty API changed in sklearn 1.8.
# We detect the version once at import time and branch accordingly.
_sklearn_ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
_SKLEARN_GTE_18 = _sklearn_ver >= (1, 8)

# ── Default paths ─────────────────────────────────────────────────────────────
# __file__ = spam_detector/src/trainer.py
# .parent  = spam_detector/src/
# .parent.parent = spam_detector/          ← project root
# / "models"  = spam_detector/models/      ← where .pkl files are saved
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RF_MODEL_PATH = MODEL_DIR / "model_rf.pkl"
LR_MODEL_PATH = MODEL_DIR / "model_lr.pkl"


# ── TF-IDF configuration ──────────────────────────────────────────────────────

def build_tfidf_vectoriser() -> TfidfVectorizer:
    """
    Build the TF-IDF vectoriser with parameters matched to your paper.

    Key choices:
      max_features=5000   → Your paper's exact setting.
      ngram_range=(1, 2)  → Adds bigrams ("free money", "click here") which
                            are strong spam signals missed by unigrams alone.
      sublinear_tf=True   → Applies log(1 + tf) to dampen high-frequency terms.
      min_df=2            → Ignores terms in < 2 documents (noise).
      max_df=0.95         → Ignores terms in > 95% of documents (near-stop-words).
      strip_accents        → Handles international characters in spam.
    """
    return TfidfVectorizer(
        max_features=5_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
        analyzer="word",
    )


# ── Model 1: Random Forest ────────────────────────────────────────────────────

def build_random_forest_pipeline() -> Pipeline:
    """
    Build a Pipeline: TF-IDF → Random Forest.

    Hyperparameters grounded in your paper's robustness analysis:
      n_estimators=300      → 300 trees; more = lower variance.
      min_samples_leaf=2    → Prevents single-sample leaves that memorise noise.
      max_features='sqrt'   → Classic RF diversity mechanism.
      class_weight='balanced' → Compensates for spam/ham imbalance.
      n_jobs=-1             → Use all CPU cores.
      random_state=42       → Reproducible results.
    """
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline([
        ("tfidf",       build_tfidf_vectoriser()),
        ("classifier",  rf),
    ])


# ── Model 2: Logistic Regression CV ──────────────────────────────────────────

def build_logistic_regression_cv_pipeline() -> Pipeline:
    """
    Build a Pipeline: TF-IDF → LogisticRegressionCV.

    BUG FIX: sklearn 1.8 deprecated the penalty='l2' + solver='lbfgs' combo
    and introduced l1_ratios + use_legacy_attributes.  This function is
    version-safe: it detects sklearn version at runtime and uses the correct
    API for the installed version, so the code works on sklearn 1.4 through 1.9+
    without any deprecation warnings or errors.

    Effective configuration in both cases:
      • Pure L2 (Ridge) regularisation  ← your paper confirms L2 is optimal
      • 10 C-values searched via 5-fold inner CV
      • class_weight='balanced' for imbalanced spam/ham data
    """
    common_kwargs = dict(
        Cs=10,
        cv=5,
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    if _SKLEARN_GTE_18:
        # sklearn >= 1.8 new API: l1_ratios replaces penalty argument
        # l1_ratios=0.0 → pure L2/Ridge (equivalent to old penalty='l2')
        lr = LogisticRegressionCV(
            solver="saga",
            l1_ratios=[0.0],
            use_legacy_attributes=False,
            **common_kwargs,
        )
    else:
        # sklearn < 1.8 classic API
        lr = LogisticRegressionCV(
            penalty="l2",
            solver="lbfgs",
            **common_kwargs,
        )

    return Pipeline([
        ("tfidf",      build_tfidf_vectoriser()),
        ("classifier", lr),
    ])


# ── Fit and save ──────────────────────────────────────────────────────────────

def train_and_save(
    pipeline: Pipeline,
    X_train,
    y_train,
    save_path: Path,
    model_name: str = "Model",
) -> Pipeline:
    """
    Fit a pipeline on training data and persist it to disk with joblib.

    Args:
        pipeline:   Unfitted sklearn Pipeline.
        X_train:    Training text strings (after preprocessing).
        y_train:    Training labels ('spam' or 'ham').
        save_path:  Where to save the fitted pipeline (.pkl).
        model_name: Display name for log messages.

    Returns:
        The fitted pipeline (ready for prediction).
    """
    logger.info("Training %s on %d samples …", model_name, len(X_train))
    pipeline.fit(X_train, y_train)
    logger.info("Training complete.")

    joblib.dump(pipeline, save_path)
    logger.info("Saved %s → %s", model_name, save_path)

    return pipeline


# ── Load helpers ──────────────────────────────────────────────────────────────

def load_pipeline(path: Path) -> Pipeline:
    """
    Load a previously saved pipeline from disk.

    Args:
        path: Path to .pkl file.

    Returns:
        Fitted sklearn Pipeline.

    Raises:
        FileNotFoundError if the model hasn't been trained yet.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Run  python train.py  first to train and save the models."
        )
    pipeline = joblib.load(path)
    logger.info("Loaded pipeline from %s", path)
    return pipeline


def load_best_pipeline():
    """
    Load the model recommended for production (Logistic Regression CV first,
    then Random Forest as fallback).

    Returns:
        (fitted_pipeline, model_name_string)
    """
    for path, name in [
        (LR_MODEL_PATH, "Logistic Regression CV"),
        (RF_MODEL_PATH, "Random Forest"),
    ]:
        if path.exists():
            return load_pipeline(path), name

    raise FileNotFoundError(
        "No trained models found in models/. Run  python train.py  first."
    )
