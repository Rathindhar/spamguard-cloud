# ============================================================
#  ENHANCEMENT 2 — Combined Model (TF-IDF + Behavioral)
#                  + Concept Drift Detection
#
#  This is the CORE upgrade over your basic version:
#
#  Basic version:    TF-IDF features → SVM classifier
#  Enhanced version: TF-IDF + 20 Behavioral features → 
#                    Stacked Ensemble + Drift Detection
#
#  Three major upgrades:
#  A) Feature fusion — combine content + behavior vectors
#  B) Stacked ensemble — meta-learner on top of 4 classifiers
#  C) Concept drift detection — knows when model is stale
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse            import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes     import MultinomialNB
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm             import LinearSVC
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.preprocessing   import MinMaxScaler
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────
CLEAN_PATH    = os.path.join("data", "cleaned_data.csv")
ENHANCED_PATH = os.path.join("data", "enhanced_features.pkl")
COMBINED_PATH = os.path.join("data", "combined_model.pkl")


# ════════════════════════════════════════════════════════════
#  CONCEPT DRIFT DETECTOR (Page-Hinkley Test)
#  Monitors prediction accuracy over time.
#  If accuracy drops — flags that the model needs retraining.
# ════════════════════════════════════════════════════════════
class ConceptDriftDetector:
    """
    Page-Hinkley drift detection.
    Tracks running accuracy and signals when it degrades.
    Call .update(correct) with True/False after each prediction.
    Call .drift_detected() to check if retraining is needed.
    """
    def __init__(self, delta=0.005, threshold=50, alpha=0.9999):
        self.delta     = delta       # sensitivity
        self.threshold = threshold   # alarm threshold
        self.alpha     = alpha       # smoothing factor
        self.reset()

    def reset(self):
        self.m_t     = 0.0
        self.M_t     = 0.0
        self.n       = 0
        self.sum_err = 0.0

    def update(self, correct: bool):
        """correct = True if prediction was right, False if wrong"""
        self.n += 1
        error        = 0 if correct else 1
        self.sum_err += error
        mean_err     = self.sum_err / self.n
        self.m_t     = self.m_t * self.alpha + (error - mean_err + self.delta)
        self.M_t     = max(self.M_t, self.m_t)

    def drift_detected(self) -> bool:
        return (self.M_t - self.m_t) > self.threshold

    def status(self) -> str:
        ph = self.M_t - self.m_t
        if ph > self.threshold:
            return f"🔴 DRIFT DETECTED (PH={ph:.1f}) — Model needs retraining!"
        elif ph > self.threshold * 0.7:
            return f"🟡 WARNING: Drift approaching (PH={ph:.1f})"
        else:
            return f"🟢 Stable (PH={ph:.1f})"


# ════════════════════════════════════════════════════════════
#  STACKED ENSEMBLE
#  Level 0: NB, LR, RF, SVM make predictions
#  Level 1: Logistic Regression combines their outputs
# ════════════════════════════════════════════════════════════
class StackedEnsemble:
    """
    Two-level stacking:
    - Level 0 classifiers each make a probability prediction
    - Level 1 meta-learner learns the best combination
    """
    def __init__(self):
        self.level0 = {
            "Naive Bayes"   : MultinomialNB(),
            "Logistic Reg"  : LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest" : RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM (calibrated)": CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42)),
        }
        self.meta_learner = LogisticRegression(max_iter=500, random_state=42)
        self.is_fitted = False

    def fit(self, X_tfidf, X_behavioral, y):
        """
        X_tfidf      — sparse TF-IDF matrix (content features)
        X_behavioral — dense behavioral feature matrix
        y            — labels (0=ham, 1=spam)
        """
        n = X_tfidf.shape[0]
        meta_features = np.zeros((n, len(self.level0)))

        print("\n  Training Level-0 classifiers:")
        for i, (name, clf) in enumerate(self.level0.items()):
            # NB only works on non-negative features, use tfidf only
            if "Naive" in name:
                clf.fit(X_tfidf, y)
                probs = clf.predict_proba(X_tfidf)[:, 1]
            else:
                X_combined = hstack([X_tfidf, csr_matrix(X_behavioral)])
                clf.fit(X_combined, y)
                probs = clf.predict_proba(X_combined)[:, 1]
            meta_features[:, i] = probs
            print(f"    ✅ {name}")

        print("  Training Level-1 meta-learner (Logistic Regression) ...")
        self.meta_learner.fit(meta_features, y)
        self.is_fitted = True
        print("    ✅ Meta-learner ready")

    SPAM_THRESHOLD = 0.35   # lowered from 0.5 to catch soft/subtle spam

    def predict(self, X_tfidf, X_behavioral):
        proba = self.predict_proba(X_tfidf, X_behavioral)
        return (proba[:, 1] >= self.SPAM_THRESHOLD).astype(int)

    def predict_proba(self, X_tfidf, X_behavioral):
        meta_features = self._get_meta_features(X_tfidf, X_behavioral)
        return self.meta_learner.predict_proba(meta_features)

    def _get_meta_features(self, X_tfidf, X_behavioral):
        meta = np.zeros((X_tfidf.shape[0], len(self.level0)))
        for i, (name, clf) in enumerate(self.level0.items()):
            if "Naive" in name:
                meta[:, i] = clf.predict_proba(X_tfidf)[:, 1]
            else:
                X_c = hstack([X_tfidf, csr_matrix(X_behavioral)])
                meta[:, i] = clf.predict_proba(X_c)[:, 1]
        return meta

    def individual_scores(self, X_tfidf, X_behavioral, y_true):
        """Return per-classifier metrics for comparison table"""
        scores = {}
        for name, clf in self.level0.items():
            if "Naive" in name:
                y_pred = clf.predict(X_tfidf)
            else:
                X_c    = hstack([X_tfidf, csr_matrix(X_behavioral)])
                y_pred = clf.predict(X_c)
            scores[name] = {
                "Accuracy" : accuracy_score (y_true, y_pred) * 100,
                "Precision": precision_score(y_true, y_pred, zero_division=0) * 100,
                "Recall"   : recall_score   (y_true, y_pred, zero_division=0) * 100,
                "F1-Score" : f1_score       (y_true, y_pred, zero_division=0) * 100,
            }
        return scores


# ════════════════════════════════════════════════════════════
#  MAIN TRAINING PIPELINE
# ════════════════════════════════════════════════════════════
def train_combined_model():
    print("=" * 58)
    print("  ENHANCEMENT 2 — Combined Model Training")
    print("  TF-IDF + Behavioral Features + Stacked Ensemble")
    print("=" * 58)

    # ── Load data ────────────────────────────────────────────
    if not os.path.exists(CLEAN_PATH):
        print(f"\n❌ Missing: {CLEAN_PATH}")
        print("   Run step2_preprocess.py from the basic version first.")
        return

    df      = pd.read_csv(CLEAN_PATH)
    df["label_num"] = df["label"].map({"spam": 1, "ham": 0})

    print(f"\n📂 Dataset: {len(df)} messages")

    # ── Build TF-IDF features ─────────────────────────────────
    print("\n🔢 Building TF-IDF features ...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X_text = df["clean_message"].fillna("")
    y      = df["label_num"].values

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf  = vectorizer.transform(X_test_text)

    # ── Get behavioral features (aligned to same split) ───────
    print("🔬 Aligning behavioral features ...")
    from enhance1_behavioral_features import extract_behavioral_features
    feat_rows = X_text.apply(extract_behavioral_features)
    feat_df   = pd.DataFrame(list(feat_rows))

    # ── Clean NaN / Inf before scaling ───────────────────────
    feat_df = feat_df.replace([np.inf, -np.inf], 0)
    feat_df = feat_df.fillna(0)

    # Keep only the 20 known behavioral feature columns.
    # This prevents any stale feat_0..feat_N columns from old
    # cached files mixing in and doubling the feature count.
    from enhance1_behavioral_features import extract_behavioral_features as _efa
    _sample    = _efa("test message")
    feat_names = list(_sample.keys())          # always exactly 20
    feat_df    = feat_df[feat_names]           # select only those columns
    print(f"   Feature columns confirmed: {feat_df.shape[1]} (expected 20)")

    # Scale to 0-1 range (works with sparse matrix stacking)
    mms = MinMaxScaler()
    X_behavioral_all = mms.fit_transform(feat_df)

    # Recreate the same split
    train_idx = X_train_text.index
    test_idx  = X_test_text.index
    X_train_b = X_behavioral_all[train_idx]
    X_test_b  = X_behavioral_all[test_idx]

    # ── Train stacked ensemble ────────────────────────────────
    print("\n🏗️  Building Stacked Ensemble ...")
    ensemble = StackedEnsemble()
    ensemble.fit(X_train_tfidf, X_train_b, y_train)

    # ── Evaluate ──────────────────────────────────────────────
    y_pred  = ensemble.predict(X_test_tfidf, X_test_b)
    y_proba = ensemble.predict_proba(X_test_tfidf, X_test_b)

    acc  = accuracy_score (y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, zero_division=0) * 100
    rec  = recall_score   (y_test, y_pred, zero_division=0) * 100
    f1   = f1_score       (y_test, y_pred, zero_division=0) * 100
    cm   = confusion_matrix(y_test, y_pred)

    print("\n" + "─" * 58)
    print("  FINAL ENSEMBLE RESULTS (Enhanced vs Basic Paper)")
    print("─" * 58)
    print(f"  {'Metric':<20} {'Basic SVM':>12} {'Enhanced':>12}")
    print("─" * 58)
    basic = {"Accuracy":97.1,"Precision":96.8,"Recall":96.9,"F1":96.8}
    for metric, bval in [("Accuracy",acc),("Precision",prec),("Recall",rec),("F1-Score",f1)]:
        bbasic = basic.get(metric, basic.get("F1", 96.8))
        delta  = bval - bbasic
        arrow  = "▲" if delta > 0 else "▼"
        print(f"  {metric:<20} {bbasic:>10.1f}%  {bval:>10.1f}% {arrow}{abs(delta):.1f}%")
    print("─" * 58)

    # ── Per-classifier scores ─────────────────────────────────
    print("\n  Individual classifier scores (with behavioral features):")
    ind_scores = ensemble.individual_scores(X_test_tfidf, X_test_b, y_test)
    print(f"  {'Classifier':<22} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("  " + "─" * 50)
    for name, sc in ind_scores.items():
        print(f"  {name:<22} {sc['Accuracy']:>6.1f}% {sc['Precision']:>6.1f}% "
              f"{sc['Recall']:>6.1f}% {sc['F1-Score']:>6.1f}%")
    print(f"  {'★ Stacked Ensemble':<22} {acc:>6.1f}% {prec:>6.1f}% {rec:>6.1f}% {f1:>6.1f}%")

    # ── Setup drift detector ──────────────────────────────────
    drift_detector = ConceptDriftDetector()
    for true, pred in zip(y_test, y_pred):
        drift_detector.update(true == pred)
    print(f"\n🔍 Drift Detector Status: {drift_detector.status()}")

    # ── Save everything ───────────────────────────────────────
    save_data = {
        "ensemble"       : ensemble,
        "vectorizer"     : vectorizer,
        "behavioral_scaler": mms,
        "drift_detector" : drift_detector,
        "results"        : {
            "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1-Score": f1
        },
        "individual_scores": ind_scores,
        "confusion_matrix"  : cm,
        "y_test"            : y_test,
        "y_pred"            : y_pred,
        "y_proba"           : y_proba,
        "feature_names"     : list(feat_df.columns),
    }
    with open(COMBINED_PATH, "wb") as f:
        pickle.dump(save_data, f)

    print(f"\n✅ Combined model saved to: {COMBINED_PATH}")
    print("\nNext → run:  python enhance3_explainability.py")


if __name__ == "__main__":
    train_combined_model()