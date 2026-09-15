"""
Train, compare, and tune churn prediction models.

Trains Logistic Regression, Random Forest, and XGBoost, handles class
imbalance with SMOTE, tunes the best-performing model with
RandomizedSearchCV, evaluates on a held-out test set, and saves:
  - the final model + preprocessing pipeline (models/churn_model.joblib)
  - ROC curve, confusion matrix, feature importance plots (reports/figures/)
  - a model comparison table + final metrics (reports/metrics.json, .md)
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier

from preprocessing import build_preprocessor, load_and_engineer

FIG_DIR = "reports/figures"
MODEL_PATH = "models/churn_model.joblib"
METRICS_JSON = "reports/metrics.json"
RANDOM_STATE = 42


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def main():
    df = load_and_engineer()
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor, cat_cols, num_cols = build_preprocessor(df)

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }

    results = []
    fitted_pipelines = {}

    print("=== Training & comparing baseline models (with SMOTE) ===")
    for name, clf in candidates.items():
        pipe = ImbPipeline(
            steps=[
                ("preprocess", preprocessor),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("clf", clf),
            ]
        )
        t0 = time.time()
        pipe.fit(X_train, y_train)
        elapsed = time.time() - t0
        metrics = evaluate(name, pipe, X_test, y_test)
        metrics["train_time_sec"] = round(elapsed, 2)
        results.append(metrics)
        fitted_pipelines[name] = pipe
        print(f"{name}: {metrics}")

    comparison_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    comparison_df.to_csv("reports/model_comparison.csv", index=False)
    print("\n=== Model comparison (sorted by ROC-AUC) ===")
    print(comparison_df)

    best_model_name = comparison_df.iloc[0]["model"]
    print(f"\nBest baseline model: {best_model_name}")

    # --- Hyperparameter tuning on the best model ---
    print(f"\n=== Hyperparameter tuning: {best_model_name} ===")
    if best_model_name == "XGBoost":
        param_dist = {
            "clf__n_estimators": [200, 300, 400, 500],
            "clf__max_depth": [3, 4, 5, 6, 8],
            "clf__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "clf__subsample": [0.6, 0.8, 1.0],
            "clf__colsample_bytree": [0.6, 0.8, 1.0],
            "clf__min_child_weight": [1, 3, 5],
        }
        base_clf = XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1
        )
    elif best_model_name == "Random Forest":
        param_dist = {
            "clf__n_estimators": [200, 300, 400, 500],
            "clf__max_depth": [None, 5, 10, 15, 20],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", "log2"],
        }
        base_clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    else:
        param_dist = {
            "clf__C": [0.01, 0.1, 0.5, 1, 5, 10],
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs", "liblinear"],
        }
        base_clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    tuning_pipe = ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", base_clf),
        ]
    )

    search = RandomizedSearchCV(
        tuning_pipe,
        param_distributions=param_dist,
        n_iter=25,
        scoring="roc_auc",
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print(f"Best params: {search.best_params_}")
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")

    best_pipe = search.best_estimator_
    final_metrics = evaluate(f"{best_model_name} (Tuned)", best_pipe, X_test, y_test)
    print(f"\n=== Final tuned model test performance ===\n{final_metrics}")

    # Compare tuned vs untuned baseline; keep whichever is better on test ROC-AUC
    baseline_metrics = [r for r in results if r["model"] == best_model_name][0]
    if final_metrics["roc_auc"] >= baseline_metrics["roc_auc"]:
        chosen_pipe = best_pipe
        chosen_metrics = final_metrics
        chosen_label = f"{best_model_name} (Tuned)"
    else:
        chosen_pipe = fitted_pipelines[best_model_name]
        chosen_metrics = baseline_metrics
        chosen_label = best_model_name
    print(f"\nSelected final model: {chosen_label}")

    # --- Save plots ---
    y_proba = chosen_pipe.predict_proba(X_test)[:, 1]
    y_pred = chosen_pipe.predict(X_test)

    # ROC curve
    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name=chosen_label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title("ROC Curve - Final Model", fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/roc_curve.png", dpi=150)
    plt.close(fig)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["No Churn", "Churn"], cmap="Blues", ax=ax
    )
    ax.set_title("Confusion Matrix - Final Model", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Feature importance
    feature_names = chosen_pipe.named_steps["preprocess"].get_feature_names_out()
    clf = chosen_pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        importances = None

    if importances is not None:
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="#4C72B0")
        ax.set_title("Top 15 Feature Importances - Final Model", fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(f"{FIG_DIR}/feature_importance.png", dpi=150)
        plt.close(fig)
        imp_df.to_csv("reports/feature_importance.csv", index=False)

    # --- Save model + metrics ---
    joblib.dump(chosen_pipe, MODEL_PATH)
    print(f"\nSaved final model pipeline to {MODEL_PATH}")

    final_report = {
        "selected_model": chosen_label,
        "test_metrics": chosen_metrics,
        "best_params": search.best_params_ if chosen_label.endswith("(Tuned)") else None,
        "cv_roc_auc": search.best_score_ if chosen_label.endswith("(Tuned)") else None,
        "baseline_comparison": results,
    }
    with open(METRICS_JSON, "w") as f:
        json.dump(final_report, f, indent=2, default=str)
    print(f"Saved metrics to {METRICS_JSON}")

    return final_report


if __name__ == "__main__":
    main()
