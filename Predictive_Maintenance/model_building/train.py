# ============================================================
# Predictive Maintenance (XGBoost) - Train + Tune + MLflow + Upload to HF
# - Loads Xtrain/Xtest/ytrain/ytest from Hugging Face dataset space
# - Tunes XGBoost using GridSearchCV
# - Logs all tuned params (nested runs) + best metrics to MLflow
# - Dumps best model as .joblib
# - Uploads ONLY the model file to Hugging Face Model Hub (no README)
# ============================================================

# If needed (run once in a separate cell):
# %pip install -q "numpy<2" pandas scikit-learn xgboost joblib huggingface_hub mlflow==3.0.1


import os
import pandas as pd
import joblib
import mlflow

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

import xgboost as xgb

from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

# -----------------------------
# MLflow config (local)
# -----------------------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Predictive_Maintenance_mlflow")

# -----------------------------
# Hugging Face auth (ENV ONLY - SAFE for GitHub)
# -----------------------------
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("No HF token found. Set HF_TOKEN as an environment variable.")

login(token=HF_TOKEN)
api = HfApi(token=HF_TOKEN)

# -----------------------------
# Dataset + Model repo config
# -----------------------------
HF_DATASET_REPO_ID = "vinodcwanted/Predictive-Maintenance"        # dataset repo (already has Xtrain/Xtest/ytrain/ytest)
HF_MODEL_REPO_ID   = "vinodcwanted/Predictive-Maintenance"        # model repo to upload best model
REPO_TYPE = "model"
PRIVATE = False

# Hugging Face dataset paths
Xtrain_path = f"hf://datasets/{HF_DATASET_REPO_ID}/Xtrain.csv"
Xtest_path  = f"hf://datasets/{HF_DATASET_REPO_ID}/Xtest.csv"
ytrain_path = f"hf://datasets/{HF_DATASET_REPO_ID}/ytrain.csv"
ytest_path  = f"hf://datasets/{HF_DATASET_REPO_ID}/ytest.csv"

# Use token explicitly for reliable hf:// reads
HF_STORAGE_OPTIONS = {"token": HF_TOKEN}

# -----------------------------
# Load train/test
# -----------------------------
Xtrain = pd.read_csv(Xtrain_path, storage_options=HF_STORAGE_OPTIONS)
Xtest  = pd.read_csv(Xtest_path,  storage_options=HF_STORAGE_OPTIONS)
ytrain = pd.read_csv(ytrain_path, storage_options=HF_STORAGE_OPTIONS).squeeze("columns").astype(int)
ytest  = pd.read_csv(ytest_path,  storage_options=HF_STORAGE_OPTIONS).squeeze("columns").astype(int)

print("✅ Data loaded")
print("Xtrain:", Xtrain.shape, "Xtest:", Xtest.shape)
print("ytrain:", ytrain.shape, "ytest:", ytest.shape)

# -----------------------------
# Feature handling
# - In your engine dataset, features are numeric sensor values.
# - We'll treat ALL columns in Xtrain as numeric.
# -----------------------------
numeric_features = list(Xtrain.columns)
categorical_features = []  # none expected here

# Optional scaling (not required for XGBoost, but keeps consistent pipeline style)
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    remainder="drop"
)

# -----------------------------
# Handle class imbalance
# scale_pos_weight = (#neg / #pos)
# -----------------------------
pos = (ytrain == 1).sum()
neg = (ytrain == 0).sum()
scale_pos_weight = (neg / pos) if pos > 0 else 1.0
print("scale_pos_weight:", scale_pos_weight)

# -----------------------------
# Model + Hyperparameter Grid
# -----------------------------
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight
)

param_grid_deploy = {
    "xgbclassifier__n_estimators": [200],
    "xgbclassifier__max_depth": [3],
    "xgbclassifier__learning_rate": [0.05],
    "xgbclassifier__subsample": [0.8],
    "xgbclassifier__colsample_bytree": [0.8, 0.9],
    "xgbclassifier__min_child_weight": [3],
    "xgbclassifier__gamma": [0, 0.2],
    "xgbclassifier__reg_lambda": [1.0],
    "xgbclassifier__scale_pos_weight": [2.0],
}

# Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Output model name
MODEL_FILE = "best_engine_xgb_model.joblib"

# Classification threshold
classification_threshold = 0.50  # change to 0.45 if you prefer

# -----------------------------
# Train + Tune + Log + Save + Upload
# -----------------------------
SCORING_METRIC ="recall"
with mlflow.start_run(run_name="xgb_gridsearch_main"):

    # Log basic run metadata
    mlflow.log_param("dataset_repo", HF_DATASET_REPO_ID)
    mlflow.log_param("features_count", Xtrain.shape[1])
    mlflow.log_param("train_rows", Xtrain.shape[0])
    mlflow.log_param("test_rows", Xtest.shape[0])
    mlflow.log_param("scale_pos_weight", float(scale_pos_weight))
    mlflow.log_param("scoring_metric", SCORING_METRIC)
    mlflow.log_param("classification_threshold", classification_threshold)

    # GridSearch
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid_deploy,
        scoring=SCORING_METRIC,
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(Xtrain, ytrain)

    # Log all tuned combinations as nested MLflow runs
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["mean_test_score"][i]

        with mlflow.start_run(nested=True, run_name=f"grid_run_{i}"):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_cv_f1", float(mean_score))
            mlflow.log_metric("std_cv_f1", float(std_score))

    # Best params
    mlflow.log_params({f"best__{k}": v for k, v in grid_search.best_params_.items()})
    print("✅ Best params:", grid_search.best_params_)

    # Best model
    best_model = grid_search.best_estimator_

    # Evaluate
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_test_proba  = best_model.predict_proba(Xtest)[:, 1]

    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)
    y_pred_test  = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report  = classification_report(ytest, y_pred_test, output_dict=True)

    train_auc = roc_auc_score(ytrain, y_pred_train_proba)
    test_auc  = roc_auc_score(ytest, y_pred_test_proba)

    mlflow.log_metrics({
        "train_accuracy": float(train_report["accuracy"]),
        "train_precision_pos": float(train_report["1"]["precision"]),
        "train_recall_pos": float(train_report["1"]["recall"]),
        "train_f1_pos": float(train_report["1"]["f1-score"]),
        "train_auc": float(train_auc),

        "test_accuracy": float(test_report["accuracy"]),
        "test_precision_pos": float(test_report["1"]["precision"]),
        "test_recall_pos": float(test_report["1"]["recall"]),
        "test_f1_pos": float(test_report["1"]["f1-score"]),
        "test_auc": float(test_auc),
    })

    print("\n✅ TRAIN REPORT:\n", classification_report(ytrain, y_pred_train))
    print("\n✅ TEST REPORT:\n", classification_report(ytest, y_pred_test))
    print("\nAUC - Train:", train_auc, "| Test:", test_auc)

    # Save model locally
    joblib.dump(best_model, MODEL_FILE)
    mlflow.log_artifact(MODEL_FILE, artifact_path="model")
    print(f"\n✅ Model saved as artifact at: {MODEL_FILE}")

    # -----------------------------
    # Upload to Hugging Face Model Hub (ONLY MODEL FILE)
    # -----------------------------
    try:
        api.repo_info(repo_id=HF_MODEL_REPO_ID, repo_type=REPO_TYPE)
        print(f"✅ Model repo '{HF_MODEL_REPO_ID}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"ℹ️ Model repo '{HF_MODEL_REPO_ID}' not found. Creating new repo...")
        create_repo(repo_id=HF_MODEL_REPO_ID, repo_type=REPO_TYPE, private=PRIVATE)
        print(f"✅ Model repo '{HF_MODEL_REPO_ID}' created.")

    try:
        api.upload_file(
            path_or_fileobj=MODEL_FILE,
            path_in_repo=MODEL_FILE,
            repo_id=HF_MODEL_REPO_ID,
            repo_type=REPO_TYPE,
        )
        print(f"\n🎉 Uploaded model to Hugging Face: {HF_MODEL_REPO_ID}/{MODEL_FILE}")
    except HfHubHTTPError as e:
        print(f"❌ Upload failed: {e}")
        raise
