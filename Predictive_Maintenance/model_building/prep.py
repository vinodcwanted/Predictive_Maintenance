
# ---------------------------------------------
# Data Preparation for Predictive Maintenance
# ---------------------------------------------
# - Load dataset directly from Hugging Face dataset repo
# - Clean data (duplicates, missing values, drop unnecessary cols)
# - Split into train/test and save locally
# - Upload train/test back to Hugging Face dataset repo
# ---------------------------------------------

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# ---------------------------
# 🔹 CONFIG (EDIT THESE 2)
# ---------------------------

# 1) Your HF dataset repo (where you want to upload train/test CSVs)
HF_DATASET_REPO_ID = "vinodcwanted/Predictive-Maintenance"

# 2) Path to the raw CSV inside HF datasets OR a direct HF file path
# Option A (recommended): hf://datasets/<user>/<repo>/<file>.csv
DATASET_PATH = "hf://datasets/vinodcwanted/Predictive-Maintenance/engine_data.csv"

# Train/test output names (kept simple for model_building step)
XTRAIN_FILE = "Xtrain.csv"
XTEST_FILE = "Xtest.csv"
YTRAIN_FILE = "ytrain.csv"
YTEST_FILE = "ytest.csv"

# Need to be changed for github


# Create HF API client (expects HF_TOKEN set in environment)
api = HfApi(token=os.getenv("HF_TOKEN"))

# ---------------------------
# 🔹 LOAD DATA
# ---------------------------
df = pd.read_csv(DATASET_PATH)
print("✅ Dataset loaded successfully.")
print("Shape:", df.shape)
print("Columns:", list(df.columns))

# ---------------------------
# 🔹 DATA CLEANING
# ---------------------------

# 1) Drop duplicate rows
df = df.drop_duplicates()

# 2) Standardize column names (optional but helpful)
df.columns = [c.strip() for c in df.columns]


# 4) Ensure target exists + standardize it
TARGET_COL = "Engine Condition"

if TARGET_COL not in df.columns:
    raise ValueError(f"❌ Target column '{TARGET_COL}' not found. Found: {list(df.columns)}")

# Convert Engine_Condition to 0/1 safely
# Handles values like: 0/1, "0"/"1", "Normal"/"Faulty", "False"/"True", etc.
def to_binary_target(x):
    if pd.isna(x):
        return None

    s = str(x).strip().lower()

    # numeric strings
    if s in {"0", "0.0"}:
        return 0
    if s in {"1", "1.0"}:
        return 1

    # common text labels (extend if your dataset uses different words)
    if s in {"false", "off", "normal", "ok", "healthy", "good"}:
        return 0
    if s in {"true", "on", "faulty", "fail", "failure", "bad", "maintenance", "needs_maintenance"}:
        return 1

    # try numeric conversion fallback
    try:
        v = float(s)
        return 1 if v >= 0.5 else 0
    except:
        return None


df[TARGET_COL] = df[TARGET_COL].apply(to_binary_target)

# Drop rows where target couldn't be parsed
before = df.shape[0]
df = df.dropna(subset=[TARGET_COL])
after = df.shape[0]

if before != after:
    print(f"ℹ️ Dropped {before - after} rows with invalid/missing target.")

df[TARGET_COL] = df[TARGET_COL].astype(int)

# 5) Handle missing feature values
# Strategy: drop columns with too many missing values; fill remaining numeric with median.

missing_ratio = df.isna().mean()
too_missing_cols = missing_ratio[missing_ratio > 0.40].index.tolist()  # threshold adjustable

# never drop target
too_missing_cols = [c for c in too_missing_cols if c != TARGET_COL]

if too_missing_cols:
    df = df.drop(columns=too_missing_cols)
    print("ℹ️ Dropped columns with >40% missing:", too_missing_cols)

# Fill numeric cols with median; non-numeric with mode
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]

for c in num_cols:
    if c == TARGET_COL:
        continue
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median())

for c in cat_cols:
    if c == TARGET_COL:
        continue
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].mode().iloc[0])

df = df.reset_index(drop=True)

print("✅ Data cleaned successfully.")
print("Cleaned shape:", df.shape)

# ---------------------------
# 🔹 FEATURE / TARGET SPLIT
# ---------------------------
y = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL])

# Optional: keep only the expected engine sensor columns if they exist
expected_features = [
    "Engine RPM",
    "Lub_Oil Pressure",
    "Fuel Pressure",
    "Coolant Pressure",
    "Lub_Oil Temperature",
    "Coolant Temperature",
]

available_expected = [c for c in expected_features if c in X.columns]

if len(available_expected) >= 2:
    X = X[available_expected]
    print("ℹ️ Using expected sensor features:", available_expected)
else:
    print("ℹ️ Expected sensor feature list not fully found. Using all remaining columns:", list(X.columns))

# ---------------------------
# 🔹 TRAIN-TEST SPLIT
# ---------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y  # keeps class balance
)

# Save split datasets locally
Xtrain.to_csv(XTRAIN_FILE, index=False)
Xtest.to_csv(XTEST_FILE, index=False)
ytrain.to_csv(YTRAIN_FILE, index=False)
ytest.to_csv(YTEST_FILE, index=False)

print("✅ Train/test files saved locally:")
print([XTRAIN_FILE, XTEST_FILE, YTRAIN_FILE, YTEST_FILE])

# ---------------------------
# 🔹 UPLOAD BACK TO HF DATASET REPO
# ---------------------------
files_to_upload = [XTRAIN_FILE, XTEST_FILE, YTRAIN_FILE, YTEST_FILE]

for file_path in files_to_upload:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,  # uploads to repo root
        repo_id=HF_DATASET_REPO_ID,
        repo_type="dataset",
    )

print(f"✅ Uploaded train/test files to HF dataset repo: {HF_DATASET_REPO_ID}")
