import subprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# 1. Configure MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("q4-reproducibility-drill")

# Helper function
def get_git_commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "initial_commit"

# 2. Fixed parameters & Seed
SEED = 42
N_ESTIMATORS = 50
MAX_DEPTH = 3

# 3. Training Loop & MLflow Instrumentation
with mlflow.start_run(run_name="partner_a_baseline") as run:
    run_id = run.info.run_id
    
    # Log Parameters & Seed
    mlflow.log_param("seed", SEED)
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("max_depth", MAX_DEPTH)
    
    # Log Git Commit Tag
    mlflow.set_tag("git_commit", get_git_commit_hash())
    mlflow.set_tag("author", "Partner_A")
    
    # Load dataset tracked by DVC
    df = pd.read_csv("data/dataset.csv")
    X = df.drop(columns=["target"])
    y = df["target"]
    
    # Split deterministically using fixed SEED
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, 
        max_depth=MAX_DEPTH, 
        random_state=SEED
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    # Log Metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    
    # Log Model Artifact
    mlflow.sklearn.log_model(model, artifact_path="model")
    
    print(f"Partner A Training Finished.")
    print(f"Run ID: {run_id}")
    print(f"Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")

# 4. Register Model and Transition to Staging
model_name = "IrisClassifier_Q4"
model_uri = f"runs:/{run_id}/model"
registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)

client = MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=registered_model.version,
    stage="Staging"
)

print(f"Model '{model_name}' version {registered_model.version} promoted to STAGING.")