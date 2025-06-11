import json
from datetime import datetime
from typing import Any

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def evaluate_model() -> dict[str, Any]:
    """Evaluate trained model and return metrics."""
    classes = pd.read_csv("data/features_iris.csv")["target"].unique().tolist()
    test_dataset = pd.read_csv("data/test.csv")
    y = test_dataset["target"].values.astype("float32")
    X = test_dataset.drop("target", axis=1).values

    clf = joblib.load("models/model.joblib")
    prediction = clf.predict(X)

    cm = confusion_matrix(y, prediction)
    f1 = f1_score(y, prediction, average="macro")
    acc = accuracy_score(y, prediction)

    return {
        "accuracy": acc,
        "f1_score": f1,
        "confusion_matrix": {"classes": classes, "matrix": cm.tolist()},
    }


if __name__ == "__main__":
    # Load run ID from train step
    with open("mlflow_run.txt") as f:
        run_id = f.read().strip()

    mlflow.set_experiment("assignment-3-mlflow")

    with mlflow.start_run(run_id=run_id):
        eval_run_name = f"eval-step-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        with mlflow.start_run(run_name=eval_run_name, nested=True):
            metrics = evaluate_model()

            # Log evaluation metrics
            mlflow.log_metric("accuracy", metrics["accuracy"])
            mlflow.log_metric("f1_score", metrics["f1_score"])

            # Save confusion matrix to MLflow and to file
            mlflow.log_text(
                json.dumps(metrics["confusion_matrix"], indent=2),
                "confusion_matrix.json",
            )

            # Save metrics to local JSON
            with open("data/eval.json", "w") as f:
                json.dump(metrics, f, indent=2)

            print("Evaluation complete. Accuracy and F1 score logged.")
