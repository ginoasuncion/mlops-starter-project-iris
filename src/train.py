from datetime import datetime

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

if __name__ == "__main__":
    mlflow.set_experiment("assignment-3-mlflow")

    # Start parent run
    with mlflow.start_run(run_name="main-train") as main_run:
        run_id = main_run.info.run_id

        # Save run ID for use in evaluation
        with open("mlflow_run.txt", "w") as f:
            f.write(run_id)

        # Timestamped nested training run
        train_run_name = f"train-step-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        with mlflow.start_run(run_name=train_run_name, nested=True):
            # Load training data
            train_dataset = pd.read_csv("data/train.csv")
            y = train_dataset["target"].values.astype("float32")
            X = train_dataset.drop("target", axis=1).values

            # Train model
            clf = LogisticRegression(C=0.01, solver="lbfgs", max_iter=10)
            clf.fit(X, y)

            # Save locally
            joblib.dump(clf, "models/model.joblib")

            # Infer model signature
            signature = infer_signature(X, clf.predict(X))

            # Log model to MLflow with signature and register it
            mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path="model",
                registered_model_name="IrisLogisticRegression",
                signature=signature,
            )

            # Log parameters and metrics
            mlflow.log_param("C", 0.01)
            mlflow.log_param("solver", "lbfgs")

            predictions = clf.predict(X)
            acc = accuracy_score(y, predictions)
            f1 = f1_score(y, predictions, average="macro")

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)

            print("Model trained and logged with accuracy and F1 score.")
