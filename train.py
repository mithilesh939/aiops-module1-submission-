import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# connect to the local tracking server (started separately)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("mnist-mlp-experiments")
print("Tracking URI:", mlflow.get_tracking_uri())

# load MNIST digits dataset and split
X, y = load_digits(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# base training function
def train_and_evaluate(hidden_layer_sizes, learning_rate, batch_size):
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate_init=learning_rate,
        batch_size=batch_size,
        max_iter=500,
        solver="adam",
        random_state=42,
        early_stopping=True,
    )
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc   = accuracy_score(y_val,   model.predict(X_val))
    test_acc  = accuracy_score(y_test,  model.predict(X_test))
    f1        = f1_score(y_test, model.predict(X_test), average="macro")
    return model, train_acc, val_acc, test_acc, f1

# instrumented logging function
def train_and_log(hidden_layer_sizes, learning_rate, batch_size, run_name=None):
    with mlflow.start_run(run_name=run_name):

        # log parameters
        mlflow.log_param("hidden_layer_sizes", str(hidden_layer_sizes))
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("solver", "adam")
        mlflow.log_param("random_state", 42)

        model, train_acc, val_acc, test_acc, f1 = train_and_evaluate(
            hidden_layer_sizes, learning_rate, batch_size
        )

        # log metrics
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("f1_macro", f1)

        mlflow.set_tag("dataset", "MNIST-digits")

        # log model with trusted types
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            skops_trusted_types=["sklearn.neural_network._stochastic_optimizers.AdamOptimizer"]
        )

        run_id = mlflow.active_run().info.run_id
        print(f"run={run_name}  id={run_id}  train={train_acc:.4f}  val={val_acc:.4f}  test={test_acc:.4f}")
        return run_id

# run 1 — baseline
train_and_log((128, 64), 0.001, 32, run_name="mlp-baseline")

# runs 2-3 — vary learning_rate
train_and_log((128, 64), 0.01,   32,  run_name="mlp-lr-0.01")
train_and_log((128, 64), 0.0001, 32,  run_name="mlp-lr-0.0001")

# runs 4-5 — vary batch_size
train_and_log((128, 64), 0.001, 16,  run_name="mlp-batch-16")
train_and_log((128, 64), 0.001, 128, run_name="mlp-batch-128")

# run 6 — vary architecture (deeper network)
train_and_log((256, 128, 64), 0.001, 32, run_name="mlp-deeper")

# find best run programmatically
runs_df = mlflow.search_runs(
    experiment_names=["mnist-mlp-experiments"],
    order_by=["metrics.val_accuracy DESC"],
)

display_cols = [c for c in runs_df.columns if c in (
    "run_id", "tags.mlflow.runName",
    "params.learning_rate", "params.batch_size",
    "metrics.train_accuracy", "metrics.val_accuracy", "metrics.test_accuracy"
)]
print("\n", runs_df[display_cols].head(10).to_string(index=False))

best = runs_df.iloc[0]
print(f"\nBest run: {best['run_id']}  val_accuracy={best['metrics.val_accuracy']:.4f}")
