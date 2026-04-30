import json
import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.utils import to_categorical
import argparse



from src.data import load_dataset
from src.models import make_reconstruction_head, make_classification_head
from src.metrics import (
    evaluate_logistic_classification,
    evaluate_linear_reconstruction
)
from src.concrete_autoencoder import ConcreteAutoencoderFeatureSelector


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def select_features_unsupervised(data, k, seed, epochs=100):
    input_dim = data["input_dim"]

    selector = ConcreteAutoencoderFeatureSelector(
        K=k,
        output_function=make_reconstruction_head(input_dim),
        loss="mean_squared_error",
        num_epochs=epochs,
        batch_size=256,
        learning_rate=1e-3,
        start_temp=10.0,
        min_temp=0.01,
        tryout_limit=1,
    )

    selector.fit(
        data["x_train"],
        data["x_train"],
        data["x_val"],
        data["x_val"],
    )

    return selector.get_support(indices=True)


def select_features_supervised(data, k, seed, epochs=100):
    num_classes = data["num_classes"]

    selector = ConcreteAutoencoderFeatureSelector(
        K=k,
        output_function=make_classification_head(num_classes),
        loss="categorical_crossentropy",
        num_epochs=epochs,
        batch_size=256,
        learning_rate=1e-3,
        start_temp=10.0,
        min_temp=0.01,
        tryout_limit=2,
    )

    selector.fit(
        data["x_train"],
        data["y_train_oh"],
        data["x_val"],
        data["y_val_oh"],
    )

    return selector.get_support(indices=True)


def evaluate_subset(data, indices, subset_name, k, seed):
    x_train_sel = data["x_train"][:, indices]
    x_val_sel = data["x_val"][:, indices]
    x_test_sel = data["x_test"][:, indices]

    results = {
        "subset": subset_name,
    }

    # Linear reconstruction head, retrained from scratch
    results.update(
        evaluate_linear_reconstruction(
            x_train_sel=x_train_sel,
            x_test_sel=x_test_sel,
            x_train_full=data["x_train"],
            x_test_full=data["x_test"],
        )
)

    # Logistic classification head, retrained from scratch
    results.update(
        evaluate_logistic_classification(
            x_train_sel=x_train_sel,
            x_test_sel=x_test_sel,
            y_train=data["y_train"],   # ✅ labels (not one-hot)
            y_test=data["y_test"],     # ✅ labels
        )
    )
    
    return results


def run_single_experiment(dataset_name="fashion_mnist", seed=0, k=20, selector_epochs=100):
    set_seed(seed)

    data = load_dataset(dataset_name, seed=seed)

    unsup_idx = select_features_unsupervised(
        data=data,
        k=k,
        seed=seed,
        epochs=selector_epochs,
    )

    sup_idx = select_features_supervised(
        data=data,
        k=k,
        seed=seed,
        epochs=selector_epochs,
    )

    rows = []

    for subset_name, indices in [
        ("unsupervised_cae", unsup_idx),
        ("supervised_cae", sup_idx),
    ]:
        row = {
            "dataset": dataset_name,
            "seed": seed,
            "k": k,
            "indices": list(map(int, indices)),
        }

        row.update(evaluate_subset(data, indices, subset_name, k, seed))

        rows.append(row)

    return rows


def summarize_results(rows):
    df = pd.DataFrame(rows)

    metric_cols = [
        col for col in df.columns
        if col not in ["dataset", "seed", "k", "subset", "indices"]
    ]

    summary = (
        df
        .groupby(["dataset", "k", "subset"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    return df, summary


def run_multiple_experiments(
    datasets,
    seeds,
    k_by_dataset,
    selector_epochs=200,
    output_dir="results",
):
    os.makedirs(output_dir, exist_ok=True)

    all_rows = []

    for dataset_name in datasets:
        k = k_by_dataset[dataset_name]

        for seed in seeds:
            print(f"Running dataset={dataset_name}, seed={seed}, k={k}")

            rows = run_single_experiment(
                dataset_name=dataset_name,
                seed=seed,
                k=k,
                selector_epochs=selector_epochs,
            )

            all_rows.extend(rows)

            with open(os.path.join(output_dir, "raw_results.json"), "w") as f:
                json.dump(all_rows, f, indent=2)

    df, summary = summarize_results(all_rows)

    df.to_csv(os.path.join(output_dir, "raw_results.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "summary_mean_std.csv"), index=False)

    return df, summary


def run_multiple_experiments_generalization(
    seeds,
    k=50,
    selector_epochs=200,
    output_dir="results_generalization",
):
    os.makedirs(output_dir, exist_ok=True)

    all_rows = []

    for seed in seeds:
        print(f"Running generalization experiment, seed={seed}, k={k}")
        set_seed(seed)

        mnist = load_dataset("mnist", seed=seed)
        fashion = load_dataset("fashion_mnist", seed=seed)

        # -----------------------------
        # Build combined train/val/test
        # -----------------------------
        combined = {
            "x_train": np.concatenate([mnist["x_train"], fashion["x_train"]], axis=0),
            "x_val": np.concatenate([mnist["x_val"], fashion["x_val"]], axis=0),
            "x_test": np.concatenate([mnist["x_test"], fashion["x_test"]], axis=0),
            "input_dim": mnist["input_dim"],
        }

        # Domain labels: 0 = MNIST, 1 = Fashion-MNIST
        combined["y_train_domain"] = np.concatenate([
            np.zeros(len(mnist["x_train"]), dtype=int),
            np.ones(len(fashion["x_train"]), dtype=int),
        ])
        combined["y_val_domain"] = np.concatenate([
            np.zeros(len(mnist["x_val"]), dtype=int),
            np.ones(len(fashion["x_val"]), dtype=int),
        ])
        combined["y_test_domain"] = np.concatenate([
            np.zeros(len(mnist["x_test"]), dtype=int),
            np.ones(len(fashion["x_test"]), dtype=int),
        ])

        # Full 20-class labels:
        # MNIST: 0–9
        # Fashion-MNIST: 10–19
        combined["y_train_full"] = np.concatenate([
            mnist["y_train"],
            fashion["y_train"] + 10,
        ])
        combined["y_val_full"] = np.concatenate([
            mnist["y_val"],
            fashion["y_val"] + 10,
        ])
        combined["y_test_full"] = np.concatenate([
            mnist["y_test"],
            fashion["y_test"] + 10,
        ])

        combined["y_train_domain_oh"] = to_categorical(combined["y_train_domain"], 2)
        combined["y_val_domain_oh"] = to_categorical(combined["y_val_domain"], 2)

        combined["y_train_full_oh"] = to_categorical(combined["y_train_full"], 20)
        combined["y_val_full_oh"] = to_categorical(combined["y_val_full"], 20)

        # ---------------------------------------------------
        # 1) Unsupervised CAE on combined data: reconstruct X
        # ---------------------------------------------------
        data_unsup = {
            "x_train": combined["x_train"],
            "x_val": combined["x_val"],
            "x_test": combined["x_test"],
            "y_train": combined["y_train_full"],
            "y_val": combined["y_val_full"],
            "y_test": combined["y_test_full"],
            "y_train_oh": combined["y_train_full_oh"],
            "y_val_oh": combined["y_val_full_oh"],
            "input_dim": combined["input_dim"],
            "num_classes": 20,
        }

        unsup_idx = select_features_unsupervised(
            data=data_unsup,
            k=k,
            seed=seed,
            epochs=selector_epochs,
        )

        # ---------------------------------------------------
        # 2) Supervised CAE: domain classification MNIST/Fashion
        # ---------------------------------------------------
        data_domain = {
            "x_train": combined["x_train"],
            "x_val": combined["x_val"],
            "x_test": combined["x_test"],
            "y_train": combined["y_train_domain"],
            "y_val": combined["y_val_domain"],
            "y_test": combined["y_test_domain"],
            "y_train_oh": combined["y_train_domain_oh"],
            "y_val_oh": combined["y_val_domain_oh"],
            "input_dim": combined["input_dim"],
            "num_classes": 2,
        }

        domain_idx = select_features_supervised(
            data=data_domain,
            k=k,
            seed=seed,
            epochs=selector_epochs,
        )


        feature_sets = {
            "unsupervised": unsup_idx,
            "supervised": domain_idx,
        }

        # ---------------------------------------------------
        # Transfer evaluation:
        # use selected features to classify all 20 classes
        # ---------------------------------------------------
        for feature_source, indices in feature_sets.items():
            x_train_sel = combined["x_train"][:, indices]
            x_test_sel = combined["x_test"][:, indices]

            row = {
                "experiment": "generalization_mnist_fashion",
                "feature_source": feature_source,
                "seed": seed,
                "k": k,
                "indices": list(map(int, indices)),
            }

            # Main downstream task: 20-class classification
            row.update(
                evaluate_logistic_classification(
                    x_train_sel=x_train_sel,
                    x_test_sel=x_test_sel,
                    y_train=combined["y_train_full"],
                    y_test=combined["y_test_full"],
                )
            )

            all_rows.append(row)

        with open(os.path.join(output_dir, "raw_generalization_results.json"), "w") as f:
            json.dump(all_rows, f, indent=2)

    df = pd.DataFrame(all_rows)

    metric_cols = [
        col for col in df.columns
        if col not in ["experiment", "feature_source", "seed", "k", "indices"]
    ]

    summary = (
        df
        .groupby(["experiment", "feature_source", "k"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    df.to_csv(os.path.join(output_dir, "raw_generalization_results.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "summary_generalization_mean_std.csv"), index=False)

    return df, summary

import numpy as np
import pandas as pd
import json

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src.data import load_mnist, load_fashion_mnist
if __name__ == "__main__":
    with open("results_generalization/raw_generalization_results.json") as f:
        results = json.load(f)

    df = pd.DataFrame(results)
    seed = 0  # doesn't matter much for evaluation

    mnist = load_mnist(seed)
    fashion = load_fashion_mnist(seed)

    X_train = np.concatenate([mnist["x_train"], fashion["x_train"]])
    X_test = np.concatenate([mnist["x_test"], fashion["x_test"]])

    y_train_domain = np.concatenate([
        np.zeros(len(mnist["x_train"])),
        np.ones(len(fashion["x_train"]))
    ])

    y_test_domain = np.concatenate([
        np.zeros(len(mnist["x_test"])),
        np.ones(len(fashion["x_test"]))
    ])

    rows = []

    for _, row in df.iterrows():
        indices = np.array(row["indices"])

        X_train_sel = X_train[:, indices]
        X_test_sel = X_test[:, indices]

        clf = LogisticRegression(max_iter=2000)

        clf.fit(X_train_sel, y_train_domain)
        pred = clf.predict(X_test_sel)

        rows.append({
            "feature_source": row["feature_source"],
            "seed": row["seed"],
            "domain_accuracy": accuracy_score(y_test_domain, pred),
            "domain_balanced_accuracy": balanced_accuracy_score(y_test_domain, pred),
            "domain_macro_f1": f1_score(y_test_domain, pred, average="macro"),
        })

    df_domain = pd.DataFrame(rows)

    summary = df_domain.groupby("feature_source").agg(["mean", "std"])
    print(summary)











    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        type=str,
        default="baseline",
        choices=["baseline", "generalization"],
        help="Which experiment to run",
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run a fast debug version (1 seed, fewer epochs)",
    )

    args = parser.parse_args()

    # ------------------------
    # Settings
    # ------------------------
    if args.fast:
        seeds = (0,)
        selector_epochs = 50
    else:
        seeds = (0, 1, 2, 3, 4)
        selector_epochs = 200

    # ------------------------
    # Run experiment
    # ------------------------
    if args.experiment == "baseline":
        df, summary = run_multiple_experiments(
            datasets=(
                "fashion_mnist",
                "mnist",
                "isolet",
                "coil20",
                "mice_protein",
                "activity",
            ),
            seeds=seeds,
            k_by_dataset={
                "coil20": 50,
                "fashion_mnist": 50,
                "mnist": 50,
                "isolet": 50,
                "mice_protein": 10,
                "activity": 50,
            },
            selector_epochs=selector_epochs,
        )

    elif args.experiment == "generalization":
        df, summary = run_multiple_experiments_generalization(
            seeds=seeds,
            k=50,
            selector_epochs=selector_epochs,
        )

    print(summary)