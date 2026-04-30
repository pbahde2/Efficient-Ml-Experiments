import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_openml


def make_split(x, y, seed=0, test_size=0.2, val_size=0.1, scale=True):
    x = x.astype("float32")
    y = np.asarray(y).astype("int64")

    if scale:
        x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    relative_val_size = val_size / (1.0 - test_size)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=relative_val_size,
        random_state=seed,
        stratify=y_train_val,
    )

    num_classes = len(np.unique(y))

    return {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "y_train_oh": to_categorical(y_train, num_classes),
        "y_val_oh": to_categorical(y_val, num_classes),
        "y_test_oh": to_categorical(y_test, num_classes),
        "input_dim": x_train.shape[1],
        "num_classes": num_classes,
    }


def load_mnist(seed=0):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x = np.concatenate([x_train, x_test], axis=0).astype("float32") / 255.0
    y = np.concatenate([y_train, y_test], axis=0)

    x = x.reshape(len(x), -1)

    return make_split(x, y, seed=seed, scale=False)


def load_fashion_mnist(seed=0):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    x = np.concatenate([x_train, x_test], axis=0).astype("float32") / 255.0
    y = np.concatenate([y_train, y_test], axis=0)

    x = x.reshape(len(x), -1)

    return make_split(x, y, seed=seed, scale=False)


def load_isolet(seed=0):
    dataset = fetch_openml("isolet", version=1, as_frame=False)

    x = dataset.data.astype("float32")
    y = dataset.target.astype("int64") - 1

    return make_split(x, y, seed=seed, scale=True)


def load_coil20(seed=0):
    dataset = fetch_openml(data_id=40982, as_frame=False)

    x = dataset.data.astype("float32")

    # FIX: encode string labels to integers
    _, y = np.unique(dataset.target, return_inverse=True)

    if x.max() > 1.0:
        x = x / 255.0

    return make_split(x, y, seed=seed, scale=False)


def load_mice_protein(seed=0):
    dataset = fetch_openml("miceprotein", version=4, as_frame=True)

    df = dataset.frame.copy()

    # Remove columns that are not numeric features
    target = dataset.target

    x = df.drop(columns=[target.name], errors="ignore")
    x = x.select_dtypes(include=[np.number])
    x = x.fillna(x.mean())

    y_raw = target.astype("category")
    y = y_raw.cat.codes.values

    return make_split(x.values, y, seed=seed, scale=True)


def load_activity(seed=0):
    dataset = fetch_openml("har", version=1, as_frame=False)

    x = dataset.data.astype("float32")
    y_raw = dataset.target

    _, y = np.unique(y_raw, return_inverse=True)

    return make_split(x, y, seed=seed, scale=True)


def load_dataset(name, seed=0):
    name = name.lower()

    if name == "mnist":
        return load_mnist(seed)

    if name in ["fashion_mnist", "fashion-mnist", "fashion"]:
        return load_fashion_mnist(seed)

    if name == "isolet":
        return load_isolet(seed)

    if name in ["coil20", "coil-20", "coil"]:
        return load_coil20(seed)

    if name in ["mice", "mice_protein", "mice-protein"]:
        return load_mice_protein(seed)

    if name in ["activity", "har", "human_activity"]:
        return load_activity(seed)

    raise ValueError(f"Unknown dataset: {name}")


def load_combined_mnist_fashion(seed=0):
    mnist = load_mnist(seed)
    fashion = load_fashion_mnist(seed)

    X = np.concatenate([mnist["x_train"], fashion["x_train"]])
    
    y_domain = np.concatenate([
        np.zeros(len(mnist["x_train"])),
        np.ones(len(fashion["x_train"]))
    ])

    y_full = np.concatenate([
        mnist["y_train"],
        fashion["y_train"] + 10
    ])

    return X, y_domain, y_full