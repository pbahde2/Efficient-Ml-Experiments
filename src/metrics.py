from sklearn.linear_model import LinearRegression

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_squared_error,

)


def evaluate_logistic_classification(x_train_sel, x_test_sel, y_train, y_test):
    clf = LogisticRegression(
        max_iter=2000,
        C=np.inf,
        solver="lbfgs",
        n_jobs=-1,
)

    clf.fit(x_train_sel, y_train)
    pred = clf.predict(x_test_sel)

    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
    }



def evaluate_linear_reconstruction(x_train_sel, x_test_sel, x_train_full, x_test_full):
    model = LinearRegression()
    model.fit(x_train_sel, x_train_full)
    pred = model.predict(x_test_sel)

    return {
        "reconstruction_mse": float(mean_squared_error(x_test_full, pred)),
    }
