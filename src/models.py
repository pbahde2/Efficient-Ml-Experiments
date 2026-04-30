from tensorflow.keras.layers import Dense


def make_reconstruction_head(input_dim):
    """
    Linear regressor: y = Wx + b
    Equivalent to Ridge (without regularization) inside CAE
    """
    def decoder(x):
        return Dense(input_dim, activation=None)(x)
    return decoder


def make_classification_head(num_classes):
    """
    Multinomial logistic regression
    """
    def classifier(x):
        return Dense(num_classes, activation="softmax")(x)
    return classifier