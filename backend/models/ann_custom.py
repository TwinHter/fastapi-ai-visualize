from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from scipy.special import expit
import schemas.predictSchema as schemas
from typing import Optional

class MLP:
    def __init__(self, layer_sizes, learning_rate=0.01, n_iterations=1000, verbose=False):
        """
        layer_sizes: tuple like (n_features, hidden1, ..., hiddenN, n_outputs)
        learning_rate: float, step size for gradient descent
        n_iterations: int, number of training epochs
        verbose: bool, print progress
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.verbose = verbose

        # parameters
        self.weights = {}
        self.biases = {}

        # initialize weights and biases (Xavier uniform)
        for l in range(1, len(layer_sizes)):
            in_dim = layer_sizes[l-1]
            out_dim = layer_sizes[l]
            limit = np.sqrt(1. / in_dim)
            self.weights[l] = np.random.uniform(-limit, limit, size=(in_dim, out_dim))
            self.biases[l] = np.zeros(out_dim)

    def _sigmoid(self, x):
        return expit(x)

    def _sigmoid_derivative(self, a):
        return a * (1.0 - a)

    def _feedforward(self, X):
        activations = [X]
        L = len(self.layer_sizes) - 1
        # hidden layers with sigmoid
        for l in range(1, L):
            z = activations[l-1] @ self.weights[l] + self.biases[l]
            a = self._sigmoid(z)
            activations.append(a)
        # output layer: linear activation
        z_out = activations[L-1] @ self.weights[L] + self.biases[L]
        activations.append(z_out)
        return activations

    def _backpropagate(self, X, y):
        activations = self._feedforward(X)
        deltas = {}
        L = len(self.layer_sizes) - 1

        # output delta (linear + MSE)
        error = activations[L] - y
        deltas[L] = error

        # hidden layers
        for l in range(L-1, 0, -1):
            deltas[l] = (deltas[l+1] @ self.weights[l+1].T) * self._sigmoid_derivative(activations[l])

        # update weights & biases (full-batch GD)
        for l in range(1, L+1):
            grad_w = activations[l-1].T @ deltas[l] / X.shape[0]
            grad_b = np.mean(deltas[l], axis=0)
            self.weights[l] -= self.learning_rate * grad_w
            self.biases[l] -= self.learning_rate * grad_b

    def fit(self, X, y):
        # reshape y
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        for epoch in range(1, self.n_iterations + 1):
            self._backpropagate(X, y)
            if self.verbose and epoch % (self.n_iterations // 10) == 0:
                preds = self.predict(X)
                loss = np.mean((preds - y) ** 2)
                print(f"Epoch {epoch}/{self.n_iterations} - Loss: {loss:.6f}")

    def predict(self, X):
        return self._feedforward(X)[-1]

datacorr = None
my_ANN = None
scaler, scaler_y = None, None
X_train, X_test, y_train, y_test = None, None, None, None
encoder1, encoder2 = None, None

def load_dataset( ):
    global datacorr, X_train, X_test, y_train, y_test, encoder1, encoder2, scaler, scaler_y
    datacorr = pd.read_csv('./models/yield_df.csv')

    datacorr = datacorr[['Area', 'Item', 'Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'hg/ha_yield']]

    encoder1 = datacorr.groupby('Area')['hg/ha_yield'].mean()
    encoder2 = datacorr.groupby('Item')['hg/ha_yield'].mean()
    datacorr['Area'] = datacorr['Area'].map(encoder1)
    datacorr['Item'] = datacorr['Item'].map(encoder2)
    X = datacorr.drop('hg/ha_yield', axis=1).values
    y = datacorr['hg/ha_yield'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

def train(req: Optional[schemas.AnnParam] = None):
    if req is None:
        req = schemas.AnnParam()
    layer_sizes = req.layer_sizes if req.layer_sizes is not None else [6, 32, 16, 1]
    learning_rate = req.learning_rate if req.learning_rate is not None else 0.1
    n_iterations = req.max_iter if req.max_iter is not None else 1000
    verbose = req.verbose if req.verbose is not None else True
    
    global my_ANN
    load_dataset()
    layer_sizes = np.concatenate(([6], layer_sizes, [1]), axis=0)
    my_ANN = MLP(layer_sizes, learning_rate, n_iterations, verbose)
    my_ANN.fit(X_train, y_train)
    return 0


def predict(req: schemas.PredictRequest):
    req.Area = pd.Series([req.Area]) # Convert Area and Item to pandas Series objects
    req.Item = pd.Series([req.Item])
    req.Area = req.Area.map(encoder1)[0]
    req.Item = req.Item.map(encoder2)[0]
    input_features = [req.Area, req.Item, req.Year, req.average_rain_fall_mm_per_year, req.pesticides_tonnes, req.avg_temp]
    input_features = scaler.transform([input_features])
    return scaler_y.inverse_transform(my_ANN.predict(input_features))[0][0]

def get_metrics_train():
    if my_ANN is None:
        return {"MSE": -1, "R2": -1}
    y_pred = scaler_y.inverse_transform(my_ANN.predict(X_train))
    y_trn = scaler_y.inverse_transform(y_train.reshape(-1, 1))
    mse = round(mean_squared_error(y_trn, y_pred), 4)
    r2 = round(r2_score(y_trn, y_pred), 4)
    return {"MSE": mse, "R2": r2}

def get_metrics_test():
    if my_ANN is None:
        return {"MSE": -1, "R2": -1}
    y_pred = scaler_y.inverse_transform(my_ANN.predict(X_test))
    y_tst = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    mse = round(mean_squared_error(y_tst, y_pred), 4)
    r2 = round(r2_score(y_tst, y_pred), 4)
    return {"MSE": mse, "R2": r2}