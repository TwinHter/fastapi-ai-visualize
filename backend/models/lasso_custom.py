from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import schemas.predictSchema as schemas
from typing import Optional

class LassoRegression:
    def __init__(self, alpha=10, learning_rate=0.0000001, n_iterations=1000000, fit_intercept = True):
        self.alpha = alpha  # Regularization parameter (L1 penalty)
        self.learning_rate = learning_rate  # Learning rate for gradient descent
        self.n_iterations = n_iterations  # Number of gradient descent iterations
        self.beta = None  # Regression coefficients (to be learned)
        self.bias = None  # Intercept term
        self.fit_intercept = fit_intercept  # Whether to include an intercept term in the model

    # Compute the Lasso loss function (MSE + L1 penalty)
    def lasso_loss(self, X, y, beta):
        n = len(y)
        # Compute Residual Sum of Squares (RSS)
        residual_sum_of_squares = np.sum((y - (self.bias if self.fit_intercept else 0.0) - X.dot(beta))**2)
        # Compute L1 regularization term
        l1_penalty = self.alpha * np.sum(np.abs(beta))
        return (1/n) * residual_sum_of_squares + l1_penalty

    # Compute the gradient of the loss function
    def gradient(self, X, y, beta):
        n = len(y)
        # Gradient of the RSS term
        gradient_rss = -2 * X.T.dot(y - self.bias - X.dot(beta)) / n
        # Gradient with respect to bias
        gradient_bias = -2 * np.sum(y - self.bias - X.dot(beta)) / n
        # Gradient of the L1 penalty (subgradient via sign)
        gradient_l1 = self.alpha * np.sign(beta)
        # Return combined gradient for beta and bias
        return gradient_rss + gradient_l1, gradient_bias

    # Train the Lasso model using gradient descent
    def fit(self, X, y):
        n_features = X.shape[1]
        # Initialize coefficients and bias to zero
        self.beta = np.zeros(n_features)
        self.bias = 0.0
        loss_history = []

        # Gradient descent loop
        for iteration in range(self.n_iterations):
            grad_beta, grad_bias = self.gradient(X, y, self.beta)
            # Update coefficients and bias
            self.beta -= self.learning_rate * grad_beta
            self.bias -= self.learning_rate * grad_bias
            # Record loss for visualization
            loss = self.lasso_loss(X, y, self.beta)
            loss_history.append(loss)


    # Predict using the trained Lasso model
    def predict(self, X):
        return X.dot(self.beta) + (self.bias if self.fit_intercept else 0.0)

datacorr = None
my_lasso = None
scaler = None
X_train, X_test, y_train, y_test = None, None, None, None
encoder1, encoder2 = None, None

def load_dataset( ):
    global datacorr, X_train, X_test, y_train, y_test, encoder1, encoder2, scaler
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

def train(req: Optional[schemas.LassoParam] = None):
    if req is None:
        req = schemas.LassoParam()
    alpha = req.alpha if req.alpha is not None else 10
    n_iterations = req.max_iter if req.max_iter is not None else 1000
    fit_intercept = req.fit_intercept if req.fit_intercept is not None else True
    global my_lasso
    load_dataset()
    my_lasso = LassoRegression(alpha=alpha, learning_rate=0.01, n_iterations=n_iterations, fit_intercept=fit_intercept)
    my_lasso.fit(X_train, y_train)
    return 0


def predict(req: schemas.PredictRequest):
    req.Area = pd.Series([req.Area]) # Convert Area and Item to pandas Series objects
    req.Item = pd.Series([req.Item])
    req.Area = req.Area.map(encoder1)[0]
    req.Item = req.Item.map(encoder2)[0]
    input_features = [req.Area, req.Item, req.Year, req.average_rain_fall_mm_per_year, req.pesticides_tonnes, req.avg_temp]
    input_features = scaler.transform([input_features])
    return my_lasso.predict(input_features)[0]

def get_metrics_train():
    if my_lasso is None:
        return {"MSE": -1, "R2": -1}
    y_pred = my_lasso.predict(X_train)
    mse = round(mean_squared_error(y_train, y_pred), 4)
    r2 = round(r2_score(y_train, y_pred), 4)
    return {"MSE": mse, "R2": r2}

def get_metrics_test():
    if my_lasso is None:
        return {"MSE": -1, "R2": -1}
    y_pred = my_lasso.predict(X_test)
    mse = round(mean_squared_error(y_test, y_pred), 4)
    r2 = round(r2_score(y_test, y_pred), 4)
    return {"MSE": mse, "R2": r2}