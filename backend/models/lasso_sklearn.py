from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, r2_score
import schemas.predictSchema as schemas
from typing import Optional

datacorr = None
scaler = None
Lasso_sklearn = None
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
    alpha = req.alpha if req.alpha is not None else 1
    max_iter = req.max_iter if req.max_iter is not None else 1000
    fit_intercept = req.fit_intercept if req.fit_intercept is not None else True
    global Lasso_sklearn
    load_dataset()
    Lasso_sklearn = Lasso(alpha=alpha, max_iter=max_iter, fit_intercept=fit_intercept)
    Lasso_sklearn.fit(X_train, y_train)
    return 0


def predict(req: schemas.PredictRequest):
    req.Area = pd.Series([req.Area]) # Convert Area and Item to pandas Series objects
    req.Item = pd.Series([req.Item])
    req.Area = req.Area.map(encoder1)[0]
    req.Item = req.Item.map(encoder2)[0]
    input_features = [req.Area, req.Item, req.Year, req.average_rain_fall_mm_per_year, req.pesticides_tonnes, req.avg_temp]
    input_features = scaler.transform([input_features])
    return Lasso_sklearn.predict(input_features)[0]

def get_metrics_train():
    if Lasso_sklearn is None:
        return {"MSE": -1, "R2": -1}
    y_pred = Lasso_sklearn.predict(X_train)
    mse = round(mean_squared_error(y_train, y_pred), 4)
    r2 = round(r2_score(y_train, y_pred), 4)
    return {"MSE": mse, "R2": r2}

def get_metrics_test():
    if Lasso_sklearn is None:
        return {"MSE": -1, "R2": -1}
    y_pred = Lasso_sklearn.predict(X_test)
    mse = round(mean_squared_error(y_test, y_pred), 4)
    r2 = round(r2_score(y_test, y_pred), 4)
    return {"MSE": mse, "R2": r2}