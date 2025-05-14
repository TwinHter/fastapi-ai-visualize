from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import schemas.predictSchema as schemas
from typing import Optional

datacorr = None
Decision_tree_sklearn = None
X_train, X_test, y_train, y_test = None, None, None, None
label_encoder1, label_encoder2 = None, None

def load_dataset():
    global datacorr, X_train, X_test, y_train, y_test, label_encoder1, label_encoder2
    datacorr = pd.read_csv('./models/yield_df.csv')

    datacorr = datacorr[['Area', 'Item', 'Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'hg/ha_yield']]

    label_encoder1 = LabelEncoder()
    datacorr['Area'] = label_encoder1.fit_transform(datacorr['Area'])
    label_encoder2 = LabelEncoder()
    datacorr['Item'] = label_encoder2.fit_transform(datacorr['Item'])
    X = datacorr.drop('hg/ha_yield', axis=1).values
    y = datacorr['hg/ha_yield'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

def train(req: Optional[schemas.DecisionTreeParam] = None):
    if req is None:
        req = schemas.DecisionTreeParam()
    min_samples_split = req.min_samples_split if req.min_samples_split is not None else 2
    max_depth = req.max_depth if req.max_depth is not None else 50
    min_samples_leaf = req.min_samples_leaf if req.min_samples_leaf is not None else 1
    random_state = req.random_state if req.random_state is not None else 42
    global Decision_tree_sklearn
    load_dataset()
    Decision_tree_sklearn = DecisionTreeRegressor(min_samples_split=min_samples_split, max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state)
    Decision_tree_sklearn.fit(X_train, y_train)
    return 0


def predict(req: schemas.PredictRequest):
    req.Area = label_encoder1.transform([req.Area])[0]
    req.Item = label_encoder2.transform([req.Item])[0]
    input_features = [req.Area, req.Item, req.Year, req.average_rain_fall_mm_per_year, req.pesticides_tonnes, req.avg_temp]
    return Decision_tree_sklearn.predict([input_features])[0]

def get_metrics_train():
    if Decision_tree_sklearn is None:
        return {"MSE": -1, "R2": -1}
    y_pred = Decision_tree_sklearn.predict(X_train)
    mse = round(mean_squared_error(y_train, y_pred), 4)
    r2 = round(r2_score(y_train, y_pred), 4)
    return {"MSE": mse, "R2": r2}

def get_metrics_test():
    if Decision_tree_sklearn is None:
        return {"MSE": -1, "R2": -1}
    y_pred = Decision_tree_sklearn.predict(X_test)
    mse = round(mean_squared_error(y_test, y_pred), 4)
    r2 = round(r2_score(y_test, y_pred), 4)
    return {"MSE": mse, "R2": r2}