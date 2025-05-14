import schemas.predictSchema as schemas

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from typing import Optional

class Node:
    def __init__(self, feature_index, threshold, left, right, value):
        # Index of the feature used for the split at this node
        self.feature_index = feature_index
        # Threshold value for the split
        self.threshold = threshold
        # Left child node
        self.left = left
        # Right child node
        self.right = right
        # Prediction value at this node (used for leaf nodes)
        self.value = value

class Decision_tree:
    def __init__(self, min_samples_split=2, max_depth=2, min_samples_leaf=1, random_state=0):
        # Initialize the decision tree with hyperparameters
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.rng = np.random.RandomState(random_state)

    def _mse(self, y):
        # Calculate Mean Squared Error for a set of targets
        return np.mean((y - np.mean(y)) ** 2)

    def _split(self, X, y, feature_index, threshold):
        # Split the dataset based on a feature and threshold
        X_left = X[X[:, feature_index] <= threshold]
        y_left = y[X[:, feature_index] <= threshold]
        X_right = X[X[:, feature_index] > threshold]
        y_right = y[X[:, feature_index] > threshold]
        return X_left, X_right, y_left, y_right

    def _information_gain(self, parent, l_child, r_child):
        # Calculate the reduction in MSE after a split (i.e., information gain)
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        gain = self._mse(parent) - (weight_l * self._mse(l_child) + weight_r * self._mse(r_child))
        return gain

    def _best_split(self, X, y):
        # Find the best feature and threshold to split the data
        n, m = X.shape
        if n < self.min_samples_split:
            return None
        list_best_split = []
        max_info = -float("inf")

        for idx in range(m):
            thresholds = np.unique(X[:, idx])
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self._split(X, y, idx, threshold)
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue
                info_gain = self._information_gain(y, y_left, y_right)
                if info_gain > max_info:
                    max_info = info_gain
                    current_split = {
                        "feature_index": idx,
                        "threshold": threshold,
                        "X_left": X_left,
                        "X_right": X_right,
                        "y_left": y_left,
                        "y_right": y_right
                    }
                    list_best_split = [current_split]
                elif info_gain == max_info:
                    list_best_split.append(current_split)

        # Randomly select one of the best splits (if multiple)
        best_split = None
        if len(list_best_split) > 0:
            best_split = self.rng.choice(list_best_split)

        return best_split

    def _build_tree(self, X, y, depth=0):
        # Recursively build the decision tree
        if depth >= self.max_depth:
            return Node(None, None, None, None, np.mean(y))
        best_split = self._best_split(X, y)
        if best_split is None:
            return Node(None, None, None, None, np.mean(y))

        node = Node(best_split["feature_index"], best_split["threshold"], None, None, np.mean(y))
        node.left = self._build_tree(best_split["X_left"], best_split["y_left"], depth + 1)
        node.right = self._build_tree(best_split["X_right"], best_split["y_right"], depth + 1)
        return node

    def fit(self, X, y):
        # Fit the tree to the training data
        self.root = self._build_tree(X, y)
        return self

    def _predict_one(self, node, X):
        # Predict a single sample by traversing the tree
        if node.left is None and node.right is None:
            return node.value
        return (self._predict_one(node.left, X) if X[node.feature_index] <= node.threshold else self._predict_one(node.right, X))

    def predict(self, X):
        # Predict for multiple samples
        return np.array([self._predict_one(self.root, x) for x in X])

datacorr = None
my_decision_tree = None
X_train, X_test, y_train, y_test = None, None, None, None
label_encoder1, label_encoder2 = None, None

encoder1, encoder2 = None, None

def load_dataset( ):
    global datacorr, X_train, X_test, y_train, y_test, encoder1, encoder2
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


def train(req: Optional[schemas.DecisionTreeParam] = None):
    if req is None:
        req = schemas.DecisionTreeParam()
    min_samples_split = req.min_samples_split if req.min_samples_split is not None else 2
    max_depth = req.max_depth if req.max_depth is not None else 50
    min_samples_leaf = req.min_samples_leaf if req.min_samples_leaf is not None else 1
    random_state = req.random_state if req.random_state is not None else 42
    
    global my_decision_tree
    load_dataset()
    my_decision_tree = Decision_tree(min_samples_split=min_samples_split, max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state)
    my_decision_tree.fit(X_train, y_train)
    return 0


def predict(req: schemas.PredictRequest):
    if my_decision_tree is None:
        return -1
    
    req.Area = pd.Series([req.Area]) # Convert Area and Item to pandas Series objects
    req.Item = pd.Series([req.Item])
    req.Area = req.Area.map(encoder1)[0]
    req.Item = req.Item.map(encoder2)[0]
    input_features = [req.Area, req.Item, req.Year, req.average_rain_fall_mm_per_year, req.pesticides_tonnes, req.avg_temp]
    return my_decision_tree.predict([input_features])[0]

def get_metrics_train():
    if my_decision_tree is None:
        return {"MSE": -1, "R2": -1}
    y_pred = my_decision_tree.predict(X_train)
    mse = round(mean_squared_error(y_train, y_pred), 4)
    r2 = round(r2_score(y_train, y_pred), 4)
    return {"MSE": mse, "R2": r2}

def get_metrics_test():
    if my_decision_tree is None:
        return {"MSE": -1, "R2": -1}
    y_pred = my_decision_tree.predict(X_test)
    mse = round(mean_squared_error(y_test, y_pred), 4)
    r2 = round(r2_score(y_test, y_pred), 4)
    return {"MSE": mse, "R2": r2}