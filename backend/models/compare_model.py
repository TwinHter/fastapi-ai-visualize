from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import schemas.predictSchema as schemas
import models.decision_tree_custom as decision_tree_custom
import models.decision_tree_sklearn as decision_tree_sklearn
import models.lasso_custom as lasso_custom
import models.lasso_sklearn as lasso_sklearn
import models.ann_sklearn as ann_sklearn
import models.ann_custom as ann_custom


def compare_models_metrics(models, X_train, y_train, X_test, y_test):
    """
    Compare multiple models based on MSE and R2 on training and testing sets.
    Any model that fails to fit or predict will be skipped.
    Returns a DataFrame of metrics and shows bar plots for MSE and R2.
    """
    # Store results
    records = []

    metrics_train = decision_tree_custom.get_metrics_train()
    metrics_test = decision_tree_custom.get_metrics_test()
    records.append({
        'Model': 'Decision Tree Custom',
        'Train MSE': metrics_train['MSE'],
        'Test MSE': metrics_test['MSE'],
        'Train R2': metrics_train['R2'],
        'Test R2': metrics_test['R2'],
    })
    
    metrics_train = decision_tree_sklearn.get_metrics_train()
    metrics_test = decision_tree_sklearn.get_metrics_test()
    records.append({
        'Model': 'Decision Tree Sklearn',
        'Train MSE': metrics_train['MSE'],
        'Test MSE': metrics_test['MSE'],
        'Train R2': metrics_train['R2'],
        'Test R2': metrics_test['R2'],
    })
    
    metrics_train = lasso_custom.get_metrics_train()
    metrics_test = lasso_custom.get_metrics_test()
    records.append({
        'Model': 'Lasso Custom',
        'Train MSE': metrics_train['MSE'],
        'Test MSE': metrics_test['MSE'],
        'Train R2': metrics_train['R2'],
        'Test R2': metrics_test['R2'],
    })
    
    metrics_train = lasso_sklearn.get_metrics_train()
    metrics_test = lasso_sklearn.get_metrics_test()
    records.append({
        'Model': 'Lasso Sklearn',
        'Train MSE': metrics_train['MSE'],
        'Test MSE': metrics_test['MSE'],
        'Train R2': metrics_train['R2'],
        'Test R2': metrics_test['R2'],
    })
    
    metrics_train = ann_custom.get_metrics_train()
    metrics_test = ann_custom.get_metrics_test()
    records.append({
        'Model': 'ANN Custom',
        'Train MSE': metrics_train['MSE'],
        'Test MSE': metrics_test['MSE'],
        'Train R2': metrics_train['R2'],
        'Test R2': metrics_test['R2'],
    })
    
    metrics_train = ann_sklearn.get_metrics_train()
    metrics_test = ann_sklearn.get_metrics_test()
    records.append({
        'Model': 'ANN Sklearn',
        'Train MSE': metrics_train['MSE'],
        'Test MSE': metrics_test['MSE'],
        'Train R2': metrics_train['R2'],
        'Test R2': metrics_test['R2'],
    })

    # Build DataFrame
    df = pd.DataFrame(records).set_index('Model')
    print("\nMetrics Summary:")
    print(df)

    # Plotting
    x = np.arange(len(df.index))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5*width, df['Train MSE'], width, label='Train MSE')
    ax.bar(x - 0.5*width, df['Test MSE'],  width, label='Test MSE')
    ax.bar(x + 0.5*width, df['Train R2'],  width, label='Train R2')
    ax.bar(x + 1.5*width, df['Test R2'],   width, label='Test R2')

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=45)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison: MSE and R2 on Train vs Test')
    ax.legend()
    plt.tight_layout()
    plt.show()

    return df
