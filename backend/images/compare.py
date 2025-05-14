import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import models.decision_tree_custom as decision_tree_custom
import models.decision_tree_sklearn as decision_tree_sklearn
import models.lasso_custom as lasso_custom
import models.lasso_sklearn as lasso_sklearn
import models.ann_custom as ann_custom
import models.ann_sklearn as ann_sklearn

def compare_models_metrics_separate(type_metric: str, type_plot: str):
    """
    type_metric: "MSE" or "R2" — chooses which of the two charts to return.
    type_plot: one of ["bar", "line", "scatter", "radar"]
    
    Returns a single io.BytesIO PNG buffer for the requested metric.
    """
    # 1) Gather metrics into a DataFrame
    records = []
    for name, get_train, get_test in [
        ('Decision Tree Custom', decision_tree_custom.get_metrics_train, decision_tree_custom.get_metrics_test),
        ('Decision Tree Sklearn', decision_tree_sklearn.get_metrics_train, decision_tree_sklearn.get_metrics_test),
        ('Lasso Custom', lasso_custom.get_metrics_train, lasso_custom.get_metrics_test),
        ('Lasso Sklearn', lasso_sklearn.get_metrics_train, lasso_sklearn.get_metrics_test),
        ('ANN Custom', ann_custom.get_metrics_train, ann_custom.get_metrics_test),
        ('ANN Sklearn', ann_sklearn.get_metrics_train, ann_sklearn.get_metrics_test),
    ]:
        mt = get_train()
        ms = get_test()
        records.append({
            'Model': name,
            'Train MSE': mt['MSE'], 'Test MSE': ms['MSE'],
            'Train R2':  mt['R2'],  'Test R2':  ms['R2'],
        })
    df = pd.DataFrame(records).set_index('Model')
    
    models = list(df.index)
    x = np.arange(len(models))
    width = 0.3

    # Prepare placeholders
    img_buf1 = img_buf2 = None

    # —— BAR ——
    if type_plot == "bar":
        # MSE
        fig, ax = plt.subplots(figsize=(12,6))
        ax.bar(x - width/2, df['Train MSE'], width, label='Train MSE')
        ax.bar(x + width/2, df['Test MSE'],  width, label='Test MSE')
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=45)
        ax.set_ylabel('MSE'); ax.set_title('Train vs Test MSE')
        ax.legend(); plt.tight_layout()
        img_buf1 = io.BytesIO(); plt.savefig(img_buf1, format='png'); plt.close(fig); img_buf1.seek(0)

        # R2
        fig, ax = plt.subplots(figsize=(12,6))
        ax.bar(x - width/2, df['Train R2'], width, label='Train R2')
        ax.bar(x + width/2, df['Test R2'],  width, label='Test R2')
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=45)
        ax.set_ylabel('R2 Score'); ax.set_title('Train vs Test R2')
        ax.legend(); plt.tight_layout()
        img_buf2 = io.BytesIO(); plt.savefig(img_buf2, format='png'); plt.close(fig); img_buf2.seek(0)

    # —— LINE ——
    elif type_plot == "line":
        # MSE
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(models, df['Train MSE'], marker='o', label='Train MSE')
        ax.plot(models, df['Test MSE'],  marker='s', label='Test MSE')
        ax.set_ylabel('MSE'); ax.set_title('Line Plot: Train vs Test MSE')
        ax.legend(); plt.xticks(rotation=45); plt.tight_layout()
        img_buf1 = io.BytesIO(); plt.savefig(img_buf1, format='png'); plt.close(fig); img_buf1.seek(0)

        # R2
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(models, df['Train R2'], marker='^', label='Train R2')
        ax.plot(models, df['Test R2'],  marker='v', label='Test R2')
        ax.set_ylabel('R2 Score'); ax.set_title('Line Plot: Train vs Test R2')
        ax.legend(); plt.xticks(rotation=45); plt.tight_layout()
        img_buf2 = io.BytesIO(); plt.savefig(img_buf2, format='png'); plt.close(fig); img_buf2.seek(0)

    # —— SCATTER ——
    elif type_plot == "scatter":
        # MSE
        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(df['Train MSE'], df['Test MSE'])
        for i, m in enumerate(models):
            ax.annotate(m, (df['Train MSE'][i], df['Test MSE'][i]))
        ax.set_xlabel('Train MSE'); ax.set_ylabel('Test MSE')
        ax.set_title('Scatter: Train vs Test MSE'); plt.tight_layout()
        img_buf1 = io.BytesIO(); plt.savefig(img_buf1, format='png'); plt.close(fig); img_buf1.seek(0)

        # R2
        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(df['Train R2'], df['Test R2'])
        for i, m in enumerate(models):
            ax.annotate(m, (df['Train R2'][i], df['Test R2'][i]))
        ax.set_xlabel('Train R2'); ax.set_ylabel('Test R2')
        ax.set_title('Scatter: Train vs Test R2'); plt.tight_layout()
        img_buf2 = io.BytesIO(); plt.savefig(img_buf2, format='png'); plt.close(fig); img_buf2.seek(0)

    # —— RADAR ——
    elif type_plot == "radar":
        def _radar(metrics, title):
            labels = ['Train', 'Test']
            angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
            angles += angles[:1]
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(111, polar=True)
            for model in models:
                vals = df.loc[model, metrics].tolist()
                vals += vals[:1]
                ax.plot(angles, vals, label=model)
                ax.fill(angles, vals, alpha=0.1)
            ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
            ax.set_title(title)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            plt.tight_layout()
            buf = io.BytesIO(); plt.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
            return buf

        img_buf1 = _radar(['Train MSE','Test MSE'], 'Radar: MSE')
        img_buf2 = _radar(['Train R2','Test R2'],   'Radar: R2')

    else:
        raise ValueError(f"Unknown plot type: {type_plot}")

    # Finally return the requested metric buffer
    return img_buf1 if type_metric == 'mse' else img_buf2