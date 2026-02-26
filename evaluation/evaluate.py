import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_auc_score, mean_squared_error, r2_score
)
import os

os.makedirs('evaluation/plots', exist_ok=True)

def evaluate_classifier(name, model, X_test, y_test, label_names=None):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"Accuracy: {acc:.4f}")
    if auc: print(f"AUC-ROC:  {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_names))

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names or ['0','1'],
                yticklabels=label_names or ['0','1'], ax=ax)
    ax.set_title(f'{name} - Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'evaluation/plots/{name.replace(" ","_")}_cm.png', dpi=120)
    plt.close()

    return {'accuracy': acc, 'auc': auc, 'report': report, 'cm': cm}

def evaluate_regressor(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"R²:    {r2:.4f}")

    # Actual vs Predicted
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, y_pred, alpha=0.5, color='steelblue', edgecolors='navy', linewidths=0.3)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Marks')
    ax.set_ylabel('Predicted Marks')
    ax.set_title(f'{name} - Actual vs Predicted', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'evaluation/plots/{name.replace(" ","_")}_regplot.png', dpi=120)
    plt.close()

    return {'rmse': rmse, 'r2': r2}

def plot_feature_importance(name, model, feature_names):
    if not hasattr(model, 'feature_importances_'):
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(feature_names)))
    ax.bar(range(len(feature_names)), importances[indices], color=colors)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=35, ha='right')
    ax.set_title(f'{name} - Feature Importance', fontsize=14, fontweight='bold')
    ax.set_ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig(f'evaluation/plots/{name.replace(" ","_")}_importance.png', dpi=120)
    plt.close()
