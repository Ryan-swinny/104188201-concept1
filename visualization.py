import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import numpy as np


def visualize_feature_importance(model, X):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_imp = pd.DataFrame(
            sorted(zip(importances, X.columns), reverse=True),
            columns=["Importance", "Feature"],
        )
        print("\nTop 10 important features:")
        print(feature_imp.head(10))

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=feature_imp.head(10))
        plt.title("Top 10 Important Features")
        plt.tight_layout()
        plt.show()
    else:
        print("This model doesn't support feature importance visualization.")


def visualize_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()


def visualize_roc_curve(model, X_test, y_test, model_name):
    n_classes = len(np.unique(y_test))
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    else:
        print(f"ROC curve not available for {model_name}")
        return

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    colors = cycle(["blue", "red", "green"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"ROC curve of class {i} (area = {roc_auc[i]:0.2f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.show()
