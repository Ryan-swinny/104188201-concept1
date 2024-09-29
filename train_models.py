import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from LSTM_extraction import LSTMFeatureExtractor, create_lstm_model
from itertools import cycle
import tensorflow as tf


def train_random_forest(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


def train_decision_tree(X, y):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    return model


def train_lstm(X, y):
    lstm_extractor = LSTMFeatureExtractor()
    lstm_features = lstm_extractor.fit_transform(X)
    additional_features = [lstm_extractor.extract_additional_features(url) for url in X]

    model = create_lstm_model(
        lstm_extractor.vocab_size,
        lstm_extractor.max_url_length,
        len(additional_features[0]),
    )

    history = model.fit(
        [lstm_features, np.array(additional_features)],
        y,
        epochs=10,
        validation_split=0.2,
        verbose=1,
    )

    return model, lstm_extractor, history


def train_model(X, y, model_type):
    if model_type == "Random Forest":
        return train_random_forest(X, y), None, None
    elif model_type == "Decision Tree":
        return train_decision_tree(X, y), None, None
    elif model_type == "LSTM":
        return train_lstm(X, y)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate_model(model, X_test, y_test, model_type):
    if model_type == "LSTM":
        lstm_features = model[1].transform(X_test)
        additional_features = [
            model[1].extract_additional_features(url) for url in X_test
        ]
        y_pred = (
            model[0].predict([lstm_features, np.array(additional_features)]).flatten()
            > 0.5
        )
    else:
        y_pred = model.predict(X_test)

    print(f"\n{model_type} Performance:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    if model_type != "LSTM":
        cv_scores = cross_val_score(model, X_test, y_test, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.4f}")

    return y_pred


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


def display_training_results(model_type, model, X, y, history=None):
    print(f"{model_type} model trained successfully.")

    if model_type == "LSTM" and history:
        # Plot training history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(history.history["accuracy"], label="Training Accuracy")
        ax1.plot(history.history["val_accuracy"], label="Validation Accuracy")
        ax1.set_title("Model Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()

        ax2.plot(history.history["loss"], label="Training Loss")
        ax2.plot(history.history["val_loss"], label="Validation Loss")
        ax2.set_title("Model Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()

        plt.show()
    elif model_type in ["Random Forest", "Decision Tree"]:
        visualize_feature_importance(model, X)
