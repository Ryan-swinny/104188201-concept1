from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np


def evaluate_model(model, X_test, y_test, model_name):
    # 打印模型名稱
    print(f"\n{model_name} Performance:")

    # 使用模型進行預測
    y_pred = model.predict(X_test)

    # 打印分類報告，包括精確度、召回率、F1分數等
    print(classification_report(y_test, y_pred))

    # 打印混淆矩陣
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 進行5折交叉驗證
    cv_scores = cross_val_score(model, X_test, y_test, cv=5)

    # 打印交叉驗證的分數
    print(f"Cross-validation scores: {cv_scores}")

    # 打印平均交叉驗證分數
    print(f"Mean CV score: {cv_scores.mean():.4f}")

    # 返回預測結果
    return y_pred
