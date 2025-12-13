import matplotlib.pyplot as plt
import numpy as np
import os


def plot_roc(fpr, tpr, auc, save_path):
    """
    Vẽ và lưu ROC Curve
    """
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Logistic Regression")
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_label_distribution(df, save_path):
    """
    Vẽ và lưu phân bố nhãn (label)
    """
    df_pd = df.select("label").toPandas()

    ax = df_pd["label"].value_counts().sort_index().plot(kind="bar")
    ax.set_title("Label Distribution")
    ax.set_xlabel("Label (0 / 1)")
    ax.set_ylabel("Count")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(weights, feature_cols, save_path):
    """
    Vẽ và lưu biểu đồ trọng số đặc trưng (coefficients)
    """
    weights = np.array(weights)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_cols, weights)
    plt.xlabel("Coefficient Weight")
    plt.title("Feature Importance - Logistic Regression")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def extract_roc_points(predictions):
    """
    Trích xuất FPR, TPR từ Spark DataFrame predictions
    """
    pdf = predictions.select("label", "probability").toPandas()

    y_true = pdf["label"].values
    y_score = np.array([p[1] for p in pdf["probability"]])

    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)

    return fpr, tpr
