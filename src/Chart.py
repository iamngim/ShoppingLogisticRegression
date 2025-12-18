import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import roc_curve

# --------- STYLE CHUNG CHO BIỂU ĐỒ ----------
def _apply_common_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10
    })


def plot_roc(fpr, tpr, auc, save_path):
    """
    Vẽ và lưu ROC Curve
    """
    _apply_common_style()
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, linewidth=2.5, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Logistic Regression")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="lower right")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_label_distribution(df, save_path, title="Label Distribution"):
    """
    Vẽ và lưu phân bố nhãn (label)
    """
    _apply_common_style()
    pdf = df.groupBy("label").count().toPandas().sort_values("label")

    plt.figure(figsize=(6.5, 4.8))
    plt.bar(pdf["label"].astype(str), pdf["count"], alpha=0.9, edgecolor="white")
    plt.xlabel("Label (0 / 1)")
    plt.ylabel("Số lượng")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_feature_importance(weights, feature_cols, save_path):
    """
    Vẽ và lưu biểu đồ trọng số đặc trưng (coefficients)
    """
    _apply_common_style()
    weights = np.array(weights)

    # Sắp xếp theo độ lớn tuyệt đối để dễ nhìn
    order = np.argsort(np.abs(weights))
    sorted_features = np.array(feature_cols)[order]
    sorted_weights = weights[order]

    plt.figure(figsize=(9.5, 6.5))
    plt.barh(sorted_features, sorted_weights, alpha=0.9)
    plt.xlabel("Coefficient Weight")
    plt.title("Feature Importance - Logistic Regression")
    plt.grid(axis="x", linestyle="--", alpha=0.35)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def extract_roc_points(predictions):
    """
    Trích xuất FPR, TPR từ Spark DataFrame predictions
    """
    pdf = predictions.select("label", "probability").toPandas()
    y_true = pdf["label"].values
    y_score = np.array([p[1] for p in pdf["probability"]])

    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr


def plot_feature_histogram(df, feature, save_path):
    """
    Histogram đẹp hơn: có màu, viền, grid nhẹ
    """
    _apply_common_style()
    pdf = df.select(feature).dropna().toPandas()

    plt.figure(figsize=(6.5, 4.8))
    plt.hist(pdf[feature], bins=30, alpha=0.88, edgecolor="white")
    plt.xlabel(feature)
    plt.ylabel("Tần suất")
    plt.title(f"Phân phối {feature}")
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_monetary_boxplot(df, save_path):
    """
    Boxplot đẹp hơn: patch_artist + grid
    """
    _apply_common_style()
    pdf = df.select("Monetary", "label").dropna().toPandas()

    data0 = pdf[pdf["label"] == 0]["Monetary"]
    data1 = pdf[pdf["label"] == 1]["Monetary"]

    plt.figure(figsize=(6.5, 4.8))
    plt.boxplot(
        [data0, data1],
        labels=["Không mua", "Mua"],
        patch_artist=True,
        boxprops=dict(alpha=0.75),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(alpha=0.8),
        capprops=dict(alpha=0.8)
    )
    plt.ylabel("Monetary")
    plt.title("So sánh giá trị chi tiêu (Monetary) theo nhãn")
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
