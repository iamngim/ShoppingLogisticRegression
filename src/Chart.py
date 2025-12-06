import matplotlib.pyplot as plt
import numpy as np

def plot_roc(fpr, tpr, auc):
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Logistic Regression")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_label_distribution(df):
    df_pd = df.select("label").toPandas()
    df_pd["label"].value_counts().plot(kind="bar")
    plt.title("Label Distribution")
    plt.xlabel("Label (0/1)")
    plt.ylabel("Count")
    plt.show()


def plot_feature_importance(weights, feature_cols):
    plt.figure(figsize=(10, 6))
    plt.barh(feature_cols, weights)
    plt.xlabel("Coefficient Weight")
    plt.title("Feature Coefficients - Logistic Regression")
    plt.show()



def extract_roc_points(predictions):
    # Spark trả về probability dạng [p0, p1]
    pdf = predictions.select("label", "probability").toPandas()

    y_true = pdf["label"].values
    y_score = np.array([x[1] for x in pdf["probability"]])

    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)

    return fpr, tpr
