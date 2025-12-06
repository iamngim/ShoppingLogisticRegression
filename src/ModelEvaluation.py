import pandas as pd
import numpy as np
import os
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


def evaluate_model(pred):
    # Basic metrics
    evaluator_auc = BinaryClassificationEvaluator(labelCol="label")
    auc = evaluator_auc.evaluate(pred)

    evaluator = MulticlassClassificationEvaluator(labelCol="label")
    accuracy = evaluator.evaluate(pred, {evaluator.metricName: "accuracy"})
    precision = evaluator.evaluate(pred, {evaluator.metricName: "weightedPrecision"})
    recall = evaluator.evaluate(pred, {evaluator.metricName: "weightedRecall"})
    f1 = evaluator.evaluate(pred, {evaluator.metricName: "f1"})

    # Confusion Matrix
    cm = pred.groupBy("label", "prediction").count().toPandas()

    # Extract values
    TP = cm[(cm.label == 1) & (cm.prediction == 1)]["count"].sum()
    TN = cm[(cm.label == 0) & (cm.prediction == 0)]["count"].sum()
    FP = cm[(cm.label == 0) & (cm.prediction == 1)]["count"].sum()
    FN = cm[(cm.label == 1) & (cm.prediction == 0)]["count"].sum()

    # Detailed metrics
    TPR = TP / (TP + FN + 1e-9)
    TNR = TN / (TN + FP + 1e-9)
    FPR = FP / (FP + TN + 1e-9)
    FNR = FN / (TP + FN + 1e-9)

    prob = np.array([p[1] for p in pred.select("probability").toPandas()["probability"]])
    y_true = pred.select("label").toPandas()["label"].values
    brier = np.mean((prob - y_true) ** 2)

    metrics = {
        "AUC": auc,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TPR": TPR,
        "TNR": TNR,
        "FPR": FPR,
        "FNR": FNR,
        "BrierScore": brier,
        "TP": TP, "TN": TN, "FP": FP, "FN": FN
    }

    return metrics, cm


def save_evaluation_to_file(metrics, cm, folder="../results"):
    os.makedirs(folder, exist_ok=True)

    # Metrics → CSV
    pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"]) \
        .to_csv(f"{folder}/model_evaluation.csv", index=False)

    # Confusion matrix → CSV
    cm.to_csv(f"{folder}/confusion_matrix.csv", index=False)

    print("📁 Đã lưu toàn bộ đánh giá mô hình dạng CSV vào thư mục:", folder)
