import pandas as pd
import numpy as np
import os
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

def evaluate_model(pred):
    # BASIC METRICS
    evaluator_auc = BinaryClassificationEvaluator(labelCol="label")
    auc_score = evaluator_auc.evaluate(pred)

    evaluator = MulticlassClassificationEvaluator(labelCol="label")
    accuracy = evaluator.evaluate(pred, {evaluator.metricName: "accuracy"})
    precision = evaluator.evaluate(pred, {evaluator.metricName: "weightedPrecision"})
    recall = evaluator.evaluate(pred, {evaluator.metricName: "weightedRecall"})
    f1 = evaluator.evaluate(pred, {evaluator.metricName: "f1"})

    # CONFUSION MATRIX
    cm = pred.groupBy("label", "prediction").count().toPandas()

    TP = cm[(cm.label == 1) & (cm.prediction == 1)]["count"].sum() if not cm[
        (cm.label == 1) & (cm.prediction == 1)].empty else 0
    TN = cm[(cm.label == 0) & (cm.prediction == 0)]["count"].sum() if not cm[
        (cm.label == 0) & (cm.prediction == 0)].empty else 0
    FP = cm[(cm.label == 0) & (cm.prediction == 1)]["count"].sum() if not cm[
        (cm.label == 0) & (cm.prediction == 1)].empty else 0
    FN = cm[(cm.label == 1) & (cm.prediction == 0)]["count"].sum() if not cm[
        (cm.label == 1) & (cm.prediction == 0)].empty else 0
    TPR = TP / (TP + FN + 1e-9)
    TNR = TN / (TN + FP + 1e-9)
    FPR = FP / (FP + TN + 1e-9)
    FNR = FN / (TP + FN + 1e-9)

    # PROBABILITY-BASED METRICS
    prob = np.array([p[1] for p in pred.select("probability").toPandas()["probability"]])
    y_true = pred.select("label").toPandas()["label"].values
    brier = np.mean((prob - y_true) ** 2)

    # Log Loss
    epsilon = 1e-15
    log_loss = -np.mean(y_true * np.log(prob + epsilon) + (1 - y_true) * np.log(1 - prob + epsilon))

    specificity = TNR
    precision_binary = TP / (TP + FP + 1e-9)
    recall_binary = TP / (TP + FN + 1e-9)
    f1_binary = 2 * (precision_binary * recall_binary) / (precision_binary + recall_binary + 1e-9)
    mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-9)
    youdens_j = TPR - FPR
    try:
        from sklearn.metrics import precision_recall_curve, auc as sk_auc
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, prob)
        pr_auc = sk_auc(recall_curve, precision_curve)
    except:
        pr_auc = 0.0

    # TỔNG HỢP TẤT CẢ METRICS
    metrics = {
        # Basic metrics
        "Accuracy": accuracy,
        "AUC-ROC": auc_score,
        "AUC-PR": pr_auc,

        # Binary metrics
        "Precision": precision_binary,
        "Recall": recall_binary,
        "Specificity": specificity,
        "F1-Score": f1_binary,

        # Weighted metrics (cho multiclass)
        "Weighted-Precision": precision,
        "Weighted-Recall": recall,
        "Weighted-F1": f1,

        # Rate metrics
        "TPR (Sensitivity)": TPR,
        "TNR (Specificity)": TNR,
        "FPR": FPR,
        "FNR": FNR,

        # Advanced metrics
        "Matthews Correlation Coeff (MCC)": mcc,
        "Youdens J Statistic": youdens_j,
        "Brier Score": brier,
        "Log Loss": log_loss,

        # Confusion Matrix components
        "TP": int(TP),
        "TN": int(TN),
        "FP": int(FP),
        "FN": int(FN)
    }

    # In chi tiết
    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n🔹 BASIC METRICS:")
    print(f"  Accuracy:        {metrics['Accuracy']:.4f}")
    print(f"  AUC-ROC:         {metrics['AUC-ROC']:.4f}")
    print(f"  AUC-PR:          {metrics['AUC-PR']:.4f}")

    print(f"\n🔹 BINARY CLASSIFICATION METRICS:")
    print(f"  Precision:       {metrics['Precision']:.4f}")
    print(f"  Recall:          {metrics['Recall']:.4f}")
    print(f"  Specificity:     {metrics['Specificity']:.4f}")
    print(f"  F1-Score:        {metrics['F1-Score']:.4f}")

    print(f"\n🔹 ADVANCED METRICS:")
    print(f"  MCC:             {metrics['Matthews Correlation Coeff (MCC)']:.4f}")
    print(f"  Youdens J:       {metrics['Youdens J Statistic']:.4f}")
    print(f"  Brier Score:     {metrics['Brier Score']:.4f}")
    print(f"  Log Loss:        {metrics['Log Loss']:.4f}")

    print(f"\n🔹 CONFUSION MATRIX:")
    print(f"  TP: {int(TP)}, TN: {int(TN)}")
    print(f"  FP: {int(FP)}, FN: {int(FN)}")
    print(f"  Total: {int(TP + TN + FP + FN)}")
    print("=" * 60 + "\n")

    return metrics, cm


def save_evaluation_to_file(metrics, cm):
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SRC_DIR)
    RESULT_DIR = os.path.join(BASE_DIR, "results")

    os.makedirs(RESULT_DIR, exist_ok=True)

    # Metrics → CSV
    metrics_df = pd.DataFrame(
        [{"Metric": k, "Value": v} for k, v in metrics.items()]
    )
    metrics_df.to_csv(os.path.join(RESULT_DIR, "model_evaluation.csv"), index=False)

    # Confusion Matrix (2x2) → CSV
    cm.to_csv(os.path.join(RESULT_DIR, "confusion_matrix.csv"))

    print("✅ Đã lưu đánh giá mô hình:")
    print(f"   - {RESULT_DIR}/model_evaluation.csv")
    print(f"   - {RESULT_DIR}/confusion_matrix.csv")
