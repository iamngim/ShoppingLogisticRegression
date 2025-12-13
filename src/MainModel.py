from pyspark.sql import SparkSession
from DescriptiveAnalysis import descriptive_statistics, save_descriptive_to_file
from ModelLogisticRegression import train_model, save_model_pkl
from ModelEvaluation import evaluate_model, save_evaluation_to_file
from Chart import (
    plot_roc,
    plot_label_distribution,
    plot_feature_importance,
    extract_roc_points
)
import os


def main():
    # ================= SPARK =================
    spark = SparkSession.builder.appName("MainModel").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")  # gọn log khi demo

    # ================= PATH =================
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))   # src/
    BASE_DIR = os.path.dirname(SRC_DIR)                    # project root

    DATA_DIR = os.path.join(BASE_DIR, "data")
    IMAGE_DIR = os.path.join(BASE_DIR, "images")
    MODEL_DIR = os.path.join(BASE_DIR, "models")

    os.makedirs(IMAGE_DIR, exist_ok=True)

    # ================= LOAD DATA =================
    df = spark.read.csv(
        os.path.join(DATA_DIR, "data_final.csv"),
        header=True,
        inferSchema=True
    )

    feature_cols = [
        "Year", "Quarter", "QuarterAmount", "QuarterFrequency", "QuarterAvgValue",
        "PurchaseTrend3Q", "Recency", "Frequency", "Monetary",
        "CustomerLifeSpan", "TotalQuarters", "AvgOrderValue", "MonetaryPerQuarter"
    ]

    # ================= 1. DESCRIPTIVE =================
    descriptive_statistics(df)
    save_descriptive_to_file(df)

    # ================= 2. TRAIN =================
    model, predictions, scaler_model = train_model(df, feature_cols)

    # ================= 3. SAVE MODEL =================
    save_model_pkl(model, scaler_model, feature_cols)

    # ================= 4. EVALUATION =================
    metrics, cm = evaluate_model(predictions)
    save_evaluation_to_file(metrics, cm)

    # ================= 5. ROC =================
    fpr, tpr = extract_roc_points(predictions)
    plot_roc(
        fpr,
        tpr,
        metrics["AUC-ROC"],
        save_path=os.path.join(IMAGE_DIR, "roc_curve.png")
    )

    # ================= 6. LABEL DISTRIBUTION =================
    plot_label_distribution(
        df,
        save_path=os.path.join(IMAGE_DIR, "label_distribution.png")
    )

    # ================= 7. FEATURE IMPORTANCE =================
    plot_feature_importance(
        model.coefficients,
        feature_cols,
        save_path=os.path.join(IMAGE_DIR, "feature_importance.png")
    )

    print("\n🎉 MAIN MODEL COMPLETE!")
    spark.stop()   # QUAN TRỌNG


if __name__ == "__main__":
    main()
