from pyspark.sql import SparkSession
from DescriptiveAnalysis import descriptive_statistics, save_descriptive_to_file
from ModelLogisticRegression import train_model, save_model_pkl
from ModelEvaluation import evaluate_model, save_evaluation_to_file

from Chart import plot_roc, plot_label_distribution, plot_feature_importance, extract_roc_points


def main():
    spark = SparkSession.builder.appName("MainModel").getOrCreate()

    df = spark.read.csv("../data/data_final.csv", header=True, inferSchema=True)

    feature_cols = [
        "Year", "Quarter", "QuarterAmount", "QuarterFrequency", "QuarterAvgValue",
        "PurchaseTrend3Q", "Recency", "Frequency", "Monetary",
        "CustomerLifeSpan", "TotalQuarters", "AvgOrderValue", "MonetaryPerQuarter"
    ]

    # 1. DESCRIPTIVE ANALYSIS
    descriptive_statistics(df)
    save_descriptive_to_file(df)

    # 2. TRAIN MODEL
    model, predictions, scaler_model = train_model(df, feature_cols)

    # 3. SAVE MODEL
    save_model_pkl(model, scaler_model, feature_cols)  # Lưu model .pkl cho web

    # 4. EVALUATE MODEL (đầy đủ: AUC, F1, Precision, Recall, CM, Brier…)
    metrics, cm = evaluate_model(predictions)
    save_evaluation_to_file(metrics, cm)

    # 5. ROC CURVE
    fpr, tpr = extract_roc_points(predictions)
    plot_roc(fpr, tpr, metrics["AUC-ROC"])

    # 6. FEATURE IMPORTANCE
    plot_feature_importance(model.coefficients, feature_cols)

    print("\n🎉 MAIN MODEL COMPLETE!")


if __name__ == "__main__":
    main()