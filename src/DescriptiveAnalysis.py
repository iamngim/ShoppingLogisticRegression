from pyspark.sql.functions import *
import pandas as pd
import os


def descriptive_statistics(df):
    """
    In thống kê mô tả ra console (phục vụ debug + bảo vệ)
    """
    print("\n" + "=" * 70)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 70)

    # 1. Tổng quan
    df.describe().show(truncate=False)

    # 2. Phân phối nhãn
    print("\n📌 LABEL DISTRIBUTION")
    df.groupBy("label").count().orderBy("label").show()

    print("=" * 70)


def save_descriptive_to_file(df):
    """
    Lưu toàn bộ thống kê mô tả phục vụ Flask
    """
    # ===== PATH =====
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SRC_DIR)
    RESULT_DIR = os.path.join(BASE_DIR, "results")

    os.makedirs(RESULT_DIR, exist_ok=True)

    # ===== NUMERIC COLUMNS =====
    numeric_cols = [
        c for c, t in df.dtypes
        if t in ("int", "double") and c not in ["CustomerID", "label"]
    ]

    summary = []

    for c in numeric_cols:
        q1, median, q3 = df.approxQuantile(c, [0.25, 0.5, 0.75], 0.01)

        row = {
            "Feature": c,
            "Count": df.count(),
            "Mean": df.select(mean(c)).first()[0],
            "Median": median,
            "Std": df.select(stddev(c)).first()[0],
            "Variance": df.select(variance(c)).first()[0],
            "Min": df.select(min(c)).first()[0],
            "Max": df.select(max(c)).first()[0],
            "Q1": q1,
            "Q3": q3,
            "Skewness": df.select(skewness(c)).first()[0],
            "Kurtosis": df.select(kurtosis(c)).first()[0]
        }

        summary.append(row)

    # ===== SAVE CSV =====
    pd.DataFrame(summary).to_csv(
        os.path.join(RESULT_DIR, "descriptive_statistics.csv"),
        index=False
    )

    # Label distribution
    df.groupBy("label").count().toPandas().to_csv(
        os.path.join(RESULT_DIR, "label_distribution.csv"),
        index=False
    )

    print("✅ Đã lưu thống kê mô tả:")
    print("   - results/descriptive_statistics.csv")
    print("   - results/label_distribution.csv")
