from pyspark.sql.functions import *
import pandas as pd
import os


def descriptive_statistics(df):
    print("\n📌 DESCRIPTIVE STATISTICS")
    df.describe().show()

    print("\n📌 LABEL DISTRIBUTION")
    df.groupBy("label").count().orderBy("label").show()


def save_descriptive_to_file(df, folder="../results"):
    os.makedirs(folder, exist_ok=True)

    # 1. Describe()
    desc = df.describe().toPandas()
    desc.to_csv(f"{folder}/descriptive_stats.csv", index=False)

    # 2. Label distribution
    label_dist = df.groupBy("label").count().toPandas()
    label_dist.to_csv(f"{folder}/label_distribution.csv", index=False)

    # 3. Quartiles (Q1, Median, Q3)
    q_data = []
    numeric_cols = [c for c in df.columns if c not in ["CustomerID", "label"]]
    for col_name in numeric_cols:
        q = df.approxQuantile(col_name, [0.25, 0.5, 0.75], 0.01)
        q_data.append([col_name, q[0], q[1], q[2]])

    pd.DataFrame(q_data, columns=["Feature", "Q1", "Median", "Q3"]) \
        .to_csv(f"{folder}/quartiles.csv", index=False)

    # 4. Skewness + Kurtosis
    sk_data = []
    for col_name in numeric_cols:
        sk = df.select(skewness(col_name)).first()[0]
        ku = df.select(kurtosis(col_name)).first()[0]
        sk_data.append([col_name, sk, ku])

    pd.DataFrame(sk_data, columns=["Feature", "Skewness", "Kurtosis"]) \
        .to_csv(f"{folder}/skewness_kurtosis.csv", index=False)

    print("📁 Đã lưu toàn bộ thống kê mô tả dạng CSV vào thư mục:", folder)
