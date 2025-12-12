from numpy import sign
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

import os


# Tạo Spark session
def create_spark():
    spark = SparkSession.builder \
        .appName("ShoppingLogisticRegression") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
    return spark


# Load dữ liệu
def load_data(spark, path):
    df = spark.read.csv(path, header=True, inferSchema=True)
    print("📥 Đã load dữ liệu:", df.count(), "dòng")
    return df


# Làm sạch dữ liệu
def clean_data(df):
    df = df.dropna(subset=["Invoice", "StockCode", "Quantity", "Price", "InvoiceDate"])
    df = df.filter(col("Quantity") > 0).filter(col("Price") > 0)

    df = df.withColumn("InvoiceDate",
                       to_timestamp("InvoiceDate", "M/d/yyyy H:mm"))

    df = df.withColumn("Year", year("InvoiceDate")) \
        .withColumn("Quarter", quarter("InvoiceDate")) \
        .withColumn("TotalAmount", col("Quantity") * col("Price"))

    return df


# Feature RFM + lifespan
def feature_rfm(df):
    max_ts = df.agg(max("InvoiceDate")).collect()[0][0]

    rfm = df.groupBy("Customer ID").agg(
        (datediff(lit(max_ts), max("InvoiceDate"))).alias("Recency"),
        countDistinct("Invoice").alias("Frequency"),
        sum("TotalAmount").alias("Monetary"),
        (datediff(max("InvoiceDate"), min("InvoiceDate"))).alias("CustomerLifeSpan"),
        countDistinct("Year", "Quarter").alias("TotalQuarters")
    )

    rfm = rfm.withColumn("AvgOrderValue", col("Monetary") / col("Frequency"))
    rfm = rfm.withColumn("MonetaryPerQuarter", col("Monetary") / col("TotalQuarters"))

    return rfm


# Feature theo quý
def feature_quarter(df):
    q = df.groupBy("Customer ID", "Year", "Quarter") \
        .agg(sum("TotalAmount").alias("QuarterAmount"),
             countDistinct("Invoice").alias("QuarterFrequency"))

    q = q.withColumn("QuarterAvgValue",
                     col("QuarterAmount") / col("QuarterFrequency"))
    return q


# Feature xu hướng (trend quý)
def feature_trend(quarter_df):
    w = Window.partitionBy("Customer ID").orderBy("Year", "Quarter")

    quarter_df = quarter_df.withColumn(
        "PrevQuarterAmount",
        lag("QuarterAmount", 1).over(w)
    )

    quarter_df = quarter_df.withColumn(
        "PurchaseTrend3Q",
        when(col("PrevQuarterAmount").isNull(), 0)
        .otherwise(sign(col("QuarterAmount") - col("PrevQuarterAmount")))
    )

    return quarter_df


# Tạo label dự báo quý tiếp theo
def create_label(qdf):
    next_q = qdf.alias("cur").join(
        qdf.alias("nxt"),
        [
            col("cur.Customer ID") == col("nxt.Customer ID"),
            (col("nxt.Year") * 4 + col("nxt.Quarter")) ==
            (col("cur.Year") * 4 + col("cur.Quarter") + 1)
        ],
        "left"
    ).select(
        col("cur.Customer ID").alias("CustomerID"),
        col("cur.Year"),
        col("cur.Quarter"),
        col("cur.QuarterAmount"),
        col("cur.QuarterFrequency"),
        col("cur.QuarterAvgValue"),
        col("cur.PurchaseTrend3Q"),

        when(col("nxt.QuarterAmount").isNull(), 0)
        .otherwise((col("nxt.QuarterAmount") > 0).cast("int"))
        .alias("label")
    )

    return next_q


# Hợp nhất tất cả features
def merge_features(label_df, rfm):
    # Lấy danh sách khách hàng hợp lệ (đã có RFM)
    valid_customers = rfm.select(col("Customer ID").alias("ID_valid"))

    # Chỉ giữ những khách nằm trong RFM
    label_df = label_df.join(valid_customers, label_df.CustomerID == valid_customers.ID_valid, "inner")

    # Join RFM
    final = label_df.join(rfm, label_df.CustomerID == rfm["Customer ID"], "left") \
        .drop("Customer ID", "ID_valid")

    cols = [c for c in final.columns if c != "label"] + ["label"]
    return final.select(cols)


# Save file output - Dùng Pandas thay vì Spark CSV
def save_output(df):
    output_folder = "../data"

    # Tạo thư mục output nếu chưa tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Chuyển Spark DataFrame sang Pandas
    pandas_df = df.toPandas()

    # Lưu file CSV bằng Pandas (không cần Hadoop)
    output_path = os.path.join(output_folder, "data_final.csv")
    pandas_df.to_csv(output_path, index=False)

    print("Lưu thành công tại: ", os.path.normpath(output_path))


# Pipeline chính
def preprocess(path):
    spark = create_spark()

    df = load_data(spark, path)
    df = clean_data(df)

    rfm = feature_rfm(df)
    quarterly = feature_quarter(df)
    quarterly = feature_trend(quarterly)

    label_df = create_label(quarterly)

    final_df = merge_features(label_df, rfm)

    save_output(final_df)

    print("Pipeline hoàn tất. Tổng số dòng:", final_df.count())
    return final_df


if __name__ == "__main__":
    preprocess("../data/data.csv")