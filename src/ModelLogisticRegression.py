from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
import os
import json
import pickle
import numpy as np


def prepare_features(train_df, test_df, feature_cols):
    # Assemble features
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="raw_features"
    )
    train_df = assembler.transform(train_df)
    test_df = assembler.transform(test_df)

    # StandardScaler
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withStd=True,
        withMean=True  # chuẩn Z-score
    )

    scaler_model = scaler.fit(train_df)

    # Transform cả train + test bằng scaler đã fit
    train_df = scaler_model.transform(train_df)
    test_df = scaler_model.transform(test_df)

    return train_df, test_df, scaler_model


def train_model(df, feature_cols):
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # CHUẨN HOÁ FEATURE
    train_df, test_df, scaler_model = prepare_features(train_df, test_df, feature_cols)

    # Logistic Regression
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=50,
        regParam=0.01,
        elasticNetParam=0.0
    )

    model = lr.fit(train_df)
    predictions = model.transform(test_df)

    return model, predictions, scaler_model


def save_model_pkl(model, scaler_model, feature_cols, pkl_path="../models/logistic_model.pkl"):
    """Lưu model + scaler dạng .pkl để dùng cho web dự báo"""
    os.makedirs(os.path.dirname(pkl_path) or ".", exist_ok=True)

    # Lấy coefficients và intercept từ Spark model
    coefficients = model.coefficients.toArray().tolist()
    intercept = float(model.intercept)

    # Lấy mean và std từ scaler
    scaler_mean = scaler_model.mean.toArray().tolist() if scaler_model.mean is not None else None
    scaler_std = scaler_model.std.toArray().tolist() if hasattr(scaler_model,
                                                                'std') and scaler_model.std is not None else None

    # Tạo dict chứa thông tin model + scaler
    model_data = {
        "coefficients": coefficients,
        "intercept": intercept,
        "feature_cols": feature_cols,
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
        "model_type": "LogisticRegression"
    }

    # Lưu thành .pkl
    with open(pkl_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"✅ Lưu model .pkl thành công tại: {os.path.normpath(pkl_path)}")