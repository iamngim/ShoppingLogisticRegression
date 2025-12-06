from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression


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
        withMean=True    # chuẩn Z-score
    )

    scaler_model = scaler.fit(train_df)

    # Transform cả train + test bằng scaler đã fit
    train_df = scaler_model.transform(train_df)
    test_df = scaler_model.transform(test_df)

    return train_df, test_df


def train_model(df, feature_cols):
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # CHUẨN HOÁ FEATURE
    train_df, test_df = prepare_features(train_df, test_df, feature_cols)

    # Logistic Regression
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=50,
        regParam=0.01,
        elasticNetParam=0.0,
        seed=42
    )

    model = lr.fit(train_df)
    predictions = model.transform(test_df)

    return model, predictions
