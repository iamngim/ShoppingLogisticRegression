from flask import Flask, render_template, request, redirect, send_from_directory
import os, subprocess, pickle, numpy as np, pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(BASE_DIR, "images")
RESULT_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(BASE_DIR, "models")

for d in [DATA_DIR, IMAGE_DIR, RESULT_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

app = Flask(__name__, template_folder=os.path.join(SRC_DIR, "views"))

@app.route("/images/<path:filename>")
def images(filename):
    return send_from_directory(IMAGE_DIR, filename)

@app.route("/results/<path:filename>")
def results(filename):
    return send_from_directory(RESULT_DIR, filename)

@app.route("/")
def home():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files["file"]
    f.save(os.path.join(DATA_DIR, "data.csv"))

    subprocess.run(
        ["python", os.path.join(SRC_DIR, "MainModel.py")],
        check=True
    )

    return redirect("/training")

@app.route("/training")
def training():
    return render_template("training.html")

@app.route("/evaluation")
def evaluation():
    metrics = pd.read_csv(
        os.path.join(RESULT_DIR, "model_evaluation.csv")
    ).to_dict("records")

    return render_template(
        "evaluation.html",
        metrics=metrics
    )

@app.route("/descriptive")
def descriptive():
    path = os.path.join(RESULT_DIR, "descriptive_statistics.csv")

    if not os.path.exists(path):
        return render_template(
            "descriptive.html",
            stats=[],
            warning="⚠️ Chưa có thống kê mô tả. Vui lòng upload và huấn luyện dữ liệu trước."
        )

    stats = pd.read_csv(path).to_dict("records")

    return render_template("descriptive.html", stats=stats)



@app.route("/explain")
def explain():
    with open(os.path.join(MODEL_DIR, "logistic_model.pkl"), "rb") as f:
        model = pickle.load(f)

    return render_template(
        "explain.html",
        explain=list(zip(model["feature_cols"], model["coefficients"])),
        feature_img="feature_importance.png"
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    model_path = os.path.join(MODEL_DIR, "logistic_model.pkl")
    if not os.path.exists(model_path):
        return render_template(
            "predict.html",
            error="⚠️ Hiện chưa có mô hình được huấn luyện. Vui lòng upload dữ liệu và huấn luyện trước."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    features = model["feature_cols"]
    prob = None
    input_data = {}

    if request.method == "POST":
        try:
            # Lưu lại các giá trị đã nhập để hiển thị lại form
            input_data = {f: float(request.form[f]) for f in features}
            X = np.array([input_data[f] for f in features])

            # --- Chuẩn hóa giống lúc train ---
            if model.get("scaler_mean") is not None and model.get("scaler_std") is not None:
                X = (X - np.array(model["scaler_mean"])) / np.array(model["scaler_std"])

            # --- Tính xác suất Logistic Regression ---
            z = np.dot(X, np.array(model["coefficients"])) + model["intercept"]
            prob = 1 / (1 + np.exp(-z))

        except Exception as e:
            return render_template("predict.html", features=features, error=f"Lỗi dữ liệu nhập: {str(e)}")

    return render_template("predict.html", features=features, prob=prob, input_data=input_data)


if __name__ == "__main__":
    app.run(debug=True)
