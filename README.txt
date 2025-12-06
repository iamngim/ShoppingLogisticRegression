📘 README.md — Shopping Behavior Prediction with Logistic Regression (Apache Spark MLlib)
📌 Giới thiệu

Dự án này thực hiện phân tích và dự báo hành vi mua sắm theo từng quý của khách hàng dựa trên dữ liệu giao dịch thương mại điện tử.
Mô hình sử dụng:

 - Apache Spark MLlib để xử lý dữ liệu lớn và huấn luyện mô hình.

 - Logistic Regression để dự báo khả năng khách hàng tiếp tục mua hàng trong quý tiếp theo (label 0/1).

 - Kết hợp RFM analysis, đặc trưng theo quý, xu hướng mua sắm và nhiều feature nâng cao.

Dự án bao gồm:

 - Tiền xử lý dữ liệu (DataPreprocessing.py)

 - Thống kê mô tả (DescriptiveAnalysis.py)

 - Huấn luyện mô hình Logistic Regression (ModelLogisticRegression.py)

 - Đánh giá mô hình (ModelEvaluation.py)

 - Vẽ biểu đồ ROC / Feature importance (Chart.py)

 - Chương trình chính chạy pipeline (MainModel.py)

📂 Cấu trúc thư mục
    Project/
    │
    ├── src/
    │   ├── DataPreprocessing.py
    │   ├── DescriptiveAnalysis.py
    │   ├── ModelLogisticRegression.py
    │   ├── ModelEvaluation.py
    │   ├── Chart.py
    │   └── MainModel.py
    │
    ├── data/
    │   ├── data.csv              # dữ liệu đầu vào
    │   └── data_final.csv        # dữ liệu sau khi xử lý
    │
    ├── results/
    │   ├── descriptive_stats.csv
    │   ├── label_distribution.csv
    │   ├── quartiles.csv
    │   ├── skewness_kurtosis.csv
    │   ├── model_evaluation.csv
    │   └── confusion_matrix.csv
    │
    └── README.md

🧱 1. Tiền xử lý dữ liệu
File: DataPreprocessing.py

Các bước chính:

    ✔ Làm sạch dữ liệu

        - Loại bỏ dòng thiếu Invoice, StockCode, Quantity, Price, InvoiceDate

        - Bỏ giao dịch Quantity ≤ 0 và Price ≤ 0

        - Chuyển InvoiceDate → timestamp
    ✔ Tạo các feature:
        - RFM & nâng cao:

            Recency
            Frequency
            Monetary
            CustomerLifeSpan
            TotalQuarters
            AvgOrderValue
            MonetaryPerQuarter

        - Theo quý (Quarter-based)

            QuarterAmount
            QuarterFrequency
            QuarterAvgValue

        - Xu hướng hành vi (Trend)

            PurchaseTrend3Q (tăng, giảm hay không đổi so với quý trước)

    ✔ Gán nhãn (label)

        - Label = 1 nếu khách hàng có mua trong quý tiếp theo

        - Label = 0 nếu không có giao dịch tiếp theo
    ✔ Xuất dữ liệu cuối cùng

        - Dữ liệu được lưu tại: data/data_final.csv

📊 2. Thống kê mô tả
File: DescriptiveAnalysis.py

Bao gồm:

    ✔ Describe() cho tất cả biến số

    ✔ Phân phối nhãn (label distribution)

    ✔ Tính Quartiles (Q1, Median, Q3)

    ✔ Tính Skewness và Kurtosis

    ✔ Xuất file CSV vào thư mục results/

🤖 3. Huấn luyện Logistic Regression
File: ModelLogisticRegression.py

    ✔ Chuẩn hóa dữ liệu đúng chuẩn (Tránh leakage)

        - Chia train/test trước

        - Fit StandardScaler trên train

        - Transform test bằng scaler đã học

        - Chuẩn Z-score (withMean=True, withStd=True)

    ✔ Train Logistic Regression

📈 4. Đánh giá mô hình
File: ModelEvaluation.py

Tất cả kết quả được lưu vào:

    ✔ results/model_evaluation.csv

    ✔ results/confusion_matrix.csv

📉 5. Biểu đồ trực quan
File: Chart.py

    ✔ ROC Curve

    ✔ Phân phối nhãn

    ✔ Độ quan trọng feature (Coefficient weights)

🚀 6. Chạy toàn bộ pipeline
File: MainModel.py

Chạy lệnh: 'python MainModel.py' hoặc Run file 'MainModel.py'

Pipeline sẽ tự động:

    1. Đọc data_final.csv

    2. Sinh thống kê mô tả vào results/

    3. Train Logistic Regression

    4. Đánh giá mô hình

    5. Vẽ biểu đồ ROC & Feature Importance

    6. Hoàn thành báo cáo

📌 7. Yêu cầu hệ thống
    |     Thành phần      |     Phiên bản       |
    | ------------------- | ------------------- |
    | Python              | 3.8–3.10            |
    | Apache Spark        | 3.x                 |
    | PySpark             | Tương thích Spark   |
    | Pandas              | ≥ 1.3               |
    | Matplotlib          | ≥ 3.x               |
    | NumPy               | ≥ 1.20              |

❤️ 7. Liên hệ / Đóng góp
Bạn có thể mở issue hoặc gửi thêm yêu cầu để mở rộng mô hình, cải thiện kết quả hoặc thêm dashboard trực quan.
