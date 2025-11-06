# Hướng dẫn Codebase

Tài liệu này cung cấp giải thích chi tiết về codebase cho dự án Phát hiện Xâm nhập Mạng, bao gồm mục đích của từng script Python và hướng dẫn cách chạy dự án.

## Cấu trúc dự án

Dự án được tổ chức thành các thư mục sau:

```
network-intrusion-detection/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   └── processed/          # Các tệp dữ liệu đã xử lý (ví dụ: data.csv, train_data.csv, v.v.)
├── trained/             # Các mô hình đã huấn luyện và lịch sử huấn luyện
└── src/
    ├── pretrained/         # Các script để tiền xử lý dữ liệu
    │   └── split_data.py
    └── model/              # Các script/notebook huấn luyện và kiểm thử mô hình
        ├── cnn/
        │   ├── train_cnn.py
        │   └── test.ipynb
        ├── logistic_regression/
        │   ├── train_logistic_regression.py
        │   └── test.ipynb
        ├── random_forest/
        │   ├── train_random_forest.py
        │   └── test.ipynb
        ├── rnn/
        │   ├── train_rnn.py
        │   └── test.ipynb
        └── xgboost/
            ├── train_xgboost.py
            └── test.ipynb
```

- **data/processed/**: Thư mục này dùng để lưu trữ tập dữ liệu đã xử lý được sử dụng để huấn luyện và đánh giá.
- **trained/**: Thư mục này lưu trữ các mô hình học máy đã huấn luyện và lịch sử huấn luyện của chúng (nếu có).
- **src/pretrained/**: Thư mục này chứa các script để chuẩn bị và tiền xử lý dữ liệu.
- **src/model/**: Thư mục này chứa mã nguồn để huấn luyện và kiểm thử các mô hình học máy khác nhau.

## Giải thích các Script Python

### `src/pretrained/split_data.py`

Script này chịu trách nhiệm chia tập dữ liệu thô thành các tập huấn luyện, xác thực và kiểm thử. Nó thực hiện các bước sau:

1.  Tải tập dữ liệu từ `data/processed/data.csv`.
2.  Mã hóa cột `label` phân loại thành các giá trị số.
3.  Chia dữ liệu thành các tập huấn luyện (80%), xác thực (10%) và kiểm thử (10%).
4.  Lưu các tập dữ liệu đã chia dưới dạng `train_data.csv`, `val_data.csv` và `test_data.csv` trong thư mục `data/processed/`.

### `src/model/<tên_mô_hình>/train_<tên_mô_hình>.py`

Đây là một biểu diễn chung của các script huấn luyện cho từng mô hình (`cnn`, `logistic_regression`, `random_forest`, `rnn`, `xgboost`). Mỗi script tuân theo một mẫu tương tự:

1.  **Tải dữ liệu**: Tải các tập dữ liệu huấn luyện, xác thực và kiểm thử từ thư mục `data/processed/`.
2.  **Tiền xử lý dữ liệu**: Chuẩn bị dữ liệu cho mô hình cụ thể. Điều này có thể bao gồm chia tỷ lệ đặc trưng (đối với Logistic Regression, CNN và RNN) và định hình lại dữ liệu (đối với CNN và RNN).
3.  **Khởi tạo mô hình**: Khởi tạo mô hình học máy với các siêu tham số cụ thể.
4.  **Huấn luyện mô hình**: Huấn luyện mô hình trên dữ liệu huấn luyện. Đối với các mô hình hỗ trợ (CNN, RNN, XGBoost), nó sử dụng tập xác thực để theo dõi hiệu suất và lưu lịch sử huấn luyện.
5.  **Đánh giá mô hình**: Đánh giá mô hình đã huấn luyện trên tập kiểm thử và in báo cáo phân loại với độ chính xác, độ chính xác, độ thu hồi và điểm F1.
6.  **Lưu mô hình**: Lưu mô hình đã huấn luyện vào thư mục `trained/` để sử dụng sau này.

## Cách chạy dự án

Để chạy dự án, hãy làm theo các bước sau:

### 1. Thiết lập môi trường

Đầu tiên, đảm bảo bạn đã cài đặt Python và các phần phụ thuộc cần thiết. Nên sử dụng môi trường ảo.

```bash
# Tạo và kích hoạt môi trường ảo (tùy chọn nhưng được khuyến nghị)
python -m venv venv
# Trên Windows
.\venv\Scripts\activate
# Trên macOS/Linux
source venv/bin/activate

# Cài đặt các phần phụ thuộc
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

Đặt tập dữ liệu của bạn vào thư mục `data/processed/` và đặt tên là `data.csv`. Sau đó, chạy script chia dữ liệu:

```bash
python src/pretrained/split_data.py
```

Thao tác này sẽ tạo các tệp `train_data.csv`, `val_data.csv` và `test_data.csv` trong cùng thư mục.

### 3. Huấn luyện mô hình

Bạn có thể huấn luyện bất kỳ mô hình nào bằng cách chạy các script huấn luyện tương ứng của chúng. Ví dụ, để huấn luyện mô hình Random Forest:

```bash
python src/model/random_forest/train_random_forest.py
```

Thao tác này sẽ huấn luyện mô hình, in kết quả đánh giá và lưu mô hình đã huấn luyện vào thư mục `trained/`.

### 4. Kiểm thử mô hình

Để thực hiện kiểm thử và phân tích chi tiết hơn các mô hình đã huấn luyện, bạn có thể sử dụng các Jupyter Notebook được cung cấp trong thư mục của từng mô hình. Ví dụ, để kiểm thử mô hình Random Forest:

```bash
jupyter notebook src/model/random_forest/test.ipynb
```

Các notebook này sẽ tải mô hình đã huấn luyện, thực hiện dự đoán trên tập kiểm thử và hiển thị các số liệu hiệu suất bao gồm độ chính xác, báo cáo phân loại và ma trận nhầm lẫn.