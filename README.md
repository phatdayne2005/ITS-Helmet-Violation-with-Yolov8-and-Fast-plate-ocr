# Hệ thống Giám sát ATGT - Phát hiện vi phạm không đội nón bảo hiểm

## Cấu trúc dự án

```
Chuyende_V2.0_ORIGINAL/
├── main.py                 # File chính - Giao diện và logic xử lý
├── module/                 # Thư mục chứa các module
│   ├── __init__.py
│   ├── traffic_core.py     # Core tracking và data structures
│   ├── persistence.py      # Quản lý dữ liệu và lưu trữ
│   └── email.py           # Hệ thống gửi email cảnh báo
├── model/                  # Thư mục chứa các model AI
│   ├── motorbike.pt       # Model phát hiện xe máy
│   └── helmet_lp.pt       # Model phát hiện nón bảo hiểm và biển số
├── data/                   # Dữ liệu được tạo ra
│   ├── tracks/            # Thông tin tracking
│   ├── incidents/         # Các vi phạm được ghi nhận
│   ├── observations/      # Các quan sát bình thường
│   └── full_frame/        # Ảnh toàn cảnh
└── requirements.txt       # Các thư viện cần thiết
```

## Tính năng chính

### 1. Phát hiện đối tượng
- **Stage 1**: Phát hiện xe máy sử dụng YOLO
- **Stage 2**: Phát hiện nón bảo hiểm và biển số xe
- **OCR**: Nhận dạng biển số xe sử dụng fast-plate-ocr

### 2. Tracking và phân tích
- Theo dõi đối tượng qua các frame video
- Sử dụng EMA (Exponential Moving Average) để ổn định kết quả
- Fusion text cho biển số xe
- Đánh giá chất lượng ảnh để chọn snapshot tốt nhất

### 3. Lưu trữ dữ liệu
- Tự động tạo incident khi phát hiện vi phạm
- Lưu trữ evidence (ảnh) cho mỗi vi phạm
- Quản lý thông tin chủ xe từ database

### 4. Thông báo
- Gửi email cảnh báo tự động khi có vi phạm
- Email HTML với ảnh minh chứng inline
- Cấu hình SMTP linh hoạt

## Cách sử dụng

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị models
- Đặt file `motorbike.pt` vào thư mục `model/`
- Đặt file `helmet_lp.pt` vào thư mục `model/`

### 3. Chạy ứng dụng
```bash
python main.py
```

### 4. Cấu hình
- **Models**: Đường dẫn model được đặt sẵn, có thể thay đổi trong giao diện
- **Email**: Cập nhật thông tin SMTP trong file `main.py`
- **Database**: Thêm thông tin chủ xe vào `data/vehicles.json`

## Cải tiến giao diện

### Tự động load models
- Models được load tự động khi khởi động
- Đường dẫn mặc định: `model/motorbike.pt` và `model/helmet_lp.pt`
- Không cần chọn lại model mỗi lần mở ứng dụng

### Cấu trúc module hóa
- Code được tổ chức thành các module riêng biệt
- Dễ bảo trì và mở rộng
- Tên file rõ ràng và dễ hiểu

### Giao diện thân thiện
- Tiêu đề ứng dụng rõ ràng
- Thông báo lỗi chi tiết
- Log real-time cho người dùng

## Phát triển

### Thêm model mới
1. Đặt file model vào thư mục `model/`
2. Cập nhật đường dẫn trong `main.py`
3. Thêm logic xử lý tương ứng

### Mở rộng tính năng
- Thêm module mới vào thư mục `module/`
- Import và sử dụng trong `main.py`
- Cập nhật giao diện nếu cần

### Tùy chỉnh email
- Chỉnh sửa template trong `module/email.py`
- Cập nhật cấu hình SMTP trong `main.py`
- Thêm tính năng email mới nếu cần
