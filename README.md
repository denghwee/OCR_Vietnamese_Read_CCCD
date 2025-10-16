# OCR_Vietnamese_Read_CCCD

> Trích xuất thông tin bằng cách đọc OCR ảnh CCCD (Căn cước công dân Việt Nam)

---

## Mục tiêu

Dự án này cung cấp một pipeline để tiền xử lý ảnh CCCD, thực hiện OCR và trích xuất các trường thông tin quan trọng (ví dụ: họ tên, số định danh cá nhân, ngày sinh, giới tính, quê quán, địa chỉ...). Mục tiêu là dễ sử dụng, có thể thử nghiệm nhanh và mở rộng để huấn luyện hoặc cải thiện mô hình nhận dạng.

## Tính năng chính

* Tiền xử lý ảnh (làm sạch, cân bằng sáng, chỉnh hướng, cắt vùng văn bản).
* Nhận dạng ký tự (OCR) kết hợp với mô-đun trích xuất thông tin (extractor).
* Cấu trúc thư mục rõ ràng: `preprocessing/`, `ocr/`, `model/`, `extractor/`, `data/`, `config/`.
* Ứng dụng mẫu (`app.py`) để chạy nhanh thử nghiệm trên ảnh.

## Yêu cầu

* Python 3.8+
* Khuyến nghị tạo virtual environment trước khi cài đặt

Gợi ý các thư viện có thể cần (tuỳ cách triển khai trong repo):

```bash
pip install -r requirements.txt  # nếu repo có file này
```

## Cấu trúc thư mục (tổng quan)

```
OCR_Vietnamese_Read_CCCD/
├─ app.py                # Ứng dụng demo / entrypoint
├─ config/               # Cấu hình (tham số, đường dẫn,...)
├─ data/                 # Tập dữ liệu mẫu, ảnh và nhãn
├─ preprocessing/        # Mã tiền xử lý ảnh
├─ ocr/                  # Wrapper cho engine OCR (pytesseract, easyocr, ...)
├─ extractor/            # Luồng trích xuất trường thông tin từ text thô
├─ model/                # Các mô hình / weights nếu có
├─ .gitignore
└─ README.md
```

## Hướng dẫn nhanh — Chạy local

1. Clone repo

```bash
git clone https://github.com/denghwee/OCR_Vietnamese_Read_CCCD.git
cd OCR_Vietnamese_Read_CCCD
```

2. Tạo virtual environment và cài phụ thuộc

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt  # nếu có
```

3. Chạy ứng dụng demo

```bash
streamlit run app.py
```

## License

MIT

---

### Liên hệ

Tác giả repo: `denghwee` (xem trang GitHub để liên hệ).
