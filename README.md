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
# hoặc cài tay những gói thường dùng
pip install opencv-python pillow pytesseract numpy pandas flask fastapi streamlit
# nếu dùng deep learning
pip install torch torchvision  # hoặc tensorflow
```

> **Lưu ý:** Kiểm tra file `requirements.txt` trong repo (nếu có) để cài chính xác các phụ thuộc.

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
python app.py
# hoặc
streamlit run app.py
# hoặc
python -m flask run --host=0.0.0.0 --port=5000
```

> Cách chạy chính xác tuỳ thuộc vào nội dung `app.py`. Kiểm tra phần đầu file `app.py` để biết framework (Flask / Streamlit / FastAPI / Click) mà repo sử dụng.

## Ví dụ sử dụng (pseudo)

```python
from ocr import ocr_engine
from extractor import extract_cccd_fields
from preprocessing import preprocess_image

img = preprocess_image('data/sample/sample_cccd.jpg')
text = ocr_engine(img)
fields = extract_cccd_fields(text)
print(fields)
```

## Định dạng dữ liệu / nhãn

* Nếu repo kèm tập dữ liệu, hãy lưu ảnh gốc vào `data/images/` và file nhãn (JSON/CSV) vào `data/labels/`.
* Mẫu định dạng nhãn (CSV):

```
filename,full_name,id_number,dob,gender,address
sample_cccd.jpg,Nguyen Van A,0123456789,1990-01-01,M,Ha Noi
```

## Huấn luyện mô hình (hướng dẫn chung)

1. Chuẩn hoá và gán nhãn dữ liệu trong `data/`.
2. Viết script training trong `model/` hoặc dùng notebook để thử nghiệm.
3. Lưu weights vào `model/checkpoints/` và cập nhật `config/` để app tải weights.

## Gợi ý cải tiến

* Thử các engine OCR khác nhau: Tesseract, EasyOCR, Google Vision API.
* Sử dụng mô hình sequence labelling (CRNN, Transformer OCR) nếu cần độ chính xác cao hơn.
* Kết hợp bước hậu xử lý ngôn ngữ (PhoBERT / VnCoreNLP) để sửa lỗi chính tả và chuẩn hoá tên/địa chỉ.

## Đóng góp

Hoan nghênh PR, issue và góp ý. Vui lòng mở issue mô tả rõ lỗi / tính năng mong muốn và cách tái tạo.

## License

Nếu bạn là chủ repo, hãy thêm file LICENSE (ví dụ: MIT) hoặc cho biết giấy phép bạn muốn dùng.

---

### Liên hệ

Tác giả repo: `denghwee` (xem trang GitHub để liên hệ). Nếu bạn muốn, tôi có thể:

* Sinh file `README.md` hoàn chỉnh và format sẵn sàng commit.
* Thêm badge (Build / License / Python version) nếu bạn cung cấp thông tin.
* Tùy chỉnh README theo nội dung chi tiết hơn sau khi bạn cho biết framework trong `app.py` hoặc file phụ thuộc (`requirements.txt`).

---

*Generated on request.*
