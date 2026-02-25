# 🪪 OCR Vietnamese - Đọc & Trích xuất thông tin Căn Cước Công Dân

> Hệ thống OCR tự động để đọc và trích xuất thông tin từ ảnh Căn cước công dân (CCCD) Việt Nam

---

## 📋 Mục lục

- [Tổng quan](#tổng-quan)
- [Tính năng](#tính-năng)
- [Demo](#demo)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Huấn luyện mô hình](#huấn-luyện-mô-hình)
- [License](#license)

---

## 🎯 Tổng quan

Dự án này cung cấp một pipeline hoàn chỉnh để xử lý ảnh CCCD, thực hiện OCR và trích xuất các trường thông tin quan trọng như:
- **Số CCCD** (Số định danh cá nhân)
- **Họ và tên**
- **Ngày sinh**
- **Giới tính**
- **Quốc tịch**
- **Quê quán**
- **Địa chỉ thường trú**

Hệ thống được xây dựng với kiến trúc modular, dễ mở rộng và tùy chỉnh để phù hợp với các nhu cầu khác nhau.

---

## ✨ Tính năng

### 🔍 OCR Pipeline
- **Tiền xử lý ảnh thông minh:**
  - Tự động căn chỉnh và cắt vùng CCCD từ ảnh gốc
  - Làm sạch ảnh (tăng cường độ tương phản, loại bỏ nhiễu)
  - Chuẩn hóa kích thước và định hướng

- **Nhận dạng văn bản:**
  - Sử dụng **PaddleOCR** cho phát hiện văn bản
  - Sử dụng **VietOCR** (VGG-Transformer) cho nhận dạng ký tự tiếng Việt
  - Hỗ trợ GPU để tăng tốc xử lý

### 📝 Post-processing
- **Sửa lỗi chính tả tự động:**
  - Sửa các lỗi OCR phổ biến trong tiếng Việt
  - Cải thiện độ chính xác của văn bản được nhận dạng

- **Trích xuất thông tin có cấu trúc:**
  - Hỗ trợ 2 loại mô hình extractor:
    - **HuggingFace** (BERT-based NER model)
    - **Ollama** (LLM-based extraction)
  - Tự động phân tích và trích xuất các trường thông tin từ văn bản OCR

### 🖥️ Giao diện Web
- **Streamlit Dashboard:**
  - Upload và xử lý ảnh trực tiếp trên trình duyệt
  - Hiển thị kết quả từng bước (ảnh gốc, ảnh sau xử lý, OCR raw, thông tin trích xuất)
  - Tùy chọn cấu hình model extractor
  - Giao diện thân thiện, dễ sử dụng

---

## 🎬 Demo

### Giao diện chính

![Demo Interface](/data/image.png)

### Kết quả trích xuất

![Extraction Results](/data/image1.png)

---

## 🚀 Cài đặt

### Yêu cầu hệ thống

- **Python:** 3.8 trở lên
- **GPU:** Khuyến nghị (CUDA 12.8) để tăng tốc xử lý
- **RAM:** Tối thiểu 8GB (khuyến nghị 16GB+)
- **Disk:** ~5GB cho dependencies và models

### Các bước cài đặt

1. **Clone repository:**

```bash
git clone <repository-url>
cd OCRProject
```

2. **Tạo virtual environment:**

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

3. **Cài đặt dependencies:**

```bash
pip install -r requirements.txt
```

4. **Tải các mô hình cần thiết:**

- Mô hình aligner: `model/model_crop.pt` (đã có trong repo)
- Mô hình extractor: Tự động tải khi chạy lần đầu (hoặc đặt trong `saved_model/`)

---

## 💻 Sử dụng

### Chạy ứng dụng Web (Streamlit)

```bash
streamlit run app.py
```

Sau đó mở trình duyệt tại `http://localhost:8501`


---

## 📁 Cấu trúc dự án

```
OCRProject/
├── app.py                      # Ứng dụng Streamlit chính
├── config/                     # Cấu hình hệ thống
│   └── settings.py            # Cấu hình OCR, detector, recognizer
├── data/                       # Dữ liệu mẫu (ảnh CCCD)
│   ├── cccd2.jpg
│   └── cccd3.jpg
├── model/                      # Mô hình đã huấn luyện
│   └── model_crop.pt          # Mô hình căn chỉnh CCCD
├── ocr/                        # Module OCR
│   ├── pipeline.py            # Pipeline chính kết hợp các bước
│   ├── detector.py            # PaddleOCR detector
│   └── recognizer.py          # VietOCR recognizer
├── preprocessing/              # Tiền xử lý ảnh
│   ├── aligner.py             # Căn chỉnh và cắt vùng CCCD
│   └── image_cleaner.py       # Làm sạch ảnh
├── postprocessing/            # Hậu xử lý và trích xuất
│   ├── extractor_factory.py   # Factory pattern cho extractor
│   ├── huggingface_extractor.py  # HuggingFace NER extractor
│   ├── ollama_extractor.py    # Ollama LLM extractor
│   ├── spelling_correction.py # Sửa lỗi chính tả
│   └── field_extractor.py     # Base extractor class
├── saved_model/               # Các mô hình NER đã huấn luyện
│   ├── 1/ ... 6/              # Các checkpoint models
├── train_test/                # Scripts huấn luyện và đánh giá
│   ├── train.py               # Script huấn luyện NER model
│   ├── test_model.py          # Script test model
│   ├── gen_data.py            # Script tạo dữ liệu
│   └── *.json                 # Dataset train/val/test
├── output/                     # Thư mục output
├── requirements.txt            # Python dependencies
└── README.md                   # Tài liệu này
```

---

## 🏗️ Kiến trúc hệ thống

### Pipeline xử lý

```
Ảnh CCCD gốc
    ↓
[Aligner] → Căn chỉnh và cắt vùng CCCD
    ↓
[Image Cleaner] → Làm sạch ảnh (tăng contrast, denoise)
    ↓
[PaddleOCR Detector] → Phát hiện các vùng văn bản
    ↓
[VietOCR Recognizer] → Nhận dạng ký tự từng vùng
    ↓
Văn bản OCR thô
    ↓
[Spelling Correction] → Sửa lỗi chính tả
    ↓
[HuggingFace/Ollama Extractor] → Trích xuất thông tin có cấu trúc
    ↓
Kết quả JSON (fields: id, name, dob, sex, ...)
```

### Các thành phần chính

1. **OCR Pipeline** (`ocr/pipeline.py`):
   - Kết hợp detector, recognizer, aligner, cleaner
   - Xử lý tuần tự từ ảnh gốc đến văn bản OCR

2. **Preprocessing**:
   - **Aligner**: Sử dụng YOLO model để phát hiện và cắt vùng CCCD
   - **Image Cleaner**: Cải thiện chất lượng ảnh trước khi OCR

3. **Postprocessing**:
   - **Spelling Correction**: Sửa lỗi OCR phổ biến
   - **Field Extractor**: Trích xuất thông tin có cấu trúc từ văn bản

---

## 🎓 Huấn luyện mô hình

### Huấn luyện NER Model

Dự án hỗ trợ huấn luyện mô hình NER để trích xuất thông tin từ văn bản OCR:

```bash
cd train_test
python train.py
```

### Dataset format

Dataset được lưu dưới dạng JSON với format:
```json
{
  "text": "Số: 001234567890 Họ và tên: NGUYEN VAN A ...",
  "entities": [
    {"start": 5, "end": 17, "label": "ID"},
    {"start": 30, "end": 40, "label": "NAME"},
    ...
  ]
}
```

### Đánh giá model

```bash
python test_model.py
```

---

## ⚙️ Cấu hình

Các tham số cấu hình có thể được chỉnh sửa trong `config/settings.py`:

- **Detector**: Loại detector (hiện tại: PaddleOCR)
- **Recognizer**: Loại recognizer (hiện tại: VietOCR với VGG-Transformer)
- **Device**: GPU/CPU cho từng component
- **Extractor**: Chọn giữa HuggingFace hoặc Ollama

---

## 📝 Ghi chú

- Để sử dụng Ollama extractor, cần cài đặt và chạy Ollama server trước
- Mô hình HuggingFace sẽ tự động tải từ HuggingFace Hub khi chạy lần đầu
- Đảm bảo có đủ GPU memory nếu sử dụng GPU (khuyến nghị ít nhất 4GB VRAM)

---

## 📄 License

MIT License

---

## 👤 Tác giả

Dự án được phát triển bởi DengHwee

---

## 🙏 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng tạo issue hoặc pull request.

---

**⭐ Nếu dự án hữu ích, hãy star repo này!**
