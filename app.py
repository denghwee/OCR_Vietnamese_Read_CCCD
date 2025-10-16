import streamlit as st
import os
import cv2
import numpy as np
from ocr.pipeline import OCR_Pipeline
from extractor.field_extractor import FieldExtractor

# ===============================
# Load model OCR chỉ 1 lần
# ===============================
@st.cache_resource
def load_model():
    return OCR_Pipeline()

ocr_model = load_model()

# ===============================
# Hàm chạy pipeline OCR
# ===============================
def run_pipeline(image: np.ndarray):
    ocr_lines, cropped, cleaned = ocr_model.predict(image)
    return ocr_lines, cropped, cleaned

# ===============================
# Streamlit App
# ===============================
def main():
    st.set_page_config(page_title="CCCD OCR", layout="wide")
    st.title("🪪 Hệ thống đọc & trích xuất thông tin Căn Cước Công Dân")
    
    uploaded_file = st.file_uploader("📸 Tải ảnh CCCD lên", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Hiển thị ảnh gốc
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Ảnh CCCD gốc", width=500)

        # OCR Pipeline
        with st.spinner("⏳ Đang xử lý OCR..."):
            ocr_lines, cropped, cleaned = run_pipeline(image)

        st.subheader("Kết quả sau xử lý ảnh")
        col1, col2 = st.columns(2)
        with col1:
            if cropped is not None and cropped.size > 0:
                st.image(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB),
                         caption="Ảnh sau khi crop", width=500)
        with col2:
            if cleaned is not None and cleaned.size > 0:
                st.image(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB),
                         caption="Ảnh sau khi clean", width=500)

        # Hiển thị OCR raw
        if ocr_lines:
            st.subheader("📜 Kết quả OCR (Raw)")
            for line in ocr_lines[::-1]:
                st.write(line)

            # Ghép các dòng OCR lại
            ocr_text = " ".join(ocr_lines[::-1])

            # Trích xuất thông tin từ raw text OCR
            extractor = FieldExtractor()
            
            with st.spinner("🧠 Đang hậu kỳ OCR và trích xuất..."):
                fields = extractor.extract(ocr_text)

            st.success("✅ Hoàn tất trích xuất thông tin!")

            st.subheader("🔎 Thông tin trích xuất:")
            st.json(fields)

            # Hiển thị dạng đẹp
            st.write("**Số CCCD:** ", fields.get("id", ""))
            st.write("**Họ và tên:** ", fields.get("name", ""))
            st.write("**Ngày sinh:** ", fields.get("dob", ""))
            st.write("**Giới tính:** ", fields.get("sex", ""))
            st.write("**Quốc tịch:** ", fields.get("nationality", ""))
            st.write("**Quê quán:** ", fields.get("placeoforigin", ""))
            st.write("**Địa chỉ/Nơi ở:** ", fields.get("placeofresidence", ""))

        else:
            st.text_area("Văn bản OCR thuần", ocr_text, height=200)

if __name__ == "__main__":
    main()
