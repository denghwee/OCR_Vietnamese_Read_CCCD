import streamlit as st
import os
import cv2
import numpy as np

from config.settings import CONFIG
from ocr.pipeline import OCR_Pipeline
from extractor.field_extractor import FieldExtractor
from preprocessing.alinger import ImageAligner
from preprocessing.image_cleaner import ImageCleaner

# ===============================
# Load OCR model chỉ 1 lần
# ===============================
@st.cache_resource
def load_ocr_model():
    return OCR_Pipeline()

ocr_model = load_ocr_model()  # giữ model OCR trong cache

# ===============================
# Hàm chạy pipeline OCR
# ===============================
def run_pipeline(image: np.ndarray):
    aligner = ImageAligner()
    cleaner = ImageCleaner()

    # Nếu cần tiền xử lý thì bật lại
    # aligned = aligner.deskew(image)
    # cleaned = cleaner.clean(aligned)

    # Dùng model đã cache
    ocr_lines = ocr_model.predict(image)
    return ocr_lines


# ===============================
# App Streamlit
# ===============================
def main():
    st.set_page_config(page_title="CCCD OCR", layout="wide")
    st.title("📑 Hệ thống OCR đọc Căn Cước Công Dân")

    uploaded_file = st.file_uploader("Tải ảnh CCCD lên", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Hiển thị ảnh nhỏ lại
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                 caption="Ảnh CCCD gốc",
                 width=400)

        with st.spinner("⏳ Đang xử lý OCR..."):
            ocr_lines = run_pipeline(image)

        st.subheader("📜 Kết quả OCR Raw")
        for line in ocr_lines[::-1]:
            st.write(line)

        extractor = FieldExtractor()
        fields = extractor.extract(ocr_lines[::-1])

        st.subheader("✅ Thông tin trích xuất")
        st.write("**Số CCCD:** ", fields.get("id_number", ""))
        st.write("**Họ và tên:** ", fields.get("name", ""))
        st.write("**Ngày sinh:** ", fields.get("dob", ""))
        st.write("**Giới tính:** ", fields.get("gender", ""))
        st.write("**Quốc tịch:** ", fields.get("nationality", ""))


if __name__ == "__main__":
    main()
