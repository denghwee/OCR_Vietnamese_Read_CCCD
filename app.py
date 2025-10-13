import streamlit as st
import os
import cv2
import numpy as np

from config.settings import CONFIG
from ocr.pipeline import OCR_Pipeline
from extractor.field_extractor import FieldExtractor
from preprocessing.aligner import CardAligner
from preprocessing.image_cleaner import ImageCleaner

# ===============================
# Load các model chỉ 1 lần
# ===============================
@st.cache_resource
def load_model():
    return OCR_Pipeline(), CardAligner(model_path='model/model_crop.pt', img_size=640)

ocr_model, aligner = load_model()

# ===============================
# Hàm chạy pipeline OCR
# ===============================
def run_pipeline(image: np.ndarray):
    # cleaner = ImageCleaner()

    cropped = aligner.align(image)

    # Dùng model đã cache
    ocr_lines = ocr_model.predict(image)
    return ocr_lines, cropped


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
            ocr_lines, cropped = run_pipeline(image)

        st.subheader("Kết quả sau khi deskew và clean")
        if cropped is None or cropped.size == 0:
            st.error("Ảnh bị rỗng, không thể hiển thị! Kiểm tra lại bước crop hoặc load ảnh.")
            st.stop()
        else:
            st.image(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB),
                    caption="Ảnh sau khi crop",
                    width=400)
        
        # st.image(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB),
        #          caption="clean",
        #          width=400)

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
