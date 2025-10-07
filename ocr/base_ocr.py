from abc import ABC, abstractmethod

class BaseOCRModel(ABC):
    """
    Lớp trừu tượng cho các config OCR model khác nhau (Detector / Recognizer)
    """
    @abstractmethod
    def load_model(self):
        """Khởi tạo hoặc load model"""
        pass

    def predict(self, image_path, *args, **kwargs):
        """Nhận đầu vào là ảnh, đầu ra trả về kết quả"""
        pass