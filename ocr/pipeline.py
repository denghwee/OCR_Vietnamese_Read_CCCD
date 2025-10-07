from .detector import PaddleOCR_Detector
from .recognizer import VietOCR_Recognizer
from config.settings import CONFIG

class OCR_Pipeline:
    def __init__(self, det_model="PP-OCRv5_server_det", rec_model="vgg_transformer"):
        self.detector = self.build_detector(CONFIG["detector"])
        self.recognizer = self.build_recognizer(CONFIG["recognizer"])

    def build_detector(self, cfg):
        if cfg["type"] == "paddleocr":
            return PaddleOCR_Detector()
        else:
            raise ValueError(f"Unknown detector type: {cfg['type']}")

    def build_recognizer(self, cfg):
        if cfg["type"] == "vietocr":
            return VietOCR_Recognizer(cfg['config_name'])
        else:
            raise ValueError(f"Unknown recognizer type: {cfg['type']}")
        
    def predict(self, image_path, *args, **kwargs):
        boxes = self.detector.predict(image_path)
        results = self.recognizer.predict(image_path, boxes)
        return results