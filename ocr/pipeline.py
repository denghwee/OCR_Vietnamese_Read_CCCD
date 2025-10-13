from .detector import PaddleOCR_Detector
from .recognizer import VietOCR_Recognizer
from preprocessing.image_cleaner import ImageCleaner
from preprocessing.aligner import CardAligner
from config.settings import CONFIG

class OCR_Pipeline:
    def __init__(self):
        self.detector = self.build_detector(CONFIG["detector"])
        self.recognizer = self.build_recognizer(CONFIG["recognizer"])
        self.aligner = self.build_aligner(CONFIG["aligner"])
        self.cleaner = self.build_cleaner(CONFIG["cleaner"])

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
        
    def build_aligner(self, cfg):
        if cfg["type"] == "default":
            return CardAligner(model_path='model/model_crop.pt', img_size=640)
        else:
            raise ValueError(f"Unknown aligner type: {cfg['type']}")

    def build_cleaner(self, cfg):
        if cfg["type"] == "default":
            return ImageCleaner()
        else:
            raise ValueError(f"Unknown cleaner type: {cfg['type']}")
        
    def predict(self, image, *args, **kwargs):
        cropped = self.aligner.align(image)
        cleaned = self.cleaner.clean(cropped)
        boxes = self.detector.predict(cleaned)
        results = self.recognizer.predict(cleaned, boxes)
        return results, cropped, cleaned