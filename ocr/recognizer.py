from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import cv2
import numpy as np

from .base_ocr import BaseOCRModel


class VietOCR_Recognizer(BaseOCRModel):
    def __init__(self, model_config='vgg_transformer'):
        self.model_config = model_config
        self.config = None
        self.recognizer = None
        self.load_model()

    def load_model(self):
        self.config = Cfg.load_config_from_name(self.model_config)
        self.config['cnn']['pretrained'] = True
        self.config['predictor']['beamsearch'] = True
        self.config['device'] = 'cuda:0'
        self.recognizer = Predictor(self.config)

    def load_image(self, image_input):
        if isinstance(image_input, str):  # path
            return Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):  # numpy
            if len(image_input.shape) == 2:  # grayscale
                image_input = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            else:
                image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image_input)
        else:
            raise ValueError("Unsupported input type for recognizer: must be path or numpy array")

    def predict(self, image, boxes=None, *args, **kwargs):
        """
        Nhận dạng text trong ảnh.
        Nếu có boxes: crop từng vùng chữ và nhận dạng.
        Nếu không: nhận dạng cả ảnh.
        """
        img = self.load_image(image)
        # img = Image.open(image_path).convert("RGB")
        
        results = []

        if boxes:
            for box in boxes:
                # lấy min/max từ polygon 4 điểm
                xmin, ymin, xmax, ymax = box
                crop = img.crop((xmin, ymin, xmax, ymax))

                text = self.recognizer.predict(crop)
                results.append(text)
        else:
            text = self.recognizer.predict(img)
            results.append(text)

        return results