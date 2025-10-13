from paddleocr import TextDetection
from .base_ocr import BaseOCRModel

class PaddleOCR_Detector(BaseOCRModel):
    def __init__(self):
        self.detector = None
        self.load_model()

    def load_model(self):
        self.detector = TextDetection(model_name="PP-OCRv5_server_det")

    def predict(self, image, *args, **kwargs):
        results = self.detector.predict(image)

        boxes = []
        padding = 4  # số pixel muốn mở rộng

        for res in results:
            if 'dt_polys' in res:
                polys = res['dt_polys']
                for poly in polys:
                    # lấy min-max tọa độ để thành bbox
                    xs = [p[0] for p in poly]
                    ys = [p[1] for p in poly]
                    xmin, xmax = min(xs), max(xs)
                    ymin, ymax = min(ys), max(ys)

                    # thêm padding
                    xmin = max(0, xmin - padding)
                    ymin = max(0, ymin - padding)
                    xmax = xmax + padding
                    ymax = ymax + padding

                    boxes.append((xmin, ymin, xmax, ymax))

        return boxes
