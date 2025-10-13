import cv2
import numpy as np
from PIL import Image

class ImageCleaner:
    def __init__(self):
        pass

    def clean(self, image):
        """
        Full pipeline: shadow removal + auto contrast + smooth
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        if image is None or image.size == 0:
            raise ValueError("Empty image input to ImageCleaner.clean()")

        # 1️⃣ Enhance image
        cleaned = self.remove_shadow_and_enhance(image)

        # 2️⃣ Convert về RGB cho PaddleOCR
        cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
        return cleaned_rgb


    def remove_shadow_and_enhance(self, image):
        """
        Loại bỏ bóng, tăng tương phản và làm mịn ảnh để OCR ổn định hơn.
        """
        # Nếu ảnh có 3 kênh, chuyển về grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # --- Khử bóng ---
        dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(gray, bg)

        # --- Giảm nhiễu, giữ nét ---
        kernel = np.ones((1, 1), np.uint8)
        diff = cv2.dilate(diff, kernel, iterations=1)
        diff = cv2.erode(diff, kernel, iterations=1)

        # --- Làm mịn biên ---
        diff = cv2.GaussianBlur(diff, (5, 5), 0)

        # --- Tự động cân bằng sáng và tương phản ---
        diff, _, _ = self.automatic_brightness_and_contrast(diff, 1)

        return diff


    def automatic_brightness_and_contrast(self, gray, clip_hist_percent=1):
        """
        Cân bằng sáng và tương phản dựa trên biểu đồ histogram.
        """
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_size = len(hist)

        accumulator = np.cumsum(hist)
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum / 100.0)
        clip_hist_percent /= 2.0

        minimum_gray = np.searchsorted(accumulator, clip_hist_percent)
        maximum_gray = np.searchsorted(accumulator, maximum - clip_hist_percent)

        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha

        auto_result = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        return auto_result, alpha, beta
