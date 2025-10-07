import cv2
import numpy as np

class ImageCleaner:
    def __init__(self):
        pass

    def enhance(self, image):
        """
        Tăng contrast + sharpen
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # sharpen
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        sharp = cv2.filter2D(enhanced, -1, kernel)
        return sharp

    def binarize(self, image):
        """
        Chuyển sang grayscale + threshold để text rõ hơn
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshed = cv2.threshold(gray, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return threshed

    def clean(self, image):
        """
        Full pipeline cleaning
        """
        enhanced = self.enhance(image)
        result = self.binarize(enhanced)
        return result
