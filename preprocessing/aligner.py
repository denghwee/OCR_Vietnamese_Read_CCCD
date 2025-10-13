import cv2
import numpy as np
from ultralytics import YOLO

class CardAligner:
    def __init__(self, model_path='model/model_crop.pt', img_size=640, device='cuda', debug=False):
        self.model = YOLO(model_path)
        self.img_size = img_size
        self.device = device
        self.debug = debug
        self.corner_names = ['top_left', 'top_right', 'bot_right', 'bot_left']
    
    def align(self, image):
            if isinstance(image, str):
                image = cv2.imread(image)
            if image is None or image.size == 0:
                print("[ERROR] Ảnh đầu vào rỗng hoặc không tồn tại.")
                return None

            corner_points = self.detect_corners(image)
            if not corner_points:
                print("[ERROR] Không detect được 4 góc.")
                return None
            elif len(corner_points) == 3:
                corner_points = self.estimate_missing_corner(corner_points)

            if not self.validate_corners(corner_points):
                return None

            pts_src = self.order_points(corner_points)
            warped = self.warp_perspective(image, pts_src)

            if warped is None:
                print("[ERROR] Ảnh bị rỗng, không thể hiển thị! Kiểm tra lại bước crop hoặc load ảnh.")
                return None

            print(f"[INFO] Deskew thành công. Kích thước: {warped.shape}")
            return warped
    
    def detect_corners(self, image):
        results = self.model.predict(source=image, imgsz=self.img_size, device=self.device, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            print("[WARN] Không phát hiện được góc nào.")
            return {}

        corner_points = {}
        for corner in self.corner_names:
            indices = [i for i, c in enumerate(boxes.cls.cpu().numpy()) if self.model.names[int(c)] == corner]
            if len(indices) == 0:
                print(f"[WARN] Không tìm thấy góc {corner}.")
                continue

            best_idx = max(indices, key=lambda i: boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            corner_points[corner] = (cx, cy)

        if self.debug:
            for name, (x, y) in corner_points.items():
                cv2.circle(image, (x, y), 6, (0, 255, 0), -1)
                cv2.putText(image, name, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Detected corners", image)
            cv2.waitKey(0)

        return corner_points

    def validate_corners(self, corner_points):
        missing = [c for c in self.corner_names if c not in corner_points]
        if missing:
            print(f"[ERROR] Thiếu góc: {missing}")
            return False
        return True

    def order_points(self, corner_points):
        return np.float32([
            corner_points['top_left'],
            corner_points['top_right'],
            corner_points['bot_right'],
            corner_points['bot_left']
        ])

    def warp_perspective(self, image, pts_src):
        width = int(max(
            np.linalg.norm(pts_src[0] - pts_src[1]),
            np.linalg.norm(pts_src[2] - pts_src[3])
        ))
        height = int(max(
            np.linalg.norm(pts_src[0] - pts_src[3]),
            np.linalg.norm(pts_src[1] - pts_src[2])
        ))

        if width <= 5 or height <= 5:
            print(f"[ERROR] Kích thước warp không hợp lệ: width={width}, height={height}")
            return None

        pts_dst = np.float32([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ])

        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped = cv2.warpPerspective(image, M, (width, height))

        if warped is None or warped.size == 0:
            print("[ERROR] cv2.warpPerspective trả về ảnh rỗng.")
            return None
        return warped

    def estimate_missing_corner(self, corners):
        """
        corners: dict với key là tên góc ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        trả về dict đã có đủ 4 góc (nếu có thể nội suy được)
        """
        missing = [name for name in ['top_left', 'top_right', 'bottom_right', 'bottom_left'] if name not in corners]
        if len(missing) != 1:
            return corners  # chỉ xử lý nếu thiếu đúng 1 góc

        m = missing[0]
        c = corners

        try:
            if m == 'top_left':
                c[m] = tuple(np.add(c['top_right'], c['bottom_left']) - c['bottom_right'])
            elif m == 'top_right':
                c[m] = tuple(np.add(c['top_left'], c['bottom_right']) - c['bottom_left'])
            elif m == 'bottom_left':
                c[m] = tuple(np.add(c['bottom_right'], c['top_left']) - c['top_right'])
            elif m == 'bottom_right':
                c[m] = tuple(np.add(c['bottom_left'], c['top_right']) - c['top_left'])
        except KeyError:
            # nếu thiếu quá nhiều góc → không thể nội suy
            return corners

        print(f"[INFO] Ước lượng góc {m}: {c[m]}")
        return c