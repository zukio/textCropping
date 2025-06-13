import os
import cv2
import pytesseract
import numpy as np

class TextExtractor:
    def extract_texts(self, file_path):
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"File not found: {file_path}")

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        boxes = pytesseract.image_to_boxes(gray)
        mask = np.zeros((h, w), dtype=np.uint8)

        for b in boxes.splitlines():
            parts = b.split(' ')
            if len(parts) >= 5:
                x1, y1, x2, y2 = map(int, parts[1:5])
                # Tesseract's origin is at bottom-left
                mask[h - y2:h - y1, x1:x2] = 255

        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = mask

        out_path = f"{os.path.splitext(file_path)[0]}_texts.png"
        cv2.imwrite(out_path, rgba)
        return out_path
