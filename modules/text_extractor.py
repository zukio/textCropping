import os
import cv2
import shutil
import pytesseract
import numpy as np


# 優先順位：
# 1. 環境変数で TESSERACT_PATH があればそれを使う
# 2. なければ、PATHから自動検出
# 3. それも失敗したらエラーを投げる

tess_path = os.environ.get("TESSERACT_PATH") or shutil.which("tesseract")

if tess_path is None:
    raise RuntimeError(
        "Tesseractが見つかりません。TESSERACT_PATH 環境変数を指定するか、PATHを通してください。")

pytesseract.pytesseract.tesseract_cmd = tess_path


class TextExtractor:
    def __init__(self, output_dir=None):
        """Create extractor with optional output directory."""
        self.output_dir = output_dir

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

        # 出力先ディレクトリが指定されている場合はそこで保存する
        out_dir = self.output_dir or os.path.dirname(file_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        out_path = os.path.join(out_dir, f"{base_name}_texts.png")
        cv2.imwrite(out_path, rgba)
        return out_path
