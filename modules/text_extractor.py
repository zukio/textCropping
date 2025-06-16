import os
import cv2
import shutil
import pytesseract
import numpy as np
import datetime
import logging

# ログの設定


def setup_text_logging():
    """文字検出用のログ設定を行います"""
    log_dir = os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    text_logger = logging.getLogger('text_detection')
    text_logger.setLevel(logging.INFO)

    # 日付ごとに新しいログファイルを作成
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    log_file = os.path.join(log_dir, f'text_detection_{today}.log')

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # ハンドラが既に存在する場合は追加しない
    if not text_logger.handlers:
        text_logger.addHandler(file_handler)

    return text_logger


# テキスト検出用のロガーを初期化
text_logger = setup_text_logging()

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

        # 文字をテキストとして検出（ロギング用）
        detected_text = pytesseract.image_to_string(gray, lang='jpn+eng')

        # 文字のバウンディングボックスを取得
        boxes = pytesseract.image_to_boxes(gray)
        mask = np.zeros((h, w), dtype=np.uint8)

        # 検出された文字の数をカウント
        char_count = 0

        for b in boxes.splitlines():
            parts = b.split(' ')
            if len(parts) >= 5:
                char_count += 1
                x1, y1, x2, y2 = map(int, parts[1:5])
                # Tesseract's origin is at bottom-left
                mask[h - y2:h - y1, x1:x2] = 255

        # 文字検出結果をログに記録
        base_name = os.path.basename(file_path)
        if char_count > 0:
            text_logger.info(f"ファイル: {base_name} - 文字検出: {char_count}文字")
            if detected_text.strip():
                # 検出されたテキストを改行ごとに分割してログに記録
                text_logger.info(f"検出テキスト: \n{detected_text.strip()}")
            else:
                text_logger.info("テキスト認識できませんでした（ボックスのみ検出）")
        else:
            text_logger.info(f"ファイル: {base_name} - 文字検出なし")

        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = mask        # 出力先ディレクトリの処理
        if self.output_dir:
            # 出力先が指定されている場合はそこを使用
            out_dir = self.output_dir
        else:
            # 指定がない場合は入力ファイルの親ディレクトリの'output'フォルダを使用
            parent_dir = os.path.dirname(os.path.dirname(file_path))
            out_dir = os.path.join(parent_dir, 'output')

        # 出力先ディレクトリが監視対象ディレクトリ内にないかチェック
        file_dir = os.path.dirname(file_path)
        from modules.utils.path_utils import is_subpath
        if is_subpath(out_dir, file_dir):
            text_logger.warning(f"出力先ディレクトリが監視対象内です: {out_dir}")
            # 安全な代替として親ディレクトリの'output'フォルダを使用
            out_dir = os.path.join(os.path.dirname(file_dir), 'output')
            text_logger.info(f"出力先を変更しました: {out_dir}")

        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        out_path = os.path.join(out_dir, f"{base_name}_texts.png")
        cv2.imwrite(out_path, rgba)
        return out_path
