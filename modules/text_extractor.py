import os
import cv2
import shutil
import pytesseract
import numpy as np
import datetime
import logging
import json

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
TESSERACT_AVAILABLE = tess_path is not None

if TESSERACT_AVAILABLE:
    pytesseract.pytesseract.tesseract_cmd = tess_path
else:
    text_logger.warning(
        "Tesseractが見つかりません。EasyOCRへフォールバックします")


class TextExtractor:
    def __init__(self, output_dir=None, crop=False, color_mode="original", mono_color="#000000"):
        """Create extractor with optional output directory and options."""
        self.output_dir = output_dir
        self.crop = crop
        self.color_mode = color_mode
        self.mono_color = mono_color
        self.ocr_engine = "tesseract"
        self.easyocr_reader = None

        # 設定ファイルからOCRエンジンを読み込む
        config_path = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.ocr_engine = config.get('ocr_engine', "tesseract")
            except Exception as e:
                text_logger.warning(f"設定ファイルの読み込みに失敗しました: {e}")
                self.ocr_engine = "tesseract"

        # Tesseractが使えない場合はEasyOCRへフォールバック
        if not TESSERACT_AVAILABLE and self.ocr_engine == "tesseract":
            self.ocr_engine = "easyocr"
            text_logger.info("Tesseractが利用できないためEasyOCRを使用します")

    def _parse_color(self, color):
        """Convert color specification to BGR tuple."""
        if isinstance(color, (list, tuple)) and len(color) == 3:
            return tuple(int(c) for c in reversed(color))
        if isinstance(color, str):
            if color.startswith("#") and len(color) == 7:
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
                return (b, g, r)
            if "," in color:
                parts = color.split(',')
                if len(parts) == 3:
                    r, g, b = [int(p.strip()) for p in parts]
                    return (b, g, r)
        # default black
        return (0, 0, 0)

    def _init_easyocr(self):
        """EasyOCRを初期化する"""
        if self.easyocr_reader is None:
            try:
                import easyocr
                text_logger.info("EasyOCRを初期化しています...")
                self.easyocr_reader = easyocr.Reader(['ja', 'en'])
                text_logger.info("EasyOCRの初期化が完了しました")
                return True
            except ImportError:
                text_logger.error("EasyOCRがインストールされていません。Tesseractを使用します。")
                self.ocr_engine = "tesseract"
                return False
            except Exception as e:
                text_logger.error(f"EasyOCRの初期化中にエラーが発生しました: {e}")
                self.ocr_engine = "tesseract"
                return False
        return True

    def extract_texts_with_easyocr(self, img):
        """EasyOCRを使用してテキストを抽出"""
        if not self._init_easyocr():
            return None, None, 0

        h, w = img.shape[:2]

        try:
            # EasyOCRでテキスト検出（信頼度の閾値を下げる）
            results = self.easyocr_reader.readtext(
                img, detail=1, paragraph=False, min_size=10)

            if not results:
                text_logger.info("EasyOCRでテキストが検出されませんでした")
                return None, None, 0

            texts = []
            boxes_list = []
            char_count = 0

            # 検出されたテキストを処理
            for bbox, text, conf in results:
                if conf > 0.1:  # 信頼度の閾値を下げる（より多くのテキストを検出）
                    if text and text.strip():  # 空のテキストをスキップ
                        texts.append(text)
                        char_count += len(text)

                        try:
                            # ボックス情報をTesseractと互換性のある形式に変換
                            # 座標が空や不正でないことを確認
                            if len(bbox) != 4 or any(not isinstance(point, (list, tuple)) or len(point) != 2 for point in bbox):
                                text_logger.warning(f"無効なbbox形式: {bbox}")
                                continue

                            x1, y1 = int(float(bbox[0][0])), int(
                                float(bbox[0][1]))
                            x2, y2 = int(float(bbox[2][0])), int(
                                float(bbox[2][1]))

                            # 座標値が有効な範囲内かチェック
                            if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                                # 各文字に対して行を作成
                                for i, char in enumerate(text):
                                    if not char or char.isspace():  # 空文字や空白はスキップ
                                        continue

                                    # 0除算を防ぐ
                                    text_len = max(len(text.strip()), 1)
                                    char_width = max((x2 - x1) / text_len, 1)

                                    char_x1 = max(int(x1 + i * char_width), 0)
                                    char_x2 = min(
                                        int(x1 + (i + 1) * char_width), w-1)

                                    # 有効なボックスのみ追加
                                    if char_x1 < char_x2:
                                        boxes_list.append(
                                            f"{char} {char_x1} {h-y2} {char_x2} {h-y1}")
                        except (ValueError, TypeError, IndexError, ZeroDivisionError) as e:
                            text_logger.warning(
                                f"ボックス座標の処理中にエラー: {e}, bbox={bbox}, text={text}")
                            continue

            detected_text = "\n".join(texts)
            boxes = "\n".join(boxes_list)

            text_logger.info(
                f"EasyOCRで検出されたテキスト数: {len(texts)}, 文字数: {char_count}")

            if not boxes_list:
                text_logger.warning("有効なボックス情報が生成されませんでした")
                return None, None, 0

            return detected_text, boxes, char_count

        except Exception as e:
            text_logger.error(f"EasyOCRでのテキスト検出中にエラーが発生しました: {e}")
            import traceback
            text_logger.error(f"詳細なエラー: {traceback.format_exc()}")
            return None, None, 0

    def extract_texts_with_saas(self, img):
        """SaaS型OCRを使用してテキストを抽出 (未実装)"""
        text_logger.warning("SaaS OCR は未実装です")
        return None, None, 0

    def extract_texts(self, file_path):
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"File not found: {file_path}")

        h, w = img.shape[:2]

        # 画像の前処理を実行
        text_logger.info(f"ファイル: {os.path.basename(file_path)} の前処理を開始します")
        enhanced_img, binary_img = self.preprocess_image(img)

        # 前処理結果の保存（デバッグ用）
        if self.output_dir:
            debug_dir = os.path.join(self.output_dir, 'debug')
        else:
            parent_dir = os.path.dirname(os.path.dirname(file_path))
            debug_dir = os.path.join(parent_dir, 'output', 'debug')

        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        cv2.imwrite(os.path.join(
            debug_dir, f"{base_name}_enhanced.jpg"), enhanced_img)
        cv2.imwrite(os.path.join(
            debug_dir, f"{base_name}_binary.jpg"), binary_img)

        # 設定に基づいてOCRエンジンを選択
        if self.ocr_engine == "easyocr":
            # まずTesseractで文字領域を検出（可能なら）
            pre_img = enhanced_img  # 前処理済み画像を使用
            tess_boxes = ""
            if TESSERACT_AVAILABLE:
                # 二値化画像を使用してテキスト領域検出
                tess_boxes = pytesseract.image_to_boxes(binary_img)
                pre_mask = np.zeros((h, w), dtype=np.uint8)
                for b in tess_boxes.splitlines():
                    parts = b.split(' ')
                    if len(parts) >= 5:
                        x1, y1, x2, y2 = map(int, parts[1:5])
                        pre_mask[h - y2:h - y1, x1:x2] = 255
                pre_img = enhanced_img.copy()
                pre_img[pre_mask == 0] = (255, 255, 255)

            detected_text, boxes, char_count = self.extract_texts_with_easyocr(
                pre_img)

            # EasyOCRが失敗した場合はTesseractにフォールバック
            if detected_text is None or boxes is None:
                text_logger.info("EasyOCRでの検出に失敗しました。Tesseractを使用します。")
                detected_text = pytesseract.image_to_string(
                    binary_img, lang='jpn+eng')
                boxes = pytesseract.image_to_boxes(binary_img)
                char_count = sum(1 for b in boxes.splitlines()
                                 if len(b.split(' ')) >= 5)

        elif self.ocr_engine == "saas":
            detected_text, boxes, char_count = self.extract_texts_with_saas(
                enhanced_img)
            if boxes is None:
                return None

        else:
            # デフォルト: Tesseractを使用
            detected_text = pytesseract.image_to_string(
                binary_img, lang='jpn+eng')
            boxes = pytesseract.image_to_boxes(binary_img)
            char_count = sum(1 for b in boxes.splitlines()
                             if len(b.split(' ')) >= 5)

        # マスク画像(文字領域)を作成
        mask = np.zeros((h, w), dtype=np.uint8)

        for b in boxes.splitlines():
            parts = b.split(' ')
            if len(parts) >= 5:
                x1, y1, x2, y2 = map(int, parts[1:5])
                # Tesseract の座標系は左下が原点
                mask[h - y2:h - y1, x1:x2] = 255

        # 文字検出結果をログに記録
        base_name = os.path.basename(file_path)
        if char_count > 0:
            msg = f"ファイル: {base_name} - 文字検出: {char_count}文字"
            text_logger.info(msg)
            logging.info(msg)
            if detected_text and detected_text.strip():
                text = detected_text.strip()
                # 検出されたテキストを改行ごとに分割してログに記録
                text_logger.info(f"検出テキスト: \n{text}")
                logging.info(f"検出テキスト: \n{text}")
            else:
                msg = "テキスト認識できませんでした（ボックスのみ検出）"
                text_logger.info(msg)
                logging.info(msg)
                return None
        else:
            msg = f"ファイル: {base_name} - 文字検出なし"
            text_logger.info(msg)
            logging.info(msg)
            return None

        # ----- 文字領域以外を白で塗りつぶす -----
        filled = img.copy()

        filled[mask == 0] = (255, 255, 255)

        # ----- 二値化してテキストだけのマスクを作成 -----
        gray_mask = cv2.cvtColor(filled, cv2.COLOR_BGR2GRAY)

        otsu_thresh, _ = cv2.threshold(gray_mask, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Otsuで求めた値より少し高い閾値を設定して非文字部分を除去
        strict_thresh = min(255, otsu_thresh)

        _, binary = cv2.threshold(gray_mask, strict_thresh, 255,
                                  cv2.THRESH_BINARY)
        text_mask = cv2.bitwise_not(binary)

        # 小さなノイズを除去
        kernel = np.ones((3, 3), np.uint8)
        text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, kernel)

        # ----- マスクを用いて元画像から文字部分のみ抽出 -----
        if self.color_mode == "original":
            masked = cv2.bitwise_and(img, img, mask=text_mask)
        else:
            color = self._parse_color(self.mono_color)
            masked = np.zeros_like(img)
            masked[text_mask > 0] = color

        # 透過PNG用にAlphaチャネルを設定
        rgba = cv2.cvtColor(masked, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = text_mask

        if self.crop:
            coords = cv2.findNonZero(text_mask)
            if coords is not None:
                x, y, w2, h2 = cv2.boundingRect(coords)
                rgba = rgba[y:y+h2, x:x+w2]
        # 出力先ディレクトリの処理
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

    def remove_noise(self, image):
        """
        ノイズ除去処理を行います。
        ガウシアンフィルタとメディアンフィルタを適用してノイズを低減します。

        Args:
            image: 入力画像 (BGR形式)

        Returns:
            ノイズ除去された画像
        """
        text_logger.info("ノイズ除去処理を適用しています...")

        # ガウシアンフィルタでノイズを低減
        blur = cv2.GaussianBlur(image, (3, 3), 0)

        # メディアンフィルタで塩コショウノイズを除去
        median = cv2.medianBlur(blur, 3)

        return median

    def adjust_contrast(self, image):
        """
        コントラスト調整処理を行います。
        CLAHE (Contrast Limited Adaptive Histogram Equalization) を使用します。

        Args:
            image: 入力画像

        Returns:
            コントラスト調整された画像
        """
        text_logger.info("コントラスト調整処理を適用しています...")

        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # CLAHEでコントラスト調整
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 元が3チャンネルの場合は3チャンネルに戻す
        if len(image.shape) == 3:
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            return enhanced_bgr

        return enhanced

    def adaptive_threshold(self, image):
        """
        適応的二値化処理を行います。
        局所的な領域ごとに最適な閾値を算出して二値化します。

        Args:
            image: 入力画像

        Returns:
            二値化された画像
        """
        text_logger.info("適応的二値化処理を適用しています...")

        # グレースケールに変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 適応的二値化
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,  # ブロックサイズ
            2    # 定数C
        )

        # 形態学的処理でノイズ除去と文字の連結
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return binary

    def preprocess_image(self, image):
        """
        テキスト抽出のための画像前処理を行います。
        ノイズ除去、コントラスト調整、適応的二値化を順に適用します。

        Args:
            image: 入力画像 (BGR形式)

        Returns:
            前処理済みの画像と二値化画像のタプル
        """
        text_logger.info("画像の前処理を開始します...")

        # ノイズ除去
        denoised = self.remove_noise(image)

        # コントラスト調整
        enhanced = self.adjust_contrast(denoised)

        # 適応的二値化
        binary = self.adaptive_threshold(enhanced)

        text_logger.info("画像の前処理が完了しました")

        return enhanced, binary
