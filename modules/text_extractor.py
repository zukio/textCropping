import os
import cv2
import shutil
import pytesseract
import numpy as np
import datetime
import logging
import json
import subprocess
import tempfile
from PIL import Image

# ログの設定

# テキスト検出結果専用のカスタムログレベル
DETECTED_TEXT = 25  # INFOとWARNINGの間のレベル
logging.addLevelName(DETECTED_TEXT, 'DETECTED_TEXT')


def setup_text_logging():
    """文字検出用のログ設定を行います"""
    log_dir = os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # メイン処理用ロガー（通常のログファイルに出力）
    text_logger = logging.getLogger('text_detection')
    text_logger.setLevel(logging.INFO)

    # 検出テキスト専用ロガー（検出テキストのみを記録）
    detected_text_logger = logging.getLogger('detected_text')
    detected_text_logger.setLevel(DETECTED_TEXT)

    # 日付ごとに新しいログファイルを作成
    today = datetime.datetime.now().strftime('%Y-%m-%d')

    # 通常のログファイル（処理詳細用）
    process_log_file = os.path.join(log_dir, f'text_process_{today}.log')
    # テキスト検出結果専用ログファイル
    text_log_file = os.path.join(log_dir, f'text_detection_{today}.log')

    # 処理詳細用ハンドラ
    process_handler = logging.FileHandler(process_log_file, encoding='utf-8')
    process_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    process_handler.setFormatter(process_formatter)
    process_handler.setLevel(logging.INFO)

    # 検出テキスト専用ハンドラ
    text_handler = logging.FileHandler(text_log_file, encoding='utf-8')
    text_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    text_handler.setFormatter(text_formatter)
    text_handler.setLevel(DETECTED_TEXT)

    # ハンドラが既に存在する場合は追加しない
    if not text_logger.handlers:
        text_logger.addHandler(process_handler)

    if not detected_text_logger.handlers:
        detected_text_logger.addHandler(text_handler)

    return text_logger, detected_text_logger


# テキスト検出用のロガーを初期化
text_logger, detected_text_logger = setup_text_logging()

# 検出テキスト専用ログメソッドを追加


def log_detected_text(text):
    """検出されたテキストを専用ログに記録する"""
    detected_text_logger.log(DETECTED_TEXT, f"検出テキスト: \n{text}")

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
    def __init__(self, output_dir=None, crop=False, color_mode="original",
                 mono_color="#000000", ocr_engine=None, gcp_credentials=None, debug_output=False,
                 enable_svg=True):
        """Create extractor with optional output directory and options.

        Parameters
        ----------
        ocr_engine : str, optional
            Specify OCR engine to use.
        gcp_credentials : str, optional
            Path to Google Cloud credentials JSON.
        debug_output : bool, optional
            Whether to save debug images to debug directory. Defaults to True.
        """
        self.output_dir = output_dir
        self.crop = crop
        self.color_mode = color_mode
        self.mono_color = mono_color
        self.ocr_engine = ocr_engine
        self.easyocr_reader = None
        self.gcp_credentials = gcp_credentials
        self.debug_output = debug_output
        self.enable_svg = enable_svg

        # OCRエンジンの正規化
        if isinstance(self.ocr_engine, str):
            self.ocr_engine = self.ocr_engine.lower()

        # 引数が指定されていない場合はデフォルトのOCRエンジンを使用
        if self.ocr_engine is None:
            self.ocr_engine = "tesseract"

        # Tesseractが使えない場合はEasyOCRへフォールバック
        if not TESSERACT_AVAILABLE and self.ocr_engine == "tesseract":
            self.ocr_engine = "easyocr"
            text_logger.info("Tesseractが利用できないためEasyOCRを使用します")

    def _save_svg(self, mask, svg_path):
        """Save mask as SVG using potrace."""
        try:
            potrace_path = os.environ.get("POTRACE_PATH") or shutil.which("potrace")
            if not potrace_path:
                text_logger.error("potraceが見つかりません。'POTRACE_PATH'を設定するか、PATHに追加してください")
                return
            temp_path = None
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pbm") as tmp:
                Image.fromarray(mask).convert("1").save(tmp.name)
                temp_path = tmp.name

            subprocess.run([potrace_path, temp_path, "-s", "-o", svg_path], check=True)
            text_logger.info(f"SVGを保存しました: {svg_path}")
        except Exception as e:
            text_logger.error(f"SVG保存中にエラーが発生しました: {e}")
        finally:
            if temp_path:
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

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
                self.easyocr_reader = easyocr.Reader(['ja', 'en'], gpu=False)
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
        """Use Google Cloud Vision API to extract texts."""
        try:
            if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") and self.gcp_credentials:
                if os.path.isfile(self.gcp_credentials):
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.gcp_credentials
            from modules.ocr_saas_gcp_visionai import extract_text
            detected_text, boxes, char_count = extract_text(img)
            return detected_text, boxes, char_count
        except Exception as e:
            text_logger.error(f"SaaS OCR error: {e}")
            return None, None, 0

    def extract_texts(self, file_path):
        # 日本語ファイル名対応のため、NumPyを使用して画像を読み込む
        try:
            # ファイルパスをUTF-8でエンコードして確実に日本語ファイル名を処理できるようにする
            text_logger.info(f"画像読み込み開始: {file_path}")

            # バイナリモードで読み込み、NumPyとimdecodを使用
            import numpy as np
            with open(file_path, 'rb') as f:
                img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                text_logger.error(f"画像の読み込みに失敗しました: {file_path}")
                # ファイルパスが正しく表示されるか確認するためにバイナリ表現も出力
                text_logger.error(f"バイナリファイルパス: {file_path.encode('utf-8')}")
                raise FileNotFoundError(
                    f"File not found or could not be read: {file_path}")

            text_logger.info(
                f"画像読み込み成功: {os.path.basename(file_path)}, サイズ: {img.shape}")
        except Exception as e:
            text_logger.error(f"画像読み込み中にエラーが発生しました: {e}, path: {file_path}")
            text_logger.error(f"バイナリファイルパス: {file_path.encode('utf-8')}")
            raise FileNotFoundError(
                f"Error reading file: {file_path}, error: {e}")

        h, w = img.shape[:2]

        # 出力ディレクトリの設定
        if self.output_dir:
            debug_dir = os.path.join(self.output_dir, 'debug')
            out_dir = self.output_dir
        else:
            parent_dir = os.path.dirname(os.path.dirname(file_path))
            debug_dir = os.path.join(parent_dir, 'output', 'debug')
            out_dir = os.path.join(parent_dir, 'output')

        # 出力先ディレクトリが監視対象ディレクトリ内にないかチェック
        file_dir = os.path.dirname(file_path)
        from modules.utils.path_utils import is_subpath
        if is_subpath(out_dir, file_dir):
            text_logger.warning(f"出力先ディレクトリが監視対象内です: {out_dir}")
            # 安全な代替として親ディレクトリの'output'フォルダを使用
            out_dir = os.path.join(os.path.dirname(file_dir), 'output')
            text_logger.info(f"出力先を変更しました: {out_dir}")        # ディレクトリ作成
        if self.debug_output and not os.path.exists(debug_dir):
            os.makedirs(debug_dir, exist_ok=True)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(file_path))[0]

        # テキスト検出
        text_logger.info(f"ファイル: {os.path.basename(file_path)} のテキスト検出を開始します")

        # 設定に基づいてOCRエンジンを選択
        if str(self.ocr_engine).lower() == "none":
            text_logger.info("OCR処理をスキップします")
            detected_text, boxes, char_count = "", "", 0
        elif self.ocr_engine == "easyocr":
            detected_text, boxes, char_count = self.extract_texts_with_easyocr(
                img)
            if detected_text is None or boxes is None:
                text_logger.info("EasyOCRでの検出に失敗しました。Tesseractを使用します。")
                if TESSERACT_AVAILABLE:
                    detected_text = pytesseract.image_to_string(
                        img, lang='jpn+eng')
                    boxes = pytesseract.image_to_boxes(img)
                    char_count = sum(1 for b in boxes.splitlines()
                                     if len(b.split(' ')) >= 5)
                else:
                    text_logger.error("有効なOCRエンジンがありません")
                    return None
        elif self.ocr_engine == "saas":
            detected_text, boxes, char_count = self.extract_texts_with_saas(
                img)
            if boxes is None:
                return None
        else:
            # デフォルト: Tesseractを使用
            if TESSERACT_AVAILABLE:
                detected_text = pytesseract.image_to_string(
                    img, lang='jpn+eng')
                boxes = pytesseract.image_to_boxes(img)
                char_count = sum(1 for b in boxes.splitlines()
                                 if len(b.split(' ')) >= 5)
            else:
                text_logger.error("Tesseractが利用できません")
                return None        # 文字検出結果をログに記録
        base_name = os.path.basename(file_path)
        if str(self.ocr_engine).lower() == "none":
            msg = f"ファイル: {base_name} - OCRスキップモード"
            text_logger.info(msg)
            logging.info(msg)
        elif char_count > 0:
            msg = f"ファイル: {base_name} - 文字検出: {char_count}文字"
            text_logger.info(msg)
            logging.info(msg)
            if detected_text and detected_text.strip():
                text = detected_text.strip()
                # 検出されたテキストを専用ログに記録
                log_detected_text(text)
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

        # テキストボックスを作成
        if str(self.ocr_engine).lower() == "none":
            # OCRスキップ時は画像全体をマスク対象とする
            text_mask = np.ones((h, w), dtype=np.uint8) * 255
        else:
            text_mask = np.zeros((h, w), dtype=np.uint8)

            # 各文字ボックスを拡大して連結
            text_logger.info("文字ボックスを拡大して連結します")
            for b in boxes.splitlines():
                parts = b.split(' ')
                if len(parts) >= 5:
                    x1, y1, x2, y2 = map(int, parts[1:5])
                    # Tesseract の座標系は左下が原点なので変換
                    y1, y2 = h - y2, h - y1

                    # ボックスを指定倍率で拡大（中心から）
                    width = x2 - x1
                    height = y2 - y1
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    box_scale = 1.2  # ボックス拡大率
                    new_width = int(round(width * box_scale))
                    new_height = int(round(height * box_scale))

                    new_x1 = max(0, cx - new_width // 2)
                    new_y1 = max(0, cy - new_height // 2)
                    new_x2 = min(w, cx + new_width // 2)
                    new_y2 = min(h, cy + new_height // 2)

                    # 拡大したボックスを描画（連結のため）
                    # 連結処理を強化するためのモルフォロジー演算
                    text_mask[new_y1:new_y2, new_x1:new_x2] = 255
            kernel = np.ones((5, 5), np.uint8)
            text_mask = cv2.dilate(text_mask, kernel, iterations=1)
            text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel)

        # デバッグ用にボックス画像を保存
        if self.debug_output:
            boxes_path = os.path.join(debug_dir, f"{base_name}_boxes.jpg")
            try:
                success, buffer = cv2.imencode('.jpg', text_mask)
                if success:
                    with open(boxes_path, 'wb') as f:
                        f.write(buffer)
                    text_logger.info(f"ボックス画像を保存しました: {boxes_path}")
            except Exception as e:
                # ボックス範囲内の画像を抽出
                text_logger.error(f"ボックス画像の保存中にエラーが発生しました: {e}")
        masked_original = img.copy()

        # 画像の前処理を実行（ボックス内のみ）
        text_logger.info(f"ファイル: {os.path.basename(file_path)} の前処理を開始します")
        enhanced_img, binary_img = self.preprocess_image(masked_original)

        # 前処理結果の保存（デバッグ用）
        if self.debug_output:
            try:
                # 前処理画像の保存
                enhanced_success, enhanced_buffer = cv2.imencode(
                    '.jpg', enhanced_img)
                if enhanced_success:
                    with open(os.path.join(debug_dir, f"{base_name}_enhanced.jpg"), 'wb') as f:
                        f.write(enhanced_buffer)

                # 二値化画像の保存
                binary_success, binary_buffer = cv2.imencode(
                    '.jpg', binary_img)
                if binary_success:
                    with open(os.path.join(debug_dir, f"{base_name}_binary.jpg"), 'wb') as f:
                        f.write(binary_buffer)

                text_logger.info(f"前処理済み画像をデバッグディレクトリに保存しました: {debug_dir}")
            except Exception as e:
                text_logger.error(f"デバッグ画像の保存中にエラーが発生しました: {e}")
                # 保存に失敗しても処理は続行

        # 二値化画像の平均値をチェックして文字領域の白黒を判断
        binary_mean = np.mean(binary_img)
        text_logger.info(f"二値化画像の平均値: {binary_mean}")

        if binary_mean > 120:
            # 文字が白の場合は反転
            final_mask = cv2.bitwise_not(binary_img)
            text_logger.info("二値化画像の文字は白です（反転して使用）")
        else:
            # 文字が黒の場合はそのまま使用
            final_mask = binary_img.copy()
            text_logger.info("二値化画像の文字は黒です（そのまま使用）")

        # 前処理した文字部分の細部を保持するため、ボックス領域内でのみマスクを適用
        refined_mask = np.zeros((h, w), dtype=np.uint8)
        refined_mask = cv2.bitwise_and(final_mask, text_mask)

        # マスクの微調整（小さなノイズを除去、文字の連結を強化）
        # morph_kernel_size = 2
        # kernel = np.ones(
        #    (morph_kernel_size, morph_kernel_size), np.uint8)
        # refined_mask = cv2.morphologyEx(
        #    refined_mask, cv2.MORPH_OPEN, kernel)
        # refined_mask = cv2.morphologyEx(
        #    refined_mask, cv2.MORPH_CLOSE, kernel)

        # デバッグ用に最終マスクを保存
        if self.debug_output:
            mask_path = os.path.join(debug_dir, f"{base_name}_text_mask.jpg")
            try:
                success, buffer = cv2.imencode('.jpg', refined_mask)
                if success:
                    with open(mask_path, 'wb') as f:
                        f.write(buffer)
                    text_logger.info(f"最終マスク画像を保存しました: {mask_path}")
            except Exception as e:
                text_logger.error(f"マスク画像の保存中にエラーが発生しました: {e}")

        # 最終出力画像の作成（透過PNG）
        if self.color_mode == "original":
            # オリジナルモードでは、元の画像の色を使用
            masked = np.zeros_like(img)
            for c in range(3):
                masked[:, :, c] = np.where(refined_mask > 0, img[:, :, c], 0)
        else:
            # 単色モードでは指定された色で文字を描画
            color = self._parse_color(self.mono_color)
            masked = np.zeros_like(img)
            masked[refined_mask > 0] = color

        # 透過PNG用にAlphaチャネルを設定
        rgba = cv2.cvtColor(masked, cv2.COLOR_BGR2BGRA)
        # アルファチャンネルにマスクを設定（文字部分は不透明、背景は透明）
        rgba[:, :, 3] = refined_mask

        # クロップオプションが有効な場合、文字領域のみにトリミング
        if self.crop:
            coords = cv2.findNonZero(refined_mask)
            if coords is not None:
                x, y, w2, h2 = cv2.boundingRect(coords)
                # 余白を追加
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w2 = min(w - x, w2 + padding * 2)
                h2 = min(h - y, h2 + padding * 2)
                rgba = rgba[y:y+h2, x:x+w2]

        # 最終的な出力画像を保存
        out_path = os.path.join(out_dir, f"{base_name}_texts.png")
        svg_path = os.path.join(out_dir, f"{base_name}_outline.svg")
        try:
            success, buffer = cv2.imencode('.png', rgba)
            if success:
                with open(out_path, 'wb') as f:
                    f.write(buffer)
                text_logger.info(f"文字抽出画像を保存しました: {out_path}")
                if self.enable_svg:
                    self._save_svg(refined_mask, svg_path)
                return out_path
            else:
                text_logger.error(f"文字抽出画像の保存に失敗しました: {out_path}")
                return None
        except Exception as e:
            text_logger.error(f"画像保存中にエラーが発生しました: {e}")
            return None

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
        morph_kernel_size = 3
        # メディアンフィルタで塩コショウノイズを除去
        blur = cv2.GaussianBlur(
            image, (morph_kernel_size, morph_kernel_size), 0)
        median = cv2.medianBlur(blur, morph_kernel_size)

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

    def enhance_brightness_contrast(self, image, alpha=1.5, beta=0):
        """
        コントラストのみを向上させて、黒いテキスト部分を保持しながら背景の凹凸（グレー部分）を除去します。
        明度は上げないことで、黒い文字が飛ばないようにします。

        Args:
            image: 入力画像
            alpha: コントラスト調整パラメータ（1.0より大きい値でコントラスト増加）
            beta: 明度調整パラメータ（0にして明度は上げない）

        Returns:
            コントラスト調整された画像
        """
        text_logger.info(f"コントラストのみを調整しています（alpha={alpha}, beta={beta}）...")

        # コントラスト調整のみ: g(x) = alpha * f(x) + beta
        # alpha > 1 でコントラスト増加、beta = 0 で明度は変えない
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        # 背景がより白くなるように閾値処理を追加
        if len(adjusted.shape) == 3:
            gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
        else:
            gray = adjusted

        # 背景の明るい部分をさらに白くする閾値処理
        # 閾値を高めに設定（200→230）して、より明確に背景と文字を分離
        _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)

        # 暗い部分（テキスト部分）を検出
        _, dark_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        # グレースケールの場合
        if len(adjusted.shape) == 2:
            # 背景を白に、テキスト部分はそのまま保持
            result = thresh.copy()
            result[dark_mask == 255] = adjusted[dark_mask == 255]
            return result

        # カラー画像の場合
        adjusted_bgr = adjusted.copy()
        # 背景を白に
        adjusted_bgr[thresh == 255] = [255, 255, 255]
        # テキスト部分の黒をより強調（オプション）
        dark_regions = dark_mask == 255
        for c in range(3):
            adjusted_bgr[:, :, c][dark_regions] = adjusted_bgr[:,
                                                               :, c][dark_regions] * 0.8

        return adjusted_bgr

    def adaptive_threshold(self, image):
        """
        単純な二値化処理を行います。
        Otsuの方法を使用して最適な閾値を自動的に決定します。
        黒い文字の形状をより正確に保存します。

        Args:
            image: 入力画像

        Returns:
            二値化された画像
        """
        text_logger.info("単純な二値化処理を適用しています...")

        # グレースケールに変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        morph_kernel_size = 2
        # 線を強調するためにガウシアンブラーを適用
        # これにより細い線の連続性が向上します
        gray = cv2.GaussianBlur(
            gray, (morph_kernel_size+1, morph_kernel_size+1), 0)

        # 線を太くするための前処理を適用
        kernel_dilate = np.ones(
            (morph_kernel_size, morph_kernel_size), np.uint8)
        gray = cv2.dilate(gray, kernel_dilate, iterations=1)

        # 単純な二値化（Otsuの方法で閾値を自動決定）
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 小さなノイズを除去と文字の連結
        kernel_open = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        # 文字の連結を強化
        kernel_close = np.ones(
            (morph_kernel_size, morph_kernel_size), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

        return binary

    def preprocess_image(self, image):
        """
        テキスト抽出のための画像前処理を行います。
        ノイズ除去、コントラスト調整、シンプルな二値化を順に適用します。

        Args:
            image: 入力画像 (BGR形式)

        Returns:
            前処理済みの画像と二値化画像のタプル
        """
        text_logger.info("画像の前処理を開始します...")
        # ステップ1: ノイズ除去
        # denoised = self.remove_noise(image)
        denoised = image

        # ステップ2: コントラストを上げて背景を飛ばす（明度は上げない）
        brightened = self.enhance_brightness_contrast(
            denoised, alpha=1.5, beta=0)

        # ステップ3: 通常のコントラスト調整
        enhanced = self.adjust_contrast(brightened)

        # ステップ4: 適応的二値化
        binary = self.adaptive_threshold(enhanced)

        text_logger.info("画像の前処理が完了しました")

        return enhanced, binary
