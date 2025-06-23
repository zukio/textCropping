import os
from watchdog.events import FileSystemEventHandler
from modules.text_extractor import TextExtractor
from modules.utils.logwriter import setup_logging
import logging

# 監視する処理対象ファイルの拡張子
patterns = ['.png', '.jpg', '.jpeg']

# 対象の処理対象ファイルのパスのリスト
target_files = []

# ログの設定を行う
setup_logging()


class TargetFileHandler(FileSystemEventHandler):
    """新たな対象ファイルの追加または既存の対象ファイルの変更を監視し、文字部分のみを抽出した透過PNGを生成します。"""

    def __init__(self, exclude_subdirectories, seconds, output_dir,
                 crop=False, color_mode="original", mono_color="#000000",
                 ocr_engine="tesseract", gcp_credentials=None, enable_svg=True):
        super().__init__()
        self.exclude_subdirectories = exclude_subdirectories
        self.seconds = seconds
        self.output_dir = output_dir
        self.crop = crop
        self.color_mode = color_mode
        self.mono_color = mono_color
        self.ocr_engine = ocr_engine
        self.gcp_credentials = gcp_credentials
        self.enable_svg = enable_svg

    def destroy(self, reason):
        # 終了メッセージをUDPで送信する
        print(reason)
        logging.info(reason)

    def on_deleted(self, event):
        """ファイル削除時に呼び出されます。"""
        try:
            if event.src_path.endswith(tuple(patterns)):
                print(
                    f'Event type: {event.event_type} path : {event.src_path}')
                logging.info(
                    f'Event type: {event.event_type} path : {event.src_path}')
                target_files.remove(event.src_path)
                # サムネイルを削除
                thumb_path = f"{os.path.splitext(event.src_path)[0]}_thumbnail.jpg"
                if os.path.isfile(thumb_path):
                    os.remove(thumb_path)
        except Exception as e:
            print('Error in file monitoring:', e)
            logging.info('[!] Error in file monitoring:', e)

    def on_modified(self, event):
        """ファイルが追加または変更された場合に呼び出されます。"""
        try:
            if event.src_path.endswith(tuple(patterns)):
                print(
                    f'Event type: {event.event_type} path : {event.src_path}')
                logging.info(
                    f'Event type: {event.event_type} path : {event.src_path}')
                target_files.append(event.src_path)
                self.extract_texts(event.src_path)
        except Exception as e:
            print('Error in file monitoring: %s', e)
            logging.info('[!] Error in file monitoring: %s', e)

    def extract_texts(self, file_path):
        """指定された画像から文字部分を抽出します。"""
        try:
            output_path = TextExtractor(
                self.output_dir,
                crop=self.crop,
                color_mode=self.color_mode,
                mono_color=self.mono_color,
                ocr_engine=self.ocr_engine,
                gcp_credentials=self.gcp_credentials,
                enable_svg=self.enable_svg
            ).extract_texts(file_path)
            if output_path:
                print(f'Text extraction succeeded: {output_path}')
                logging.info(f'Text extraction succeeded: {output_path}')
            else:
                print('No text detected. Skipped saving image.')
                logging.info('No text detected. Skipped saving image.')
        except Exception as e:
            print('Text extraction failed: %s', e)
            logging.info('[!] Text extraction failed: %s', e)

    def list_files(self, start_path):
        """指定したディレクトリ（およびそのサブディレクトリ）内のすべてのファイルにイベントを設定"""
        try:
            # 起動時にファイルを読み込んだときのUDP送信
            logging.info('===============')
            logging.info(f'Starting to monitor the directory: {start_path}')
            if self.output_dir:
                from modules.utils.path_utils import is_subpath
                if is_subpath(self.output_dir, start_path):
                    logging.warning(
                        f"警告: 出出力先ディレクトリが監視対象内です: {self.output_dir}")
                    logging.warning("出出力先を監視対象の親ディレクトリに変更します")
                    self.output_dir = os.path.join(
                        os.path.dirname(start_path), 'output')
                    if not os.path.exists(self.output_dir):
                        os.makedirs(self.output_dir, exist_ok=True)
                    logging.info(f"新しい出出力先: {self.output_dir}")
            set_filehandle(self, start_path,
                           self.exclude_subdirectories, target_files)
        except Exception as e:
            print('Error in listing files: %s', e)
            logging.info('[!] Error in listing files: %s', e)


def set_filehandle(event_handler, start_path, exclude_subdirectories, filelist):
    """指定したディレクトリ（およびそのサブディレクトリ）内のすべてのファイルにイベントを設定"""
    if exclude_subdirectories:
        for file in os.listdir(start_path):
            if file.endswith(tuple(patterns)):
                file_path = os.path.join(start_path, file)
                filelist.append(file_path)
                event_handler.extract_texts(file_path)
    else:
        for root, dirs, files in os.walk(start_path):
            current_depth = root.count(
                os.path.sep) - start_path.count(os.path.sep)
            # サブディレクトリの深さが 4 以下の場合のみ処理を行う（誤使用を想定した暴走ガード）
            if current_depth < 5:
                for file in files:
                    if file.endswith(tuple(patterns)):
                        file_path = os.path.join(root, file)
                        filelist.append(file_path)
                        event_handler.extract_texts(file_path)
