import os
from watchdog.events import FileSystemEventHandler
from modules.text_extractor import TextExtractor, text_logger
import json
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

    def __init__(self, exclude_subdirectories, sender, ip, port, seconds, output_dir,
                 crop=False, color_mode="original", mono_color="#000000", enable_udp=True,
                 ocr_engine="tesseract", gcp_credentials=None, debug_output=True, enable_svg=True):
        super().__init__()
        self.exclude_subdirectories = exclude_subdirectories
        self.ip = ip
        self.port = port
        self.sender = sender
        self.event_queue = []  # イベントキューを追加
        self.seconds = seconds
        self.output_dir = output_dir
        self.enable_udp = enable_udp
        self.enable_svg = enable_svg
        self.crop = crop
        self.color_mode = color_mode
        self.mono_color = mono_color
        self.ocr_engine = ocr_engine
        self.gcp_credentials = gcp_credentials
        self.debug_output = debug_output        # 設定情報をログに記録
        logging.info(f"==============")
        logging.info(f"OCRエンジン: {self.ocr_engine}")
        logging.info(f"サブディレクトリ除外設定: {self.exclude_subdirectories}")
        logging.info(f"カラーモード: {self.color_mode}")
        logging.info(f"デバッグ出力: {self.debug_output}")
        if self.enable_udp:
            logging.info(f"UDP通信: 有効 (IP: {self.ip}, ポート: {self.port})")
        else:
            logging.info("UDP通信: 無効")

        # 処理済みファイルを追跡するためのセット
        self.processed_files = set()

    def destroy(self, reason):
        # 終了メッセージをUDPで送信する
        if self.enable_udp:
            self.sender.send_message(self.ip, self.port, reason)
        print(reason)
        logging.info(reason)

    def send_udp_message(self):
        if not self.enable_udp:
            return
        try:
            # イベントキューが空でない場合にUDPメッセージを送信
            if self.event_queue:
                # イベント情報を逆順にソートして配列に格納
                events = [{
                    "type": event.event_type,
                    "path": event.src_path,
                } for event in self.event_queue[::-1]]
                # メッセージ送信
                message = json.dumps({
                    "events": events,
                    "files": target_files
                })
                self.sender.send_message(self.ip, self.port, message)
                self.event_queue.clear()  # イベントキューをクリア

        except Exception as e:
            print('Error in udp sending:', e)
            logging.info('[!] Error in udp sending: %s', e)

    def queue_event(self, event):
        # メッセージ送信キューに追加
        if self.enable_udp:
            self.event_queue.append(event)
            self.send_udp_message()

    def on_created(self, event):
        """ファイル作成時に呼び出されます。"""
        try:
            # ディレクトリの場合はスキップ
            if event.is_directory:
                return

            if event.src_path.endswith(tuple(patterns)):
                # 既に処理済みのファイルはスキップ
                if event.src_path in self.processed_files:
                    logging.debug(f"スキップ（既に処理済み）: {event.src_path}")
                    return

                # 処理済みリストに追加
                self.processed_files.add(event.src_path)

                # リストを更新
                if event.src_path not in target_files:
                    target_files.append(event.src_path)

                # 文字抽出を実行
                # メッセージ送信キューに追加
                self.extract_texts(event.src_path)
                self.queue_event(event)
        except Exception as e:
            print('Error in file monitoring:', e)
            logging.info('[!] Error in file monitoring: %s', e)

    def on_deleted(self, event):
        """ファイル削除時に呼び出されます。"""
        try:
            # ディレクトリの場合はスキップ
            if event.is_directory:
                return

            if event.src_path.endswith(tuple(patterns)):
                # 処理済みリストから削除
                if event.src_path in self.processed_files:
                    self.processed_files.remove(event.src_path)

                target_files.remove(event.src_path)
                # サムネイルを削除
                thumb_path = f"{os.path.splitext(event.src_path)[0]}_thumbnail.jpg"
                if os.path.isfile(thumb_path):
                    os.remove(thumb_path)
                # メッセージ送信キューに追加
                self.queue_event(event)
        except Exception as e:
            print('Error in file monitoring:', e)
            logging.info('[!] Error in file monitoring: %s', e)

    def on_modified(self, event):
        """ファイルが追加または変更された場合に呼び出されます。"""
        try:
            # ディレクトリの場合はスキップ
            if event.is_directory:
                return

            if event.src_path.endswith(tuple(patterns)):
                # 既に処理済みのファイルはスキップ
                if event.src_path in self.processed_files:
                    logging.debug(f"スキップ（既に処理済み）: {event.src_path}")
                    return

                # 処理済みリストに追加
                self.processed_files.add(event.src_path)

                # リストを更新
                if event.src_path not in target_files:
                    target_files.append(event.src_path)

                # 文字抽出を実行
                self.extract_texts(event.src_path)

                # メッセージ送信キューに追加
                self.queue_event(event)
        except Exception as e:
            print('Error in file monitoring:', e)
            logging.info('[!] Error in file monitoring:', e)

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
                debug_output=self.debug_output,
                enable_svg=self.enable_svg
            ).extract_texts(file_path)
            if output_path:
                print(f'Text extraction succeeded: {output_path}')
                logging.info(f'Text extraction succeeded: {output_path}')
                # 文字検出ログにもファイル処理の成功を記録
                text_logger.info(
                    f'ファイル処理完了: {os.path.basename(file_path)} -> {os.path.basename(output_path)}')
            else:
                print('No text detected. Skipped saving image.')
                logging.info('No text detected. Skipped saving image.')
                text_logger.info(
                    f'ファイル処理スキップ: {os.path.basename(file_path)} - 文字なし')
        except Exception as e:
            print('Text extraction failed:', e)
            logging.info('[!] Text extraction failed: %s', e)
            # 文字検出ログにもエラーを記録
            text_logger.error(
                f'ファイル処理失敗: {os.path.basename(file_path)} - エラー: {e}')

    def reset_processed_files(self):
        """
        一定時間後に処理済みファイルリストをリセットします。
        長時間実行時のメモリ使用量を抑えるために使用します。
        このメソッドは定期的に呼び出されるべきです。
        """
        # 現在の時刻からn時間前（例：1時間前）に追加されたファイルを対象とする場合、
        # タイムスタンプとともに保存する必要があります。
        # 簡易実装として、単にリストをクリアします。
        file_count = len(self.processed_files)
        if file_count > 1000:  # ファイル数が1000を超えた場合にリセット
            logging.info(f"処理済みファイルリストをリセットします（{file_count}ファイル）")
            self.processed_files.clear()

    def list_files(self, start_path):
        """指定したディレクトリ（およびそのサブディレクトリ）内のすべてのファイルを一覧表示します。"""
        try:
            # 処理済みファイルリストをリセット
            self.reset_processed_files()

            logging.info('===============')
            logging.info(f'Starting to monitor the directory: {start_path}')

            # 出力ディレクトリが監視対象パス内に含まれていないかチェック
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
            if self.enable_udp:
                self.sender.send_message(self.ip, self.port, json.dumps({
                    "events": [{"type": "Startup", "path": ""}],
                    "files": target_files
                }))
        except Exception as e:
            print('Error in listing files: %s', e)
            logging.info('[!] Error in listing files: %s', e)


def set_filehandle(event_handler, start_path, exclude_subdirectories, filelist):
    """指定したディレクトリ（およびそのサブディレクトリ）内のすべてのファイルを一覧表示します。"""
    if exclude_subdirectories:
        for file in os.listdir(start_path):
            if file.endswith(tuple(patterns)):
                file_path = os.path.join(start_path, file)
                filelist.append(file_path)
                # 処理済みリストに追加
                event_handler.processed_files.add(file_path)
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
                        # 処理済みリストに追加
                        event_handler.processed_files.add(file_path)
                        event_handler.extract_texts(file_path)
