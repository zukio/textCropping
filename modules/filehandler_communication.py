import os
from watchdog.events import FileSystemEventHandler
from modules.video_thumbGenerator import VideoThumbnailGenerator
import json
from modules.utils.logwriter import setup_logging
import logging

# 監視する動画ファイルの拡張子
patterns = ['.mp4', '.avi', '.mkv', '.flv', '.mov']

# 対象の動画ファイルのパスのリスト
video_files = []

# ログの設定を行う
setup_logging()

class VideoFileHandler(FileSystemEventHandler):
    """新たな動画ファイルの追加または既存の動画ファイルの変更を監視し、その動画の最初のフレームをサムネイルとして保存します。"""

    def __init__(self, exclude_subdirectories, sender, ip, port, seconds):
        super().__init__()
        self.exclude_subdirectories = exclude_subdirectories
        self.ip = ip
        self.port = port
        self.sender = sender
        self.event_queue = []  # イベントキューを追加
        self.seconds = seconds

    def destroy(self, reason):
        # 終了メッセージをUDPで送信する
        self.sender.send_message(self.ip, self.port, reason)
        print(reason)
        logging.info(reason)


    def use_udp(self):
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
                    "files": video_files
                })
                self.sender.send_message(self.ip, self.port, message)
                self.event_queue.clear()  # イベントキューをクリア

        except Exception as e:
            print('Error in udp sending:', e)
            logging.info('[!] Error in udp sending:', e)

    def queue_event(self, event):
        # メッセージ送信キューに追加
        self.event_queue.append(event)
        self.use_udp()

    # def on_created(self, event):

    def on_deleted(self, event):
        """ファイル削除時に呼び出されます。"""
        try:
            if event.src_path.endswith(tuple(patterns)):
                video_files.remove(event.src_path)
                # サムネイルを削除
                thumb_path = f"{os.path.splitext(event.src_path)[0]}_thumbnail.jpg"
                if os.path.isfile(thumb_path):
                    os.remove(thumb_path)
                # メッセージ送信キューに追加
                self.queue_event(event)
        except Exception as e:
            print('Error in file monitoring:', e)
            logging.info('[!] Error in file monitoring:', e)

    def on_modified(self, event):
        """ファイルが追加または変更された場合に呼び出されます。"""
        try:
            if event.src_path.endswith(tuple(patterns)):
                # リストを更新
                if event.src_path not in video_files:
                    video_files.append(event.src_path)
                # サムネイルを生成
                self.create_thumbnail(event.src_path)
                # メッセージ送信キューに追加
                self.queue_event(event)
        except Exception as e:
            print('Error in file monitoring:', e)
            logging.info('[!] Error in file monitoring:', e)

    def create_thumbnail(self, file_path):
        """指定された動画ファイルからサムネイルを生成します。"""
        try:
            thumbnail_path = VideoThumbnailGenerator().create_thumbnail(file_path, self.seconds)
            print(f'Thumbnail generation succeeded: {thumbnail_path}')
            logging.info(f'Thumbnail generation succeeded: {thumbnail_path}')
        except Exception as e:
            print('Thumbnail generation failed:', e)
            logging.info('[!] Thumbnail generation failed:', e)

    def list_files(self, start_path):
        """指定したディレクトリ（およびそのサブディレクトリ）内のすべてのファイルを一覧表示します。"""
        try:
            # 起動時にファイルを読み込んだときのUDP送信
            logging.info('===============')
            logging.info(f'Starting to monitor the directory: {start_path}')
            set_filehandle(self, start_path, self.exclude_subdirectories, video_files)
            self.sender.send_message(self.ip, self.port, json.dumps({
                "events":[{"type":"Startup", "path": ""}],
                "files": video_files
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
                event_handler.create_thumbnail(file_path)
    else:
        for root, dirs, files in os.walk(start_path):
            current_depth = root.count(os.path.sep) - start_path.count(os.path.sep)
            # サブディレクトリの深さが 4 以下の場合のみ処理を行う（誤使用を想定した暴走ガード）
            if current_depth < 5:
                for file in files:
                    if file.endswith(tuple(patterns)):
                        file_path = os.path.join(root, file)
                        filelist.append(file_path)
                        event_handler.create_thumbnail(file_path)

