import os
from watchdog.events import FileSystemEventHandler
from modules.video_thumbGenerator import VideoThumbnailGenerator
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

    def __init__(self, exclude_subdirectories, seconds):
        super().__init__()
        self.exclude_subdirectories = exclude_subdirectories
        self.seconds = seconds

    def destroy(self, reason):
        # 終了メッセージをUDPで送信する
        print(reason)
        logging.info(reason)
    
    def on_deleted(self, event):
        """ファイル削除時に呼び出されます。"""
        try:
            if event.src_path.endswith(tuple(patterns)):
                print(f'Event type: {event.event_type} path : {event.src_path}')
                logging.info(f'Event type: {event.event_type} path : {event.src_path}')
                video_files.remove(event.src_path)
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
                print(f'Event type: {event.event_type} path : {event.src_path}')
                logging.info(f'Event type: {event.event_type} path : {event.src_path}')
                video_files.append(event.src_path)
                self.create_thumbnail(event.src_path)
        except Exception as e:
            print('Error in file monitoring: %s', e)
            logging.info('[!] Error in file monitoring: %s', e)

    def create_thumbnail(self, file_path):
        """指定された動画ファイルからサムネイルを生成します。"""
        try:
            thumbnail_path = VideoThumbnailGenerator().create_thumbnail(file_path, self.seconds)
            print(f'Thumbnail generation succeeded: {thumbnail_path}')
            logging.info(f'Thumbnail generation succeeded: {thumbnail_path}')
        except Exception as e:
            print('Thumbnail generation failed: %s', e)
            logging.info('[!] Thumbnail generation failed: %s', e)

    def list_files(self, start_path):
        """指定したディレクトリ（およびそのサブディレクトリ）内のすべてのファイルにイベントを設定"""
        try:
            # 起動時にファイルを読み込んだときのUDP送信
            logging.info('===============')
            logging.info(f'Starting to monitor the directory: {start_path}')
            set_filehandle(self, start_path, self.exclude_subdirectories, video_files)
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