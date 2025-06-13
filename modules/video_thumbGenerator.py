import os
import subprocess

class VideoThumbnailGenerator:
    def create_thumbnail(self, file_path, user_second):
        """指定された動画ファイルからサムネイルを生成します。"""
        thumbnail_path = f"{os.path.splitext(file_path)[0]}_thumbnail.jpg"
        duration = self.get_video_duration(file_path)
        
        if user_second == 0 or duration <= user_second:
            # 動画の長さが指定秒以下の場合、最初のフレームをサムネイルとして生成
            cmd = f'ffmpeg -y -i "{file_path}" -vframes 1 "{thumbnail_path}"'
        else:
            # 分と秒に分割して時間指定
            minutes, seconds = divmod(user_second, 60)
            cmd = f'ffmpeg -y -i "{file_path}" -ss {int(minutes):02d}:{int(seconds):02d} -t 00:00:01 -vframes 1 "{thumbnail_path}"'
        
        subprocess.check_output(cmd, shell=True)
        return thumbnail_path

    def get_video_duration(self, file_path):
        """指定された動画ファイルの長さを取得します。"""
        cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{file_path}"'
        output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
        duration = float(output)
        return duration
