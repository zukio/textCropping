import os
import logging
from datetime import datetime

def setup_logging():
    # ログファイルを保存するディレクトリパス
    logs_dir = './logs'

    # logsディレクトリが存在しない場合は作成する
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # ログの設定
    logging.basicConfig(
        filename=f'{logs_dir}/{datetime.now():%Y-%m-%d}.log',
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p'
    )
    
# ログ設定を行う
setup_logging()
