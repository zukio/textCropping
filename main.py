import os
import sys
import signal
import argparse
import asyncio
import threading
import json
from aioconsole import ainput
from watchdog.observers import Observer
from PIL import Image, ImageDraw
import pystray
from modules.communication.udp_client import DelayedUDPSender, hello_server
from modules.filehandler_communication import TargetFileHandler
from modules.communication.ipc_client import check_existing_instance
from modules.communication.ipc_server import start_server


# プロセスサーバのタスクハンドルを保持する変数
server_task = None

# トレイアイコンのインスタンスを保持する変数
tray_icon = None


def _create_image():
    """Tray icon image."""
    image = Image.new('RGB', (64, 64), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 56, 56), fill=(0, 128, 255))
    return image


def setup_tray(exit_callback):
    """Start system tray icon."""
    icon = pystray.Icon('textCropping', _create_image(), 'textCropping')
    icon.menu = pystray.Menu(
        pystray.MenuItem(
            'Exit', lambda: exit_callback('[Exit] Tray')
        )
    )
    threading.Thread(target=icon.run, daemon=True).start()
    return icon


async def main(args):
    try:
        # プロセスサーバのタスクを開始する
        global server_task
        server_task = asyncio.create_task(start_server(12321, path))

        # ファイルのリストを取得する
        event_handler.list_files(path)

        if args.no_console:
            # バックグラウンドモード（コンソール入力なし）
            print("バックグラウンドモードで実行中です。タスクトレイアイコンから終了できます。")
            while True:
                await asyncio.sleep(1)
        else:
            # 開発モード（コンソール入力あり）
            print("開発モードで実行中です。コンソールに「exit」と入力するか、タスクトレイアイコンから終了できます。")
            while True:
                command = await ainput("Enter a command exit: ")
                if command.lower() == "exit":
                    exit_handler("[Exit] Command Exit")
                    break
                await asyncio.sleep(1)

        # プロセスサーバのタスクが完了するまで待機する
        await server_task

    except asyncio.CancelledError:
        # プロセスサーバのタスクがキャンセルされた場合の処理
        pass
    finally:
        # プロセスサーバのクリーンアップ処理（必要な場合は実装）
        pass


if __name__ == "__main__":
    # 引数 --exclude_subdirectories が指定された場合、ルートディレクトリのみが監視されます。引数が指定されていない場合、サブディレクトリも監視します。
    parser = argparse.ArgumentParser(
        description='Monitor a directory and create thumbnails for video files.')
    parser.add_argument('--config', default='config.json', type=str,
                        help='Path to configuration file')
    parser.add_argument('--exclude_subdirectories', default=False, action='store_true',
                        help='Exclude subdirectories in monitoring and thumbnail creation.')
    parser.add_argument('--target', default='', type=str,
                        help='Directory path to monitor')
    parser.add_argument('--seconds', default=1, type=int,
                        help='Specify the seconds of the frame to be used for thumbnail generation')
    parser.add_argument('--ip', default='localhost', type=str,
                        help='IP address to send the UDP messages')
    parser.add_argument('--port', default=12345, type=int,
                        help='Port number to send the UDP messages')
    parser.add_argument('--delay', default=1, type=int,
                        help='Delay in seconds for sending UDP messages')
    parser.add_argument('--output_dir', default='', type=str,
                        help='Directory path to save processed files')
    parser.add_argument('--no_console', action='store_true',
                        help='Run in background mode without console input')
    parser.add_argument('--disable_udp', action='store_true',
                        help='Disable UDP notifications')
    # 監視するディレクトリパスは、Pythonプロジェクトフォルダが置かれたディレクトリ（およびそのサブディレクトリ）
    args = parser.parse_args()
    config = {}
    if os.path.isfile(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            try:
                config = json.load(f)
            except Exception:
                config = {}

    # 設定ファイルの値で上書きし、さらに起動引数があればそちらを優先
    for key in ['exclude_subdirectories', 'target', 'seconds', 'ip', 'port', 'delay', 'output_dir', 'no_console']:
        if getattr(args, key) == parser.get_default(key) and key in config:
            setattr(args, key, config[key])

    if args.disable_udp == parser.get_default('disable_udp'):
        use_udp = config.get('use_udp', True)
    else:
        use_udp = not args.disable_udp
    path = os.path.abspath(args.target) if args.target else os.path.abspath(
        os.path.join(os.getcwd(), os.pardir))

    # 出力先ディレクトリの設定
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        # Python ディレクトリ直下に output フォルダを作成（n:/data/Python/output）
        output_dir = os.path.join(os.path.dirname(os.getcwd()), 'output')

    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 出力先と監視対象が同じ場合はエラー
    if output_dir == path:
        print('Output directory must be different from the target directory.')
        sys.exit(1)

    # 既に起動しているインスタンスをチェックする
    if check_existing_instance(12321, path):
        print("既に起動しています。")
        sys.exit(0)

    # 既に別のインスタンスが実行中でないかロックファイルをチェックする
    # check_previous_instance()

    # 実行ファイルと同じディレクトリにロックファイルを作成する
    # create_pid_file(os.path.dirname(os.path.abspath(__file__)))

    # [UDP] DelayedUDPSenderのインスタンスを作成し、それをTargetFileHandlerのインスタンスとlist_files関数に渡します。
    udp_sender = DelayedUDPSender(args.delay)
    # [UDP] ファイルが変更されるたびにudp_sender.send_udp_messageが呼び出され、UDPメッセージが適切なタイミングで送信されます。
    event_handler = TargetFileHandler(
        args.exclude_subdirectories, udp_sender, args.ip, args.port, args.seconds, output_dir, use_udp)

    # サーバーとの通信を試みる
    response = hello_server(path)
    if response is not None:
        print("Hello UDP: " + response)
        if response == "overlapping":
            # remove_pid_file()
            sys.exit("[Exit] Overlapping")

    # 監視を開始する
    observer = Observer()
    observer.schedule(event_handler, path,
                      recursive=not args.exclude_subdirectories)
    observer.start()    # 終了処理

    def exit_handler(reason):
        global tray_icon, server_task
        print(f"終了処理を開始します: {reason}")

        # 監視を停止
        observer.stop()
        observer.join(timeout=1.0)  # 最大1秒間待機

        # ファイルハンドラのクリーンアップ
        result = event_handler.destroy(reason)

        # トレイアイコンを停止
        if tray_icon:
            print("トレイアイコンを停止します")
            tray_icon.stop()

        # サーバータスクをキャンセル
        if server_task:
            print("サーバータスクをキャンセルします")
            server_task.cancel()

        print(f"終了コード: {result}")
        sys.exit(result)

    # プログラムが終了する際に呼び出されるハンドラを登録する
    # atexit.register(exit_handler("[Exit] Normal"))

    # Ctrl+Cなどのシグナルハンドラを登録する
    def exit_wrapper(reason):
        return lambda sig, frame: exit_handler(reason)
    signal.signal(signal.SIGINT, exit_wrapper("[Exit] Signal Interrupt"))

    # タスクトレイアイコンを表示する
    tray_icon = setup_tray(exit_handler)    # アプリケーションのメイン処理
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        # 例外処理
        exit_handler("[Exit] Keyboard Interrupt")
