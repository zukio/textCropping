import os
import sys
import signal

# PIDファイルのパス
PID_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".pid")
# PID_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".lock")
  

def check_previous_instance():
    if os.path.isfile(PID_FILE_PATH):
        print("Another instance is already running.")
        sys.exit(1)

def create_pid_file(path):
    PID_FILE_PATH = os.path.join(path, ".pid")
    with open(PID_FILE_PATH, "w") as file:
        file.write(str(os.getpid()))

def remove_pid_file():
    if os.path.isfile(PID_FILE_PATH):
        os.remove(PID_FILE_PATH)

def exit_handler(signum, frame):
    remove_pid_file()
    sys.exit(0)

# if __name__ == "__main__":
    ## 既に別のインスタンスが実行中でないかをチェックする
    #check_previous_instance()

    ## PIDファイルを作成する
    #create_pid_file()

    ## 終了時にPIDファイルを削除するためのシグナルハンドラを登録する
    #signal.signal(signal.SIGINT, exit_handler)
    #signal.signal(signal.SIGTERM, exit_handler)

