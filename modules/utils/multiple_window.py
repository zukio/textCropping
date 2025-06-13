import tkinter as tk
from tkinter import messagebox

# 監視中のディレクトリ一覧
monitored_directories = []

# start_monitoring関数は、event_handlerオブジェクトを作成した後に呼び出す必要があります。具体的には、observer.start()の前に呼び出すことができます。
def start_monitoring(directory):
    # 監視処理の開始
    monitored_directories.append(directory)
    messagebox.showinfo("Start Monitoring", f"Start monitoring directory: {directory}")

def launch_duplicate_warning(directory):
    # 多重起動の警告ダイアログを表示
    result = messagebox.askyesno("Duplicate Launch Warning", f"The directory '{directory}' is already being monitored. Do you want to launch another instance?")
    if result:
        start_monitoring(directory)

def handle_directory(directory):
    # 既に起動しているかどうかの判定と処理の分岐
    if directory in monitored_directories:
        launch_duplicate_warning(directory)
    else:
        start_monitoring(directory)

def open_new_directory_dialog():
    # 新しいディレクトリの選択ダイアログを表示
    directory = tk.filedialog.askdirectory()
    if directory:
        handle_directory(directory)

def create_main_window():
    # メインウィンドウの作成
    window = tk.Tk()

    # 監視中のディレクトリボタン
    for directory in monitored_directories:
        button = tk.Button(window, text=directory, command=lambda dir=directory: handle_directory(dir))
        button.pack()

    # 新しいディレクトリボタン
    new_button = tk.Button(window, text="Add New Directory", command=open_new_directory_dialog)
    new_button.pack()

    # window.mainloop()

# メインウィンドウの作成
# create_main_window()
