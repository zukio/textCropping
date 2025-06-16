@ECHO OFF

REM ビルドスクリプト - アプリケーションをexeファイルにビルドする

setlocal ENABLEDELAYEDEXPANSION
CD /D "%~dp0"

REM 仮想環境の activate
call .venv\Scripts\activate.bat

REM 必要なパッケージがインストールされていることを確認
pip install -r requirements.txt --break-system-packages
pip install pyinstaller --break-system-packages

REM 以前のビルドディレクトリをクリーンアップ
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

REM アプリケーションをビルド（コンソールウィンドウなし）
pyinstaller --noconfirm --onefile --windowed ^
  --add-data ".venv\Lib\site-packages\pystray;pystray" ^
  --hidden-import=PIL ^
  --hidden-import=PIL._imagingtk ^
  --hidden-import=PIL._tkinter_finder ^
  --name="ThumbCrafter" ^
  main.py -- --no_console

echo.
echo ビルドが完了しました。dist\ThumbCrafter.exeが生成されました。
echo.

pause
