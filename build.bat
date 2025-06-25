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

REM [console|windowed] を引数で切り替えられるようにする
set MODE=windowed
if /I "%1"=="console" set MODE=console

REM PyInstaller オプションを分岐
set OPTIONS=--noconfirm --onefile
if "%MODE%"=="windowed" (
    set OPTIONS=!OPTIONS! --windowed
)

REM アプリケーションをビルド
pyinstaller !OPTIONS! ^
  --add-data ".venv\Lib\site-packages\pystray;pystray" ^
  --hidden-import=PIL ^
  --hidden-import=PIL._imagingtk ^
  --hidden-import=PIL._tkinter_finder ^
  --name="textCroping" ^
  main.py

echo.
echo ビルドが完了しました。dist\textCroping.exeが生成されました。
echo.
pause
