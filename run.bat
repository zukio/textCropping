@ECHO OFF

REM 起動バッチ - ユーザー環境は変更しない、一時的に環境変数を設定して起動

setlocal ENABLEDELAYEDEXPANSION
CD /D "%~dp0"

REM https://github.com/UB-Mannheim/tesseract/wiki からtesseractのインストールしてパスを設定
set "TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe"

set "PORTRACE_PATH=C:\Users\A-SUZUKI\potrace\potrace.exe"

REM 仮想環境の activate
call .venv\Scripts\activate.bat

REM Pythonスクリプト実行
python main.py ^
--ip localhost ^
--port 12345 ^
--delay 1 ^
--target "." ^
--output_dir "..\output" ^
--color_mode="mono" ^
--color="#FFFFFF" ^
--ocr_engine "easyocr"
REM --crop オプションは--crop と書くと True になります。指定しない場合は False になります（デフォルト値）。--crop=False という構文はエラーになります。

REM エラーコードを返す
exit /b %ERRORLEVEL%
