@ECHO OFF

REM 起動バッチ - ユーザー環境は変更しない、一時的に環境変数を設定して起動

setlocal ENABLEDELAYEDEXPANSION
CD /D "%~dp0"

REM https://github.com/UB-Mannheim/tesseract/wiki からtesseractのインストールしてパスを設定
set "TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe"

REM 仮想環境の activate
call .venv\Scripts\activate.bat

REM Pythonスクリプト実行
python main.py

pause
