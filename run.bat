@ECHO OFF

REM �N���o�b�` - ���[�U�[���͕ύX���Ȃ��A�ꎞ�I�Ɋ��ϐ���ݒ肵�ċN��

setlocal ENABLEDELAYEDEXPANSION
CD /D "%~dp0"

REM https://github.com/UB-Mannheim/tesseract/wiki ����tesseract�̃C���X�g�[�����ăp�X��ݒ�
set "TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe"

set "PORTRACE_PATH=C:\Users\A-SUZUKI\potrace\potrace.exe"

REM ���z���� activate
call .venv\Scripts\activate.bat

REM Python�X�N���v�g���s
python main.py ^
--ip localhost ^
--port 12345 ^
--delay 1 ^
--target "." ^
--output_dir "..\output" ^
--color_mode="mono" ^
--color="#FFFFFF" ^
--ocr_engine "easyocr"
REM --crop �I�v�V������--crop �Ə����� True �ɂȂ�܂��B�w�肵�Ȃ��ꍇ�� False �ɂȂ�܂��i�f�t�H���g�l�j�B--crop=False �Ƃ����\���̓G���[�ɂȂ�܂��B

REM �G���[�R�[�h��Ԃ�
exit /b %ERRORLEVEL%
