@ECHO OFF

REM �N���o�b�` - ���[�U�[���͕ύX���Ȃ��A�ꎞ�I�Ɋ��ϐ���ݒ肵�ċN��

setlocal ENABLEDELAYEDEXPANSION
CD /D "%~dp0"

REM https://github.com/UB-Mannheim/tesseract/wiki ����tesseract�̃C���X�g�[�����ăp�X��ݒ�
set "TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe"

REM ���z���� activate
call .venv\Scripts\activate.bat

REM Python�X�N���v�g���s
python main.py 
--ignore_subfolders false ^
--ip localhost ^
--port 12345 ^
--send_interval 1 ^
--single_instance_only true ^
--target "." ^
--output_dir "..\output"

REM �G���[�R�[�h��Ԃ�
exit /b %ERRORLEVEL%
