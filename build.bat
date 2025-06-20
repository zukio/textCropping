@ECHO OFF

REM �r���h�X�N���v�g - �A�v���P�[�V������exe�t�@�C���Ƀr���h����

setlocal ENABLEDELAYEDEXPANSION
CD /D "%~dp0"

REM ���z���� activate
call .venv\Scripts\activate.bat

REM �K�v�ȃp�b�P�[�W���C���X�g�[������Ă��邱�Ƃ��m�F
pip install -r requirements.txt --break-system-packages
pip install pyinstaller --break-system-packages

REM �ȑO�̃r���h�f�B���N�g�����N���[���A�b�v
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

REM �A�v���P�[�V�������r���h�i�R���\�[���E�B���h�E�Ȃ��j
pyinstaller --noconfirm --onefile --windowed ^
  --add-data ".venv\Lib\site-packages\pystray;pystray" ^
  --hidden-import=PIL ^
  --hidden-import=PIL._imagingtk ^
  --hidden-import=PIL._tkinter_finder ^
  --name="textCroping" ^
  main.py -- --no_console

echo.
echo �r���h���������܂����Bdist\textCroping.exe����������܂����B
echo.

pause
