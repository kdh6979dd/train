@echo off
cd /d "%~dp0"
set "PY=C:\Users\Administrator\AppData\Local\Programs\Python\Python312\pythonw.exe"
"%PY%" "%~dp0train.py"
exit /b %ERRORLEVEL%
