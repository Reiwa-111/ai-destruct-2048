@echo off
cd /d "%~dp0"
set PYTHONPATH=%CD%\src
python -m destroyer.app
pause