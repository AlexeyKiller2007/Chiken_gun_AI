@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

where py >nul 2>nul && goto :haspy
where python >nul 2>nul && goto :haspython
echo Python not found. Opening python.org - install Python 3.12+ with "Add python.exe to PATH".
start https://www.python.org/downloads/
goto :end

:haspy
py -3 setup.py
goto :end

:haspython
python setup.py

:end
pause
