@echo off
rem Runs the program from the Python sources (build_exe.bat must have run once: it makes build_env).
cd /d "%~dp0"
start "" build_env\Scripts\pythonw.exe src\xydapp.py
