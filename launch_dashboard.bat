@echo off
title NeuroSense AI Clinician Dashboard
cd /d "%~dp0"
set PYTHON_EXE=C:\Users\navan\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe

echo ========================================================
echo   Starting NeuroSense AI Clinician Dashboard...
echo   (The webpage will automatically open once ready)
echo   IMPORTANT: Keep this window open while using the app!
echo ========================================================
echo.

"%PYTHON_EXE%" -u app.py --port 8055
pause
