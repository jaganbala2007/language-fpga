@echo off
cd /d "%~dp0"
title 100% Offline Multilingual AI Receptionist - Desktop GUI
echo ========================================================
echo   Launching 100% Offline Desktop AI Receptionist GUI
echo ========================================================
start "" ".\.venv\Scripts\python.exe" desktop_app.py
