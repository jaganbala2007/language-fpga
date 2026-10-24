@echo off
cd /d "%~dp0"
title 100% Offline Multilingual AI Receptionist - Web Dashboard
echo ========================================================
echo   Launching Streamlit Web Dashboard (Local Browser)
echo ========================================================
".\.venv\Scripts\python.exe" -m streamlit run app.py
pause
