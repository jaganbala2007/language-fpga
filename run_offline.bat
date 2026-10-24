@echo off
title 100% Offline Multilingual AI Receptionist
echo ========================================================
echo   100% OFFLINE MULTILINGUAL AI RECEPTIONIST LAUNCHER
echo   PYNQ-Z2 FPGA / Fixed-Point Q8.8 ALU - Zero Cloud / Zero Browser
echo ========================================================
echo.

if exist .venv\Scripts\python.exe (
    .\.venv\Scripts\python.exe main.py
) else (
    py -3 main.py
)

pause
