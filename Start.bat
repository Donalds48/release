@echo off
cd /d "%~dp0"
title Live-Uebersetzer (Log)
if exist "Start.bat.new" (
    move /y "Start.bat.new" "Start.bat" >nul & start "" "%~dp0Start.bat" & exit /b
)
if not exist ".venv\.installed" call :setup || exit /b 1
".venv\Scripts\python.exe" update.py
if errorlevel 2 call :setup || exit /b 1
if exist "Start.bat.new" (
    move /y "Start.bat.new" "Start.bat" >nul & start "" "%~dp0Start.bat" & exit /b
)
".venv\Scripts\python.exe" translator.py
pause
exit /b

:setup
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
if errorlevel 1 (
    echo.
    echo Einrichtung fehlgeschlagen - Internetverbindung pruefen und Start.bat erneut ausfuehren.
    pause
    exit /b 1
)
exit /b 0
