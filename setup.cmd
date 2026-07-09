@echo off
REM Platform: Windows — run: setup.cmd
REM Launcher for setup.ps1 when PowerShell blocks script execution.
REM Fix permanently: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
