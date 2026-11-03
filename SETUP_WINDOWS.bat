@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if not errorlevel 1 (
 py -3 -m venv .venv
) else (
 python -m venv .venv
)
if errorlevel 1 (
 echo Install Python 3.10 or newer, including the Python launcher, then retry.
 pause
 exit /b 1
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r backend\requirements.txt
if errorlevel 1 (
 echo Dependency installation failed. Check your internet connection and the errors above.
 pause
 exit /b 1
)
echo Setup finished. Double-click START_PROJECT.bat.
pause
