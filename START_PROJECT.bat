@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
 set "PROJECT_PYTHON=.venv\Scripts\python.exe"
) else (
 where conda >nul 2>nul
 if not errorlevel 1 (
  call conda activate new_env
 ) else if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
  call "%USERPROFILE%\anaconda3\Scripts\activate.bat" new_env
 )
 set "PROJECT_PYTHON=python"
)
"%PROJECT_PYTHON%" -c "import fastapi,ultralytics,reportlab" >nul 2>nul
if errorlevel 1 (
 echo First run SETUP_WINDOWS.bat to install the project environment.
 pause
 exit /b 1
)
if not exist "ml\models\best.pt" (
 echo Trained model missing. Restore best.pt or run the training notebook.
 pause
 exit /b 1
)
echo Open http://127.0.0.1:8000 after the server starts.
echo Press Ctrl+C here to stop the application.
start "" /b "%PROJECT_PYTHON%" -c "import time,webbrowser; time.sleep(6); webbrowser.open('http://127.0.0.1:8000')"
"%PROJECT_PYTHON%" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
pause
