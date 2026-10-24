@echo off
setlocal
title DeepSignal Backend

pushd "%~dp0backend" || (
    echo ERROR: Could not open the backend folder.
    pause
    exit /b 1
)

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python was not found. Install Python and add it to PATH, then try again.
    popd
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating the backend virtual environment...
    python -m venv .venv
    if errorlevel 1 goto :failed
)

if not exist ".env" if exist ".env.example" (
    copy /Y ".env.example" ".env" >nul
    echo Created backend\.env from backend\.env.example.
    echo Add your Supabase settings there if you want online storage.
)

echo Installing or updating backend dependencies...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto :failed

echo.
echo Starting the backend at http://localhost:8000
echo Keep this window open while using DeepSignal.
".venv\Scripts\python.exe" -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
set "EXIT_CODE=%ERRORLEVEL%"
popd
if not "%EXIT_CODE%"=="0" (
    echo.
    echo The backend stopped with exit code %EXIT_CODE%.
    pause
)
exit /b %EXIT_CODE%

:failed
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo ERROR: Backend setup failed with exit code %EXIT_CODE%.
popd
pause
exit /b %EXIT_CODE%
