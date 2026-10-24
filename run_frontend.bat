@echo off
setlocal
title DeepSignal Frontend

pushd "%~dp0frontend" || (
    echo ERROR: Could not open the frontend folder.
    pause
    exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
    echo ERROR: npm was not found. Install Node.js, then try again.
    popd
    pause
    exit /b 1
)

if not exist "node_modules" (
    echo Installing frontend dependencies...
    if exist "package-lock.json" (
        call npm ci
    ) else (
        call npm install
    )
    if errorlevel 1 goto :failed
)

echo.
echo Starting the frontend. Open the local URL shown below (usually http://localhost:5173).
echo Keep this window open while using DeepSignal.
call npm run dev
set "EXIT_CODE=%ERRORLEVEL%"
popd
if not "%EXIT_CODE%"=="0" (
    echo.
    echo The frontend stopped with exit code %EXIT_CODE%.
    pause
)
exit /b %EXIT_CODE%

:failed
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo ERROR: Frontend setup failed with exit code %EXIT_CODE%.
popd
pause
exit /b %EXIT_CODE%
