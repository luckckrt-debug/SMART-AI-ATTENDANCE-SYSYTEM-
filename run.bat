@echo off
setlocal enabledelayedexpansion
title Smart Attendance System - Localhost Server

:: 1. Navigate to project root directory
cd /d "%~dp0"
echo ======================================================================
echo           SMART ATTENDANCE SYSTEM - LOCALHOST STARTUP
echo ======================================================================

:: 2. Find system base Python executable
set "BASE_PYTHON="
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    set "BASE_PYTHON=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
) else if exist "%ProgramFiles%\Python310\python.exe" (
    set "BASE_PYTHON=%ProgramFiles%\Python310\python.exe"
) else (
    where python >nul 2>&1
    if !errorlevel! equ 0 (
        set "BASE_PYTHON=python"
    )
)

:: 3. Test or heal venv\pyvenv.cfg
if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe --version >nul 2>&1
    if !errorlevel! equ 0 (
        goto :VENV_READY
    ) else (
        echo [HEAL] Virtual environment interpreter failed. Auto-repairing pyvenv.cfg...
        if defined BASE_PYTHON (
            (
                echo home = %LOCALAPPDATA%\Programs\Python\Python310
                echo include-system-site-packages = false
                echo version = 3.10.11
                echo executable = %LOCALAPPDATA%\Programs\Python\Python310\python.exe
                echo command = %LOCALAPPDATA%\Programs\Python\Python310\python.exe -m venv "%~dp0venv"
            ) > "venv\pyvenv.cfg"
        )
    )
)

:: 4. If venv still missing, create it
if not exist "venv\Scripts\python.exe" (
    if not defined BASE_PYTHON (
        echo [ERROR] Python 3.10 is not found on your system PATH or standard folders.
        pause
        exit /b 1
    )
    echo [SETUP] Creating Python virtual environment in 'venv'...
    "%BASE_PYTHON%" -m venv venv
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [SETUP] Virtual environment created successfully.
)

:VENV_READY
:: 5. Activate virtual environment
call venv\Scripts\activate.bat

:: 6. Verify and install requirements if needed
python -c "import flask, cv2, tensorflow, dlib, pandas" >nul 2>&1
if %errorlevel% neq 0 (
    echo [SETUP] Installing / verifying required Python dependencies...
    python -m pip install -r requirements.txt
)

:: 7. Check database connectivity
echo [CHECK] Checking database connectivity...
python -c "from database_manager import db_manager; print('[DB STATUS]', db_manager.get_status()['status_message'])"

:: 8. Verify required model resources
echo [CHECK] Verifying model resources...
if not exist "PreTrained_model\facenet_keras.h5" (
    echo [WARNING] facenet_keras.h5 is missing from PreTrained_model\ folder!
)
if not exist "shape_predictor_68_face_landmarks.dat" (
    echo [WARNING] shape_predictor_68_face_landmarks.dat is missing from project root!
)
if not exist "FaceDetection\faces.xml" (
    echo [WARNING] faces.xml is missing from FaceDetection\ folder!
)

:: 9. Open browser and run localhost server
echo ======================================================================
echo  Launching Smart Attendance System at http://127.0.0.1:5000
echo  Press CTRL+C in this terminal window to stop the server.
echo ======================================================================

:: Launch browser in background after 2 seconds
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:5000"

:: Start Flask app
python app.py

pause
