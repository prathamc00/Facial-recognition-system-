@echo off
REM Facial Recognition System - Windows Setup Script
REM This script sets up the environment for the facial recognition system

echo ================================================
echo Facial Recognition System - Setup
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

echo [1/5] Python detected
python --version
echo.

REM Create virtual environment (optional but recommended)
set /p CREATE_VENV=Do you want to create a virtual environment? (y/n): 
if /i "%CREATE_VENV%"=="y" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created: venv
    echo.
    echo To activate it, run: venv\Scripts\activate
    echo.
    pause
)

REM Install dependencies
echo [2/5] Installing Python dependencies...
echo This may take several minutes...
echo.
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo.
echo [3/5] Dependencies installed successfully
echo.

REM Create necessary directories
echo [4/5] Creating project directories...
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist "data\faces" mkdir data\faces
echo Directories created:
echo   - data/
echo   - data/faces/
echo   - models/
echo   - logs/
echo.

REM Initialize database
echo [5/5] Initializing database...
cd src
python -c "from face_database import FaceDatabase; db = FaceDatabase('../data/face_database.db'); print('Database initialized successfully'); db.close()"
if errorlevel 1 (
    echo Warning: Failed to initialize database
    echo The application will create it on first run
)
cd ..
echo.

echo ================================================
echo Setup Complete!
echo ================================================
echo.
echo Next steps:
echo 1. Run the application: python src/enhanced_main.py
echo 2. Or run the GUI: python src/gui_app.py
echo.
echo For help and documentation, see README.md
echo.
pause
