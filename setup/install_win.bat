@echo off
echo Setting up ECG Glove Analyzer for Windows...

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python 3 not found. Please install Python 3.11 from python.org
    exit /b 1
)

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Install requirements
echo Installing Python packages...
python -m pip install --upgrade pip
python -m pip install wheel setuptools
python -m pip install -r setup\requirements.txt

echo Installation completed!
pause