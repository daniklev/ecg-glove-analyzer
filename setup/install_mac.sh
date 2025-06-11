#!/bin/bash

echo "Setting up ECG Glove Analyzer for macOS..."

# Check Python version and install if needed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 not found. Installing via Homebrew..."
    brew install python@3.11
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing Python packages..."
python3 -m pip install --upgrade pip
python3 -m pip install wheel setuptools
# Install PyWavelets first
python3 -m pip install PyWavelets\>=1.4.1
# Then install the rest of the requirements
python3 -m pip install -r setup/requirements.txt

echo "Installation completed!"