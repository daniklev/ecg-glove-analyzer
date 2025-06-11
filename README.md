# ECG Glove Analyzer

## System Requirements

- Python 3.9-3.12 (3.11 recommended)
- macOS 10.15 or later / Windows 10 or later

## Installation

1. Clone this repository:
```bash
git clone https://github.com/daniklev/ecg-glove-analyzer.git
cd ecg-glove-analyzer
```

2. Run the appropriate setup script:

### For macOS:
```bash
chmod +x setup/install_mac.sh
./setup/install_mac.sh
```

### For Windows:
```bash
setup\install_win.bat
```

## Running the Application

1. Activate the virtual environment:

### On macOS:
```bash
source venv/bin/activate
```

### On Windows:
```bash
venv\Scripts\activate
```

2. Run the application:
```bash
python src/gui_ecg.py
```

## Troubleshooting

If you encounter installation issues:

1. Make sure you have the correct Python version installed:
```bash
python3 --version  # Should be 3.9-3.12
```

2. Try creating a fresh virtual environment:
```bash
rm -rf venv  # On macOS
# or
rmdir /s /q venv  # On Windows

# Then run the installation script again
```

3. If specific packages fail to install, try installing them manually:
```bash
pip install numpy>=1.26.0
pip install neurokit2==0.2.11
pip install matplotlib>=3.7.0
pip install PyQt5>=5.15.0
```