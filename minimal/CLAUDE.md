# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment Setup

The project requires Python 3.9-3.12 (3.11 recommended). Use the setup scripts in the root directory:

**macOS:**
```bash
chmod +x setup/install_mac.sh
./setup/install_mac.sh
```

**Windows:**
```bash
setup\install_win.bat
```

After setup, activate the virtual environment:
- **macOS:** `source venv/bin/activate`
- **Windows:** `venv\Scripts\activate`

## Core Dependencies

- `numpy>=1.26.0` - Numerical computations
- `neurokit2==0.2.11` - ECG signal processing (pinned version)
- `streamlit>=1.0.0` - Web UI framework
- `plotly>=5.0.0` - Interactive plotting
- `scipy` - Signal processing filters
- `numba>=0.60.0` - Performance optimization

## Running the Application

**Main Streamlit Application:**
```bash
streamlit run main.py
```

**Alternative GUI (from root):**
```bash
python src/gui_ecg.py
```

## Testing

Run tests from the root directory:
```bash
python tests/main.py
python tests/test_implementation_comparison.py
```

## Code Architecture

This is a **minimal ECG analysis implementation** focused on real-time quality assessment. Key components:

### Core Processing Pipeline
1. **decoder.py** - `ECGPacketDecoder` class converts binary .ret files to 12-lead ECG data
2. **filters.py** - Signal preprocessing with morphology, high-pass, notch, and baseline filters
3. **quality.py** - ECG quality analysis using clinical metrics and windowed assessment
4. **main.py** - Streamlit web interface for interactive analysis

### ECG Signal Flow
```
Binary .ret file → Decoder → 8 base channels → Derive 4 additional leads → Apply filters → Quality analysis
```

### Key Configuration
- **Sampling rate:** 500 Hz
- **Base channels:** I, III, V1-V6 (from hardware)
- **Derived leads:** II, aVR, aVL, aVF (calculated)
- **Window analysis:** 2.5s windows with 5s result windows for quality assessment

### Quality Analysis Features
- **Metrics:** SNR, QRS amplitude, muscle artifacts, powerline interference, baseline drift, electrode contact
- **Clinical weights:** Different lead importance for ambulance vs clinical settings
- **Real-time flags:** Automatic problem detection with user-friendly messages
- **Best window selection:** Finds optimal consecutive 5-second segments for analysis

## Data Format

ECG files are stored as binary .ret files in `/data/` directory. The decoder handles:
- Packet-based binary protocol
- 16-bit signed integer samples
- Automatic lead derivation and synchronization
- Built-in data validation and error handling

## Filter Options

1. **Morphology Filter (0)** - Median-based spike removal
2. **High-pass 0.15Hz (1)** - Standard clinical filtering
3. **High-pass 0.05Hz (2)** - Preserve low frequencies
4. **High-pass 0.5Hz (3)** - Aggressive baseline removal

Additional filters: notch (50/60/100Hz), spike removal, baseline correction, human-range bandpass (0.05-40Hz).

## Quality Classification

- **Good:** >0.85 total quality score
- **Questionable:** 0.65-0.85 total quality score  
- **Not usable:** <0.65 total quality score

Quality scores use weighted averages across all 12 leads with clinical importance factors.