import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))
from ecg_glove import EcgGlove
from glove_decoder import ECGPacketDecoder

# C# interop
import clr
import System
from System import Array
from System.IO import File

# Load the CommwellEcgCalculation assembly
dll_path = str(Path(__file__).parent.parent / "CommwellEcgCalculation/bin/Debug/net7.0/CommwellEcgCalculation.dll")
clr.AddReference(dll_path)
from CommwellEcgCalculation import CommCalc

def load_cs_data(file_path):
    """Load and process data using C# implementation."""
    data = File.ReadAllBytes(file_path)
    calc = CommCalc()
    result = calc.gloveCalc(data)
    
    if result != 0:
        raise ValueError(f"C# gloveCalc failed with result {result}")
    
    waves = calc.getWaves()
    if waves is None:
        raise ValueError("No wave data returned from C#")
        
    # Convert waves to numpy arrays for easier comparison
    cs_leads = {}
    for i, wave in enumerate(waves):
        # Map channel index to lead name according to documentation
        lead_name = {
            0: "I",
            1: "III",
            2: "V1",
            3: "V2", 
            4: "V3",
            5: "V4",
            6: "V5",
            7: "V6"
        }.get(i)
        
        if lead_name:
            # Convert System.Collections.Generic.List to numpy array
            cs_leads[lead_name] = np.array([float(x) for x in wave])
            
    # Calculate derived leads
    if "I" in cs_leads and "III" in cs_leads:
        cs_leads["II"] = cs_leads["I"] + cs_leads["III"]
        cs_leads["aVR"] = -(cs_leads["I"] + cs_leads["II"]) / 2
        cs_leads["aVL"] = cs_leads["I"] - cs_leads["II"] / 2
        cs_leads["aVF"] = cs_leads["II"] - cs_leads["I"] / 2
        
    return cs_leads

def load_python_data(file_path):
    """Load and process data using Python implementation."""
    with open(file_path, 'rb') as f:
        data = f.read()
    
    glove = EcgGlove()
    glove.decode_data(data)
    return glove.lead_signals

def compare_implementations(file_path):
    """Compare lead data between C# and Python implementations."""
    print(f"\nProcessing file: {Path(file_path).name}")
    
    # Load data from both implementations
    try:
        cs_leads = load_cs_data(file_path)
        py_leads = load_python_data(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Compare leads
    all_leads = sorted(set(cs_leads.keys()) & set(py_leads.keys()))
    
    if not all_leads:
        print("No common leads found between implementations!")
        return
        
    # Create comparison plots
    n_leads = len(all_leads)
    fig, axes = plt.subplots(n_leads, 1, figsize=(15, 3*n_leads))
    if n_leads == 1:
        axes = [axes]
    
    stats = {}
    for lead, ax in zip(all_leads, axes):
        cs_data = cs_leads[lead]
        py_data = py_leads[lead]
        
        # Trim to shorter length if needed
        min_len = min(len(cs_data), len(py_data))
        cs_data = cs_data[:min_len]
        py_data = py_data[:min_len]
        
        # Calculate statistics
        diff = cs_data - py_data
        rmse = np.sqrt(np.mean(diff**2))
        max_diff = np.max(np.abs(diff))
        correlation = np.corrcoef(cs_data, py_data)[0,1]
        
        stats[lead] = {
            "RMSE": rmse,
            "Max Difference": max_diff,
            "Correlation": correlation
        }
        
        # Plot
        ax.plot(cs_data, label='C#', alpha=0.7)
        ax.plot(py_data, label='Python', alpha=0.7)
        ax.set_title(f'Lead {lead} Comparison')
        ax.legend()
        ax.grid(True)
        
    plt.tight_layout()
    
    # Save plot
    output_dir = Path(file_path).parent / "comparison_plots"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f"{Path(file_path).stem}_comparison.png")
    
    # Print statistics
    print("\nComparison Statistics:")
    for lead, lead_stats in stats.items():
        print(f"\nLead {lead}:")
        for metric, value in lead_stats.items():
            print(f"  {metric}: {value:.6f}")

def main():
    # Test with sample files
    test_files = [
        "../data/2310301106481048.ret",
        "../data/220209012940240.ret",
        "../data/230910065042942.ret"
    ]
    
    for file_path in test_files:
        abs_path = str(Path(__file__).parent.resolve() / file_path)
        if os.path.exists(abs_path):
            compare_implementations(abs_path)
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    main()
