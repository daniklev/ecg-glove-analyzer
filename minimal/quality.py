import numpy as np
from typing import Dict, Any, List
from scipy.signal import welch, butter, filtfilt

# Clinical lead weights for ECG Quality aggregation
CLINICAL_WEIGHTS: Dict[str, float] = {
    "I": 0.07,
    "II": 0.12,
    "III": 0.06,
    "aVR": 0.04,
    "aVL": 0.06,
    "aVF": 0.09,
    "V1": 0.10,
    "V2": 0.10,
    "V3": 0.10,
    "V4": 0.08,
    "V5": 0.09,
    "V6": 0.09,
}

AMBULANCE_WEIGHTS: Dict[str, float] = {
    "I": 0.06,
    "II": 0.20,
    "III": 0.07,
    "aVR": 0.03,
    "aVL": 0.05,
    "aVF": 0.10,
    "V1": 0.12,
    "V2": 0.10,
    "V3": 0.08,
    "V4": 0.07,
    "V5": 0.06,
    "V6": 0.06,
}

# Mapping of flag names to user-friendly messages
FLAG_MESSAGES: Dict[str, str] = {
    "Muscle_Artifact": "Excess muscle noise",
    "Bad_Electrode_Contact": "Poor electrode contact",
    "Powerline_Interference": "Power-line interference detected",
    "Baseline_Drift": "Baseline drift present",
    "Low_SNR": "Low signal-to-noise ratio",
}


def analyze_lead_quality(
    signal: np.ndarray, sampling_rate: int = 500
) -> Dict[str, Any]:
    """
    Compute signal-quality metrics for a single ECG segment.
    Returns:
      - flags: Dict of boolean flags
      - values: Dict of computed values
      - QRS_Amplitude (float)
      - SNR_dB (float)
    """
    # Remove DC offset
    sig = signal - np.mean(signal)

    # Estimate power spectral density
    freqs, psd = welch(sig, fs=sampling_rate)
    total_power = np.sum(psd) + 1e-12

    # Initialize flags
    flags: Dict[str, bool] = {}

    # Muscle artifact: 40–100 Hz >10%
    hf = np.sum(psd[(freqs > 40) & (freqs < 100)])
    ma = hf / total_power
    flags["Muscle_Artifact"] = ma > 0.1

    # Bad electrode contact: 0.01–0.5 Hz >20%
    lf = np.sum(psd[(freqs > 0.01) & (freqs < 0.5)])
    bc = lf / total_power
    flags["Bad_Electrode_Contact"] = bc > 0.2
    # flags["Bad_Electrode_Contact"] = (lf / total_power) > 0.20

    # Powerline interference: 49–51 & 59–61 Hz >5%
    p50 = np.sum(psd[(freqs > 49) & (freqs < 51)])
    p60 = np.sum(psd[(freqs > 59) & (freqs < 61)])
    pi = (p50 + p60) / total_power
    flags["Powerline_Interference"] = pi > 0.05
    # flags["Powerline_Interference"] = ((p50 + p60) / total_power) > 0.05

    # Baseline drift: <0.5 Hz >10%
    bd = lf / total_power
    flags["Baseline_Drift"] = bd > 0.1
    # flags["Baseline_Drift"] = (lf / total_power) > 0.10

    # QRS amplitude
    amp = float(np.ptp(sig))

    # SNR in dB: using bandpass 0.5–40 Hz
    b, a = butter(2, [0.5, 40], btype="bandpass", fs=sampling_rate, output="ba")
    clean = filtfilt(b, a, sig)
    noise = sig - clean
    noise_power = np.mean(noise**2) + 1e-12
    snr = 10 * np.log10((amp**2) / noise_power)
    flags["Low_SNR"] = snr < 25

    return {
        "flags": flags,
        "values": {
            "m_a": ma,
            "b_e_c": bc,
            "p_i": pi,
            "b_d": bd,
        },
        "QRS_Amplitude": amp,
        "SNR_dB": snr,
    }


def compute_quality_score(flags: Dict[str, bool]) -> float:
    """
    Derive quality score (0.0–1.0) where each true flag deducts 0.2.
    """
    score = max(0.0, 1.0 - 0.2 * sum(flags.values()))
    return score


def analyze_ecg_all_leads(
    leads: Dict[str, np.ndarray],
    sampling_rate: int = 500,
    weights: Dict[str, float] = CLINICAL_WEIGHTS,
    window_sec: float = 2.0,
) -> Dict[str, Any]:
    """
    Analyze all 12 leads, segmenting into windows, computing per-window metrics,
    averaging per-lead, aggregating total, and returning problems for feedback.

    Returns:
      lead_quality: Dict of lead -> {QualityScore, Problems: List[str], SNR_dB, QRS_Amplitude}
      total_quality: float
      classification: str
    """
    # Window parameters
    wlen = int(window_sec * sampling_rate)
    n = len(next(iter(leads.values())))
    nwin = n // wlen if wlen > 0 else 0

    lead_quality: Dict[str, Any] = {}
    total_quality = 0.0

    for lead, sig in leads.items():
        # Accumulate quality scores and metrics across windows
        q_list: List[float] = []
        snr_list: List[float] = []
        qrs_amp_list: List[float] = []
        ma: List[float] = []
        bec: List[float] = []
        pi: List[float] = []
        bd: List[float] = []
        flag_counts: Dict[str, int] = {k: 0 for k in FLAG_MESSAGES}

        for i in range(nwin):
            seg = sig[i * wlen : (i + 1) * wlen]
            metrics = analyze_lead_quality(seg, sampling_rate)
            flags = metrics["flags"]
            # Count flags
            for k, v in flags.items():
                if v:
                    flag_counts[k] += 1
            # Collect metrics
            q_list.append(compute_quality_score(flags))
            snr_list.append(metrics["SNR_dB"])
            qrs_amp_list.append(metrics["QRS_Amplitude"])
            ma.append(metrics["values"]["m_a"])
            bec.append(metrics["values"]["b_e_c"])
            pi.append(metrics["values"]["p_i"])
            bd.append(metrics["values"]["b_d"])

        # Average quality and metrics
        avg_q = float(np.mean(q_list)) if q_list else 0.0
        avg_snr = float(np.mean(snr_list)) if snr_list else 0.0
        avg_qrs_amp = float(np.mean(qrs_amp_list)) if qrs_amp_list else 0.0
        avg_ma = float(np.mean(ma)) if ma else 0.0
        avg_bec = float(np.mean(bec)) if bec else 0.0
        avg_pi = float(np.mean(pi)) if pi else 0.0
        avg_bd = float(np.mean(bd)) if bd else 0.0

        # Determine problems: any flag present in >50% of windows
        problems: List[str] = []
        for flag, count in flag_counts.items():
            if nwin > 0 and (count / nwin) > 0.5:
                problems.append(FLAG_MESSAGES[flag])

        lead_quality[lead] = {
            "QualityScore": avg_q,
            "Problems": problems,
            "SNR_dB": avg_snr,
            "QRS_Amplitude": avg_qrs_amp,
            "m_a": avg_ma,
            "b_e_c": avg_bec,
            "p_i": avg_pi,
            "b_d": avg_bd,
        }
        total_quality += avg_q * weights.get(lead, 0.0)

    # Classification
    if total_quality > 0.8:
        classification = "Good"
    elif total_quality > 0.5:
        classification = "Questionable"
    else:
        classification = "Not usable"

    return {
        "lead_quality": lead_quality,
        "total_quality": total_quality,
        "classification": classification,
    }
