import numpy as np
from typing import Dict, Any, List, Tuple
from scipy.signal import welch, butter, sosfiltfilt

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

FLAGS_WEIGHTS: Dict[str, float] = {
    "Muscle_Artifact": 0.2,
    "Bad_Electrode_Contact": 0.3,
    "Powerline_Interference": 0.1,
    "Baseline_Drift": 0.15,
    "Low_SNR": 0.25,
}

# Thresholds for flagging issues for (T_good,T_bad)
T_GRADES: Dict[str, Tuple[float, float]] = {
    "Muscle_Artifact": (0.1, 0.3),
    "Bad_Electrode_Contact": (200, 600),  # qrs threshold above 0.5
    "Powerline_Interference": (0.05, 0.2),
    "Baseline_Drift": (0.02, 0.1),
    "Low_SNR": (18, 6),  # SNR in dB
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
    flags: Dict[str, float] = {}

    # Muscle artifact: 40–100 Hz excluding 50/60 Hz >10%
    # hf = np.sum(psd[(freqs > 40) & (freqs < 100)])
    hf = np.sum(psd[(freqs > 40) & (freqs < 100) & ~((freqs > 49) & (freqs < 51))])
    ma = hf / total_power
    flags["Muscle_Artifact"] = np.clip(
        (ma - T_GRADES["Muscle_Artifact"][0])
        / (T_GRADES["Muscle_Artifact"][1] - T_GRADES["Muscle_Artifact"][0]),
        0,
        1,
    )
    # flags["Muscle_Artifact"] = 1.0 if ma > 0.1 else -np.log10(1 - 9 * ma)

    # Powerline interference: 49–51 & 59–61 Hz
    p50 = np.sum(psd[(freqs > 49) & (freqs < 51)])
    p60 = np.sum(psd[(freqs > 59) & (freqs < 61)])
    pi = (p50 + p60) / total_power
    flags["Powerline_Interference"] = np.clip(
        (pi - T_GRADES["Powerline_Interference"][0])
        / (
            T_GRADES["Powerline_Interference"][1]
            - T_GRADES["Powerline_Interference"][0]
        ),
        0,
        1,
    )
    # flags["Powerline_Interference"] = 1.0 if pi > 0.05 else -np.log10(1 - 20 * pi)

    # Baseline drift: <0.5 Hz
    lf = np.sum(psd[(freqs < 0.5)])
    bd = lf / total_power
    flags["Baseline_Drift"] = np.clip(
        (bd - T_GRADES["Baseline_Drift"][0])
        / (T_GRADES["Baseline_Drift"][1] - T_GRADES["Baseline_Drift"][0]),
        0,
        1,
    )
    # flags["Baseline_Drift"] = 1.0 if bd > 0.1 else -np.log10(1 - 10 * bd)

    # QRS amplitude
    amp = float(np.ptp(sig))

    # Bad electrode contact: low qrs amlitude - flat line or really high qrs amplitude
    qrs_threshold = 10  # Adjust this threshold based on expected QRS amplitude
    lf = np.sum(psd[(freqs > 0.01) & (freqs < 0.5)])
    bc = lf / total_power

    if amp < qrs_threshold:
        # If amplitude is too low, we consider it a bad contact
        flags["Bad_Electrode_Contact"] = 1.0
    else:
        flags["Bad_Electrode_Contact"] = np.clip(
            (amp - T_GRADES["Bad_Electrode_Contact"][0])
            / (
                T_GRADES["Bad_Electrode_Contact"][1]
                - T_GRADES["Bad_Electrode_Contact"][0]
            ),
            0,
            1,
        )
    # flags["Bad_Electrode_Contact"] = 1.0 if bc > 0.2 else -np.log10(1 - 5 * bc)

    # SNR in dB: using bandpass 0.5–40 Hz
    try:
        sos = butter(2, [0.5, 40], btype="bandpass", fs=sampling_rate, output="sos")
        clean = sosfiltfilt(sos, sig)
        noise = sig - clean
        noise_power = np.mean(noise**2) + 1e-12
        snr = 10 * np.log10((amp**2) / noise_power)
    except Exception:
        # Fallback: simple SNR calculation
        signal_power = np.mean(sig**2) + 1e-12
        noise_power = np.var(sig) + 1e-12
        snr = 10 * np.log10(signal_power / noise_power)

    flags["Low_SNR"] = np.clip(
        (snr - T_GRADES["Low_SNR"][0])
        / (T_GRADES["Low_SNR"][1] - T_GRADES["Low_SNR"][0]),
        0,
        1,
    )
    # flags["Low_SNR"] = 1.0 if snr < 25 else -np.log10(1 - (snr / 50))

    return {
        "flags": flags,
        "values": {
            "m_a": ma,
            "b_e_c": bc,
            "p_i": pi,
            "b_d": bd,
            "snr": snr,
            "qrs_amp": amp,
        },
    }


def compute_quality_score(flags: Dict[str, float]) -> float:
    """
    Derive quality score (0.0–1.0) where each true flag deducts 0.2.
    """
    # score = max(0.0, 1.0 - 0.2 * sum(flags.values()))
    score = 1.0
    for flag, value in flags.items():
        if value > 0.0:
            score -= FLAGS_WEIGHTS.get(flag, 0.2) * value
    score = max(0.0, min(1.0, score))
    return score


def analyze_ecg_all_leads(
    leads: Dict[str, np.ndarray],
    sampling_rate: int = 500,
    weights: Dict[str, float] = CLINICAL_WEIGHTS,
    window_sec: float = 2.5,
    result_window_sec: float = 5.0,
) -> Dict[str, Any]:
    """
    Analyze all 12 leads, segmenting into windows, computing per-window metrics,
    finding the best consecutive windows that form a 5-second result window,
    aggregating total, and returning problems for feedback.

    Returns:
      lead_quality: Dict of lead -> {QualityScore, Problems: List[str], SNR_dB, QRS_Amplitude}
      total_quality: float
      classification: str
      best_windows_used: List[int] - consecutive window indices used for result
    """
    # Window parameters
    wlen = int(window_sec * sampling_rate)
    n = len(next(iter(leads.values())))
    nwin = n // wlen if wlen > 0 else 0

    # Calculate how many consecutive windows we need for the result window
    windows_needed = int(result_window_sec / window_sec)  # Should be 2 for 5sec/2.5sec

    # Store all window metrics for finding best consecutive windows
    all_window_metrics: List[Dict[str, Any]] = []

    # Compute metrics for all windows
    for i in range(nwin):
        window_metrics = {}
        window_quality_scores = []

        for lead, sig in leads.items():
            seg = sig[i * wlen : (i + 1) * wlen]
            metrics = analyze_lead_quality(seg, sampling_rate)
            flags = metrics["flags"]
            quality_score = compute_quality_score(flags)

            window_metrics[lead] = {
                "quality_score": quality_score,
                "metrics": metrics,
                "flags": flags,
            }
            window_quality_scores.append(quality_score)

        # Calculate overall window quality (average across all leads)
        window_overall_quality = np.mean(window_quality_scores)
        all_window_metrics.append(
            {
                "window_index": i,
                "overall_quality": window_overall_quality,
                "lead_metrics": window_metrics,
            }
        )

    # Find the best consecutive windows that form the result window
    best_consecutive_windows = []
    best_consecutive_quality = -1.0

    # Try all possible consecutive window combinations
    for start_idx in range(nwin - windows_needed + 1):
        consecutive_windows = all_window_metrics[start_idx : start_idx + windows_needed]

        # Calculate average quality of this consecutive sequence
        avg_quality = np.mean([w["overall_quality"] for w in consecutive_windows])

        if avg_quality > best_consecutive_quality:
            best_consecutive_quality = avg_quality
            best_consecutive_windows = consecutive_windows

    lead_quality: Dict[str, Any] = {}
    total_quality = 0.0

    for lead in leads.keys():
        if best_consecutive_windows:
            # Use metrics from best consecutive windows for this lead
            best_metrics = [w["lead_metrics"][lead] for w in best_consecutive_windows]

            # Average quality scores and metrics from best windows
            quality_scores = [m["quality_score"] for m in best_metrics]
            qrs_amp_values = [m["metrics"]["values"]["qrs_amp"] for m in best_metrics]
            snr_values = [m["metrics"]["values"]["snr"] for m in best_metrics]
            ma_values = [m["metrics"]["values"]["m_a"] for m in best_metrics]
            bec_values = [m["metrics"]["values"]["b_e_c"] for m in best_metrics]
            pi_values = [m["metrics"]["values"]["p_i"] for m in best_metrics]
            bd_values = [m["metrics"]["values"]["b_d"] for m in best_metrics]

            avg_q = float(np.mean(quality_scores))
            avg_qrs_amp = float(np.mean(qrs_amp_values))
            avg_snr = float(np.mean(snr_values))
            avg_ma = float(np.mean(ma_values))
            avg_bec = float(np.mean(bec_values))
            avg_pi = float(np.mean(pi_values))
            avg_bd = float(np.mean(bd_values))

            # Count flags across best windows (flag is present if it appears in >50% of best windows)
            flag_counts: Dict[str, int] = {k: 0 for k in FLAG_MESSAGES.keys()}
            for metrics in best_metrics:
                flags = metrics["flags"]
                for flag in FLAG_MESSAGES.keys():
                    if (
                        flag in flags and flags[flag] > 0.5
                    ):  # Consider flag active if value > 0.5
                        flag_counts[flag] += 1

            # Determine problems: any flag present in >50% of best windows
            problems: List[str] = []
            for flag, count in flag_counts.items():
                if (
                    len(best_consecutive_windows) > 0
                    and (count / len(best_consecutive_windows)) >= 0.5
                ):
                    problems.append(FLAG_MESSAGES[flag])
        else:
            # Fallback if no windows available
            avg_q = 0.0
            avg_snr = 0.0
            avg_qrs_amp = 0.0
            avg_ma = 0.0
            avg_bec = 0.0
            avg_pi = 0.0
            avg_bd = 0.0
            problems = []

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
        "best_windows_used": (
            [w["window_index"] for w in best_consecutive_windows]
            if best_consecutive_windows
            else []
        ),
    }
