import numpy as np
from scipy.signal import butter, filtfilt, welch
import neurokit2 as nk
from typing import Dict, Any, TypedDict
import warnings
import pandas as pd
from numpy.typing import NDArray

warnings.filterwarnings("ignore")


class LeadData(TypedDict):
    """Type definition for lead data structure."""

    lead_signals: Dict[str, NDArray[np.float64]]
    cleaned_signals: Dict[str, NDArray[np.float64]]


class EcgQualityProcessor:
    # Weights for clinical use
    CLINICAL_LEAD_WEIGHTS = {
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

    # Weights for ambulance use
    AMBULANCE_LEAD_WEIGHTS = {
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

    def __init__(self, sampling_rate: int = 500, use_ambulance_weights: bool = False):
        self.sampling_rate = sampling_rate
        self.lead_weights = (
            self.AMBULANCE_LEAD_WEIGHTS
            if use_ambulance_weights
            else self.CLINICAL_LEAD_WEIGHTS
        )

    def analyze_lead_quality(self, signal: np.ndarray, cleaned) -> Dict[str, Any]:
        """Analyze the quality of a single ECG lead."""
        results = {
            "Muscle_Artifact": False,
            "Bad_Electrode_Contact": False,
            "Powerline_Interference": False,
            "Baseline_Drift": False,
            "Low_SNR": False,
            "QRS_Amplitude": None,
            "SNR_dB": None,
            "nk_quality": None,
        }

        # Validate inputs first
        if signal.size == 0 or cleaned.size == 0:
            print(f"Warning: Empty signal provided to analyze_lead_quality")
            return results

        if np.all(np.isnan(signal)) or np.all(np.isnan(cleaned)):
            print(f"Warning: Signal contains only NaN values")
            return results

        if np.std(signal) == 0 or np.std(cleaned) == 0:
            print(f"Warning: Signal has zero variance (flat line)")
            results["Bad_Electrode_Contact"] = True
            return results

        # Initialize cleaned_safe at the beginning to ensure it's always available
        cleaned_safe = np.array(cleaned, dtype=np.float64)
        if np.any(np.isnan(cleaned_safe)):
            cleaned_safe = cleaned_safe[~np.isnan(cleaned_safe)]

        # Calculate power spectrum
        try:
            freqs, psd = welch(signal, fs=self.sampling_rate)
            total_power = np.sum(psd) if psd.size > 0 else 0

            if total_power > 0:
                # 1. Check for muscle artifact (high-frequency noise > 40 Hz)
                hf_mask = (freqs > 40) & (freqs < 100)
                hf_power = np.sum(
                    psd[hf_mask & ~((freqs > 49) & (freqs < 51))]
                )  # Exclude powerline
                if hf_power / total_power > 0.1:
                    results["Muscle_Artifact"] = True

                # 2. Bad electrode contact → high variance baseline
                low_freq_power = np.sum(psd[(freqs > 0.01) & (freqs < 0.5)])
                if low_freq_power / total_power > 0.2:
                    results["Bad_Electrode_Contact"] = True

                # 3. Powerline interference (50/60 Hz)
                pl_50hz = np.sum(psd[(freqs > 49) & (freqs < 51)])
                pl_60hz = np.sum(psd[(freqs > 59) & (freqs < 61)])
                if (pl_50hz + pl_60hz) / total_power > 0.05:
                    results["Powerline_Interference"] = True

                # 4. Baseline drift
                if low_freq_power / total_power > 0.1:
                    results["Baseline_Drift"] = True

        except Exception as psd_error:
            print(f"Error calculating power spectrum: {str(psd_error)}")

        # Compute quality index with better error handling
        try:
            # Check minimum length requirement
            min_length = max(
                int(self.sampling_rate * 2), 1000
            )  # At least 2 seconds or 1000 samples
            if len(cleaned_safe) < min_length:
                print(
                    f"Warning: Signal too short for quality analysis ({len(cleaned_safe)} < {min_length})"
                )
                results["nk_quality"] = 0.5  # Default middle value
            else:
                # Try NeuroKit2 quality assessment
                quality_idx = nk.ecg_quality(
                    cleaned_safe, sampling_rate=self.sampling_rate
                )

                if quality_idx is not None:
                    quality_idx = np.array(quality_idx, dtype=np.float64)

                    # Handle different return types from NeuroKit2
                    if quality_idx.size > 0:
                        quality_score = float(np.nanmean(quality_idx))
                        # Ensure the score is between 0 and 1
                        quality_score = max(0.0, min(1.0, quality_score))
                        results["nk_quality"] = quality_score
                    else:
                        results["nk_quality"] = 0.5
                else:
                    results["nk_quality"] = 0.5

        except Exception as nk_error:
            print(f"NeuroKit2 quality analysis failed: {str(nk_error)}")
            # Fallback to simple quality metric based on SNR and variance
            try:
                signal_var = np.var(cleaned_safe)
                if signal_var > 0:
                    # Simple quality based on signal characteristics
                    amplitude_range = np.max(cleaned_safe) - np.min(cleaned_safe)
                    if amplitude_range > 0.1:  # Minimum meaningful amplitude
                        results["nk_quality"] = min(
                            1.0, amplitude_range / 5.0
                        )  # Normalize to ~5mV max
                    else:
                        results["nk_quality"] = 0.1
                else:
                    results["nk_quality"] = 0.0
            except Exception as fallback_error:
                print(f"Fallback quality calculation failed: {str(fallback_error)}")
                results["nk_quality"] = 0.5  # Safe default

        # 5. Signal-to-noise ratio (SNR) with better error handling
        try:
            signal_amplitude = np.max(signal) - np.min(signal)
            if signal_amplitude > 0:
                # Use SOS (second-order sections) format for better numerical stability
                try:
                    sos = butter(
                        2,
                        [0.5, 40],
                        btype="bandpass",
                        fs=self.sampling_rate,
                        output="sos",
                    )
                    from scipy.signal import sosfiltfilt
                    clean_signal = sosfiltfilt(sos, signal)
                except Exception:
                    # Fallback to ba format
                    butter_result = butter(
                        2,
                        [0.5, 40],
                        btype="bandpass",
                        fs=self.sampling_rate,
                        output="ba",
                    )
                    if butter_result is None or len(butter_result) != 2:
                        raise ValueError("Failed to create butterworth filter")
                    b, a = butter_result
                    clean_signal = filtfilt(b, a, signal)

                noise = signal - clean_signal
                noise_power = np.mean(noise**2)

                if noise_power > 0:
                    snr = 10 * np.log10(signal_amplitude**2 / noise_power)
                else:
                    snr = 40.0  # Very high SNR if no noise detected
            else:
                snr = 0.0  # No signal amplitude

            results["SNR_dB"] = float(snr)
            if snr < 10:
                results["Low_SNR"] = True

            # QRS amplitude check
            results["QRS_Amplitude"] = float(signal_amplitude)

        except Exception as snr_error:
            print(f"Error calculating SNR: {str(snr_error)}")
            results["SNR_dB"] = 0.0
            results["Low_SNR"] = True
            results["QRS_Amplitude"] = 0.0

        return results

    def analyze_all_leads(
        self, data: LeadData
    ) -> Dict[str, Any]:
        """
        Analyze the quality and measurements of all ECG leads.

        Args:
            data: Dictionary containing raw and cleaned signals for all leads
                 Format: {
                     'lead_signals': Dict[str, NDArray],
                     'cleaned_signals': Dict[str, NDArray]
                 }
            clean_method: Method used for ECG signal cleaning

        Returns:
            Dictionary containing quality analysis and measurements for all leads

        Raises:
            ValueError: If lead_signals or cleaned_signals are missing
        """
        results = {
            "lead_quality": {},
            "overall_quality": None,
            "problem_summary": [],
        }

        if not data.get("lead_signals") or not data.get("cleaned_signals"):
            raise ValueError("Both lead_signals and cleaned_signals must be provided")

        total_weighted_quality = 0.0
        total_weights = 0.0

        # Analyze each lead's quality and measurements
        for lead_name, raw_signal in data["lead_signals"].items():
            if lead_name not in data["cleaned_signals"]:
                print(f"Warning: No cleaned signal for lead {lead_name}")
                continue

            cleaned_signal = data["cleaned_signals"][lead_name]

            # Skip invalid signals
            if raw_signal.size == 0 or cleaned_signal.size == 0:
                print(f"Warning: Empty signal for lead {lead_name}")
                continue

            try:
                # Quality analysis
                quality_results = self.analyze_lead_quality(raw_signal, cleaned_signal)
                results["lead_quality"][lead_name] = quality_results

                # Calculate quality score (0-1)
                quality_score = self._calculate_lead_quality_score(quality_results)
                weight = self.lead_weights.get(lead_name, 0.08)
                total_weighted_quality += quality_score * weight
                total_weights += weight

                # Generate problem description if quality is poor
                if quality_score < 0.8:
                    problem_desc = self._generate_problem_description(
                        lead_name, quality_results
                    )
                    if problem_desc:
                        results["problem_summary"].append(problem_desc)

            except Exception as e:
                print(f"Error analyzing lead {lead_name}: {str(e)}")
                continue

        # Overall quality assessment (normalized by total weights)
        results["overall_quality"] = (
            total_weighted_quality / total_weights if total_weights > 0 else 0.0
        )

        return results

    def _calculate_lead_quality_score(self, quality_results: Dict) -> float:
        """Calculate a quality score between 0 and 1 for a lead."""
        score = 1.0
        if quality_results["Muscle_Artifact"]:
            score -= 0.3
        if quality_results["Bad_Electrode_Contact"]:
            score -= 0.4
        if quality_results["Powerline_Interference"]:
            score -= 0.2
        if quality_results["Baseline_Drift"]:
            score -= 0.2
        if quality_results["Low_SNR"]:
            score -= 0.3
        return max(0.0, score)

    def _generate_problem_description(
        self, lead_name: str, quality_results: Dict[str, Any]
    ) -> str:
        """Generate a user-friendly description of lead quality issues."""
        problems = []

        if quality_results["Bad_Electrode_Contact"]:
            problems.append("poor electrode contact")
        if quality_results["Muscle_Artifact"]:
            problems.append("muscle movement interference")
        if quality_results["Powerline_Interference"]:
            problems.append("electrical interference")
        if quality_results["Baseline_Drift"]:
            problems.append("baseline wandering")
        if quality_results["Low_SNR"]:
            problems.append("low signal quality")

        if problems:
            return f"Lead {lead_name}: {', '.join(problems)}"
        return ""
