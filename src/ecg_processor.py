import numpy as np
from scipy.signal import butter, filtfilt, welch
import neurokit2 as nk
from typing import Dict, Any
import warnings
import pandas as pd

warnings.filterwarnings("ignore")


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

    def analyze_lead_quality(self, signal: np.ndarray) -> Dict[str, Any]:
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

        # Normalize signal
        signal = signal - np.mean(signal)

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

            # Convert signal to float64 to avoid dtype issues
            signal = np.array(signal, dtype=np.float64)

            # Clean the ECG signal first
            cleaned = nk.ecg_clean(
                signal, sampling_rate=self.sampling_rate  # , method=clean_method
            )

            # Ensure cleaned signal is float64
            cleaned = np.array(cleaned, dtype=np.float64)

            # Compute quality index based on method
            quality_idx = nk.ecg_quality(
                cleaned, sampling_rate=self.sampling_rate  # , method=quality_method
            )
            # For averageQRS and other numeric methods
            quality_idx = np.array(quality_idx, dtype=np.float64)
            quality_score = float(np.nanmean(quality_idx))
            # Ensure the score is between 0 and 1
            quality_score = max(0.0, min(1.0, quality_score))
            results["nk_quality"] = quality_score

        except Exception as e:
            print(f"Error calculating power spectrum: {str(e)}")
            freqs = np.array([])
            psd = np.array([])

        # 5. Signal-to-noise ratio (SNR)
        signal_amplitude = np.max(signal) - np.min(signal)
        b, a = butter(2, [0.5, 40], btype="bandpass", fs=self.sampling_rate, output="ba")  # type: ignore
        clean_signal = filtfilt(b, a, signal)
        noise = signal - clean_signal
        noise_power = np.mean(noise**2)
        snr = 10 * np.log10(signal_amplitude**2 / (noise_power + 1e-10))

        results["SNR_dB"] = float(snr)
        if snr < 10:
            results["Low_SNR"] = True

        # QRS amplitude check
        results["QRS_Amplitude"] = float(signal_amplitude)

        return results

    def analyze_all_leads(self, leads_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze the quality and measurements of all ECG leads."""
        results = {
            "lead_quality": {},
            "lead_measurements": {},
            "overall_quality": None,
            "problem_summary": [],
            "measurements": {
                "heart_rate": None,
                "rr_interval": None,
                "p_duration": None,
                "pr_interval": None,
                "qrs_duration": None,
                "qt_interval": None,
                "qtc_interval": None,
            },
        }

        total_weighted_quality = 0

        # Analyze each lead's quality and measurements
        for lead_name, signal in leads_data.items():
            # Quality analysis
            quality_results = self.analyze_lead_quality(signal)
            # Include NeuroKit-based quality separately
            results["lead_quality"][lead_name] = quality_results

            # Calculate quality score (0-1)
            quality_score = self._calculate_lead_quality_score(quality_results)
            weight = self.lead_weights.get(
                lead_name, 0.08
            )  # default weight if not specified
            total_weighted_quality += quality_score * weight

            # Process ECG measurements for good quality leads
            if quality_score > 0.7:  # Only process if lead quality is acceptable
                try:
                    processed_signal, info = nk.ecg_process(
                        signal, sampling_rate=self.sampling_rate
                    )
                    results["lead_measurements"][lead_name] = (
                        self._extract_measurements(processed_signal, info)
                    )
                except Exception as e:
                    print(f"Error processing lead {lead_name}: {str(e)}")

            # Generate problem description if quality is poor
            if quality_score < 0.8:
                problem_desc = self._generate_problem_description(
                    lead_name, quality_results
                )
                if problem_desc:
                    results["problem_summary"].append(problem_desc)

        # Overall quality assessment
        results["overall_quality"] = total_weighted_quality

        # Calculate final measurements from the best leads
        self._calculate_final_measurements(results)

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

    def _extract_measurements(
        self, processed_signal: pd.DataFrame, info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract ECG measurements from processed signal."""
        ecg_info = info.get("ECG", {})
        r_peaks = info.get("ECG_R_Peaks", [])

        return {
            "heart_rate": float(ecg_info.get("Heart_Rate", 0)),
            "rr_interval": (
                float(np.mean(np.diff(r_peaks))) / self.sampling_rate * 1000
                if len(r_peaks) > 1
                else 0
            ),
            "p_duration": float(ecg_info.get("P_Duration", 0)),
            "pr_interval": float(ecg_info.get("PR_Interval", 0)),
            "qrs_duration": float(ecg_info.get("QRS_Duration", 0)),
            "qt_interval": float(ecg_info.get("QT_Interval", 0)),
            "qtc_interval": float(ecg_info.get("QTc", 0)),
        }

    def _calculate_final_measurements(self, results: Dict):
        """Calculate final ECG measurements using the best leads."""
        measurements = results["lead_measurements"]
        if not measurements:
            return

        # For each parameter, collect values from all leads
        params = [
            "heart_rate",
            "rr_interval",
            "p_duration",
            "pr_interval",
            "qrs_duration",
            "qt_interval",
            "qtc_interval",
        ]

        for param in params:
            values = [m[param] for m in measurements.values() if m[param] is not None]
            if values:
                if param in ["qrs_duration", "qt_interval"]:
                    results["measurements"][param] = np.max(
                        values
                    )  # Take maximum for QRS and QT
                else:
                    results["measurements"][param] = np.median(
                        values
                    )  # Take median for others

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
