from typing import Dict, TypeVar, Any, cast
import numpy as np
from numpy.typing import NDArray
import neurokit2 as nk
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
from ecg_processor import EcgQualityProcessor
from glove_decoder import ECGPacketDecoder

T = TypeVar("T")


class EcgGlove:
    """Interface for decoding and analyzing data from an ECG12 Glove device."""

    def __init__(self, sampling_rate: int = 500) -> None:
        """
        Initialize the ECG glove processor.
        :param sampling_rate: Sampling rate of the ECG device (Hz). Default 500 Hz.
        """
        self.sampling_rate = sampling_rate
        self.lead_signals: Dict[str, NDArray[np.float64]] = {}
        self.cleaned_signals: Dict[str, NDArray[np.float64]] = {}
        self.quality_scores: Dict[str, Dict[str, Any]] = {}
        self.quality_processor = EcgQualityProcessor(sampling_rate=sampling_rate)

    def decode_data(self, data_bytes: bytes) -> None:
        """
        Decode raw byte data from the ECG glove into individual lead signals.
        Populates self.lead_signals with keys for each lead.
        :param data_bytes: Byte string from ECG glove.
        :raises ValueError: If no valid ECG data is found in the byte stream.
        """
        decoder = ECGPacketDecoder()
        # Decode the data using the ECGPacketDecoder
        decoded_leads = decoder.decode(data_bytes)
        if not decoded_leads:
            raise ValueError("No valid ECG data found in the provided byte stream.")
        # Store the decoded lead signals
        self.lead_signals = {
            lead: np.array(signal, dtype=np.float64)
            for lead, signal in decoded_leads.items()
        }
        # Clean ecg using NeuroKit2
        self.cleaned_signals = {
            lead: np.array(
                nk.ecg_clean(signal, sampling_rate=self.sampling_rate), dtype=np.float64
            )
            for lead, signal in self.lead_signals.items()
        }

    def compute_quality(
        self, clean_method: str = "neurokit"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute quality metrics for each ECG lead.
        :param clean_method: Method used for ECG signal cleaning.
        :return: Dictionary containing quality analysis results for all leads.
        :raises RuntimeError: If no decoded lead signals are available.
        """
        if not self.lead_signals:
            raise RuntimeError(
                "No decoded lead signals available. Call decode_data first."
            )

        # Analyze all leads using the quality processor
        results = self.quality_processor.analyze_all_leads(self.lead_signals)

        # Store quality scores
        self.quality_scores = {
            lead: results["lead_quality"][lead] for lead in self.lead_signals.keys()
        }

        return cast(Dict[str, Dict[str, Any]], results)

    def process(
        self, clean_method: str = "neurokit", peak_method: str = "neurokit"
    ) -> Dict[str, Any]:
        """
        Process the ECG signals to detect R-peaks and compute basic metrics.
        :param clean_method: Method for cleaning ECG ('neurokit', 'biosppy', 'pantompkins', etc.)
        :param peak_method: Method for R-peak detection ('neurokit', 'pantompkins1985', 'hamilton2002', etc.)
        :return: Dictionary containing analysis results
        :raises RuntimeError: If no decoded lead signals are available or analysis fails
        """
        if not self.lead_signals:
            raise RuntimeError(
                "No decoded lead signals available. Call decode_data first."
            )

        # Compute quality if not already done
        if not self.quality_scores:
            self.compute_quality(clean_method=clean_method)

        # Use Lead II if available and of good quality, otherwise find best lead
        available_leads = {
            lead: self.lead_signals[lead]
            for lead in ["II", "I", "V5", "V6"]
            if self.lead_signals.get(lead, np.array([])).size > 0
        }

        if not available_leads:
            raise RuntimeError("No suitable leads available for analysis")

        # Select best lead based on quality and clinical importance
        lead_qualities = {
            lead: self.quality_processor._calculate_lead_quality_score(
                self.quality_scores[lead]
            )
            for lead in available_leads
        }
        primary_lead_name = max(lead_qualities.items(), key=lambda x: x[1])[0]

        results: Dict[str, Any] = {
            "AnalysisLead": primary_lead_name,
            "measurements": {},
            "quality": self.quality_scores[primary_lead_name],
        }

        try:
            # Clean the ECG signal
            # ecg_signal = self.lead_signals[primary_lead_name]
            # cleaned = nk.ecg_clean(
            #     ecg_signal, sampling_rate=self.sampling_rate, method=clean_method
            # )
            cleaned = self.cleaned_signals[primary_lead_name]
            # Process ECG signal with modified parameters
            signals, info = nk.ecg_process(
                cleaned,
                sampling_rate=self.sampling_rate,
                method=peak_method,
            )

            # Extract waves delineation
            _, waves = nk.ecg_delineate(
                cleaned,
                info["ECG_R_Peaks"],
                sampling_rate=self.sampling_rate,
                method="dwt",
            )

            # Turn each wave list into an array and remove NaN values
            p_onsets = np.array(waves.get("ECG_P_Onsets", []), dtype=float)
            p_offsets = np.array(waves.get("ECG_P_Offsets", []), dtype=float)
            r_onsets = np.array(waves.get("ECG_R_Onsets", []), dtype=float)
            r_offsets = np.array(waves.get("ECG_R_Offsets", []), dtype=float)
            r_peaks = np.array(info["ECG_R_Peaks"], dtype=float)
            q_peaks = np.array(waves.get("ECG_Q_Peaks", []), dtype=float)
            t_offsets = np.array(waves.get("ECG_T_Offsets", []), dtype=float)

            # Remove NaN values and ensure arrays are properly aligned
            def clean_and_align_arrays(arr1, arr2):
                mask = ~np.isnan(arr1) & ~np.isnan(arr2)
                return arr1[mask], arr2[mask]

            # Build measurements with proper array handling
            measurements = {
                "HeartRate_BPM": (
                    float(np.mean(signals["ECG_Rate"]))
                    if "ECG_Rate" in signals
                    else None
                ),
                "RR_Interval_ms": (
                    float(np.mean(np.diff(info["ECG_R_Peaks"])))
                    / self.sampling_rate
                    * 1000
                    if "ECG_R_Peaks" in info and len(info["ECG_R_Peaks"]) > 1
                    else None
                ),
            }

            # Calculate P wave duration
            if p_onsets.size > 0 and p_offsets.size > 0:
                p_on_clean, p_off_clean = clean_and_align_arrays(p_onsets, p_offsets)
                if p_on_clean.size > 0:
                    measurements["P_Duration_ms"] = (
                        float((p_off_clean - p_on_clean).mean()) / self.sampling_rate * 1000
                    )
                else:
                    measurements["P_Duration_ms"] = None
            else:
                measurements["P_Duration_ms"] = None

            # Calculate PR interval
            if p_onsets.size > 0 and r_peaks.size > 0:
                # Ensure we only use the R peaks that have corresponding P onsets
                max_index = min(len(p_onsets), len(r_peaks))
                p_on_clean = p_onsets[:max_index]
                r_clean = r_peaks[:max_index]
                mask = ~np.isnan(p_on_clean)
                if mask.any():
                    measurements["PR_Interval_ms"] = (
                        float((r_clean[mask] - p_on_clean[mask]).mean()) / self.sampling_rate * 1000
                    )
                else:
                    measurements["PR_Interval_ms"] = None
            else:
                measurements["PR_Interval_ms"] = None

            # Calculate QRS duration
            if r_onsets.size > 0 and r_offsets.size > 0:
                r_on_clean, r_off_clean = clean_and_align_arrays(r_onsets, r_offsets)
                if r_on_clean.size > 0:
                    measurements["QRS_Duration_ms"] = (
                        float((r_off_clean - r_on_clean).mean()) / self.sampling_rate * 1000
                    )
                else:
                    measurements["QRS_Duration_ms"] = None
            else:
                measurements["QRS_Duration_ms"] = None

            # Calculate QT interval
            if q_peaks.size > 0 and t_offsets.size > 0:
                q_clean, t_off_clean = clean_and_align_arrays(q_peaks, t_offsets)
                if q_clean.size > 0:
                    measurements["QT_Interval_ms"] = (
                        float((t_off_clean - q_clean).mean()) / self.sampling_rate * 1000
                    )
                else:
                    measurements["QT_Interval_ms"] = None
            else:
                measurements["QT_Interval_ms"] = None

            measurements["QTc_Interval_ms"] = None

            # Calculate QTc if both QT and RR are present
            if measurements["QT_Interval_ms"] and measurements["RR_Interval_ms"]:
                measurements["QTc_Interval_ms"] = measurements["QT_Interval_ms"] * np.sqrt(
                    1000 / measurements["RR_Interval_ms"]
                )

            results["measurements"] = measurements
            results["R_Peaks"] = (
                info["ECG_R_Peaks"].tolist()
                if isinstance(info["ECG_R_Peaks"], np.ndarray) and "ECG_R_Peaks" in info
                else []
            )

        except Exception as err:
            raise RuntimeError(f"ECG analysis failed: {str(err)}")

        return results
