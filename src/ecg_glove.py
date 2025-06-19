from typing import Dict, TypeVar, Any, cast, Optional, List
from numpy.typing import NDArray
import numpy as np
import neurokit2 as nk
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
from ecg_processor import EcgQualityProcessor
from glove_decoder import ECGPacketDecoder
from ecg_filters import (
    MorphologyFilter,
    NotchEcgFilter,
    HiPassFilter,
    HPFilterType,
    MultiNotchFilter,
    BaselineFilter,
    SmoothingFilter,
)


T = TypeVar("T")


class EcgGlove:
    """Interface for decoding and analyzing data from an ECG12 Glove device."""

    # Constants for signal processing
    DEFAULT_SAMPLING_RATE = 500  # Hz
    DEFAULT_CLEAN_METHOD = "neurokit"
    DEFAULT_PEAK_METHOD = "neurokit"
    DEFAULT_DELINEATE_METHOD = "dwt"

    # Quality thresholds
    MIN_SNR_DB = 10
    MIN_QRS_AMPLITUDE = 0.5  # mV

    def __init__(
        self,
        sampling_rate: int = DEFAULT_SAMPLING_RATE,
        clean_method: str = DEFAULT_CLEAN_METHOD,
        peak_method: str = DEFAULT_PEAK_METHOD,
        filters: Optional[List[int]] = None,
        spike_removal: bool = True,
        hp_filter_type: HPFilterType = HPFilterType.HP015,
        powerline_freq: int = 50,
        enable_baseline_correction: bool = False,
        enable_smoothing: bool = False,
        smoothing_window: int = 5,
    ) -> None:
        """
        Initialize the ECG glove processor.

        Args:
            sampling_rate: Sampling rate of the ECG device (Hz). Default 500 Hz.
            clean_method: Method for cleaning ECG signals.
            peak_method: Method for R-peak detection.
            filters: Optional list of notch filter frequencies (50, 60, 100, 120 Hz)
            spike_removal: Enable morphology-based spike removal filter.
            hp_filter_type: Type of high-pass filter to use.
            powerline_freq: Primary powerline frequency for notch filtering.
            enable_baseline_correction: Enable baseline drift correction.
            enable_smoothing: Enable signal smoothing.
            smoothing_window: Window size for smoothing filter.
        """
        self.sampling_rate = sampling_rate
        self.filters = filters or []
        self.spike_removal = spike_removal
        self.enable_baseline_correction = enable_baseline_correction
        self.enable_smoothing = enable_smoothing

        # Initialize filter components
        self.hp_filter = HiPassFilter(hp_filter_type)

        # Use MultiNotchFilter if multiple frequencies specified, otherwise single NotchEcgFilter
        if len(self.filters) > 1:
            # Filter valid notch frequencies
            valid_notch_freqs = [f for f in self.filters if f in [50, 60, 100, 120]]
            if valid_notch_freqs:
                self.notch = MultiNotchFilter(valid_notch_freqs)
            else:
                self.notch = NotchEcgFilter(powerline_freq)
        else:
            self.notch = NotchEcgFilter(powerline_freq)

        self.morph = MorphologyFilter()

        # Optional filters
        if enable_baseline_correction:
            self.baseline_filter = BaselineFilter(sampling_rate=sampling_rate)

        if enable_smoothing:
            self.smoothing_filter = SmoothingFilter(window_size=smoothing_window)
        self.raw_signals: Dict[str, NDArray[np.float64]] = {}
        self.lead_signals: Dict[str, NDArray[np.float64]] = {}
        self.cleaned_signals: Dict[str, NDArray[np.float64]] = {}
        self.ecg_data = {"raw_signals": {}, "lead_signals": {}, "cleaned_signals": {}}
        self.quality_scores: Dict[str, Dict[str, Any]] = {}
        self.quality_processor = EcgQualityProcessor(sampling_rate=sampling_rate)
        self.clean_config = {
            "function": nk.ecg_clean,
            "method": clean_method,
            "filter_params": {"lowcut": 0.5, "highcut": 40},
        }
        self.peak_config = {
            "function": nk.ecg_peaks,
            "method": peak_method,
            "correct_artifacts": True,
            "show": True,
        }
        self.delineate_config = {
            "function": nk.ecg_delineate,
            "method": "dwt",  # Default method for wave delineation
        }

    def decode_data(self, data_bytes: bytes) -> None:
        """
        Decode raw byte data from the ECG glove into individual lead signals.
        Populates raw_signals, lead_signals (filtered), and cleaned_signals.

        Args:
            data_bytes: Byte string from ECG glove.

        Raises:
            ValueError: If no valid ECG data is found in the byte stream.
        """
        decoder = ECGPacketDecoder()
        # Decode the data using the ECGPacketDecoder
        decoded_leads = decoder.decode(data_bytes)
        if not decoded_leads:
            raise ValueError("No valid ECG data found in the provided byte stream.")

        # Store raw signals first
        self.raw_signals = {
            lead: np.array(signal_data, dtype=np.float64)
            for lead, signal_data in decoded_leads.items()
        }

        # Apply filters to get lead signals
        self.lead_signals = {}
        for lead, signal_data in self.raw_signals.items():
            # First apply bandpass filter
            filtered = self._filter_signal(signal_data)
            self.lead_signals[lead] = filtered

        # Clean the filtered signals
        self.cleaned_signals = {
            lead: np.array(
                nk.ecg_clean(
                    signal_data,
                    sampling_rate=self.sampling_rate,
                    method=self.clean_config["method"],
                ),
                dtype=np.float64,
            )
            for lead, signal_data in self.lead_signals.items()
        }

        # Update ecg_data dictionary
        self.ecg_data["raw_signals"] = self.raw_signals
        self.ecg_data["lead_signals"] = self.lead_signals
        self.ecg_data["cleaned_signals"] = self.cleaned_signals

    def process(self) -> Dict[str, Any]:
        """
        Process the ECG signals to detect R-peaks and compute basic metrics.

        Args:
            clean_method: Method for cleaning ECG ('neurokit', 'biosppy', 'pantompkins', etc.)
            peak_method: Method for R-peak detection ('neurokit', 'pantompkins1985', 'hamilton2002', etc.)

        Returns:
            Dictionary containing analysis results

        Raises:
            RuntimeError: If no decoded lead signals are available or analysis fails
        """
        self._validate_signal_data()

        # Compute quality if not already done
        if not self.quality_scores:
            self.compute_quality()

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

        try:
            # Process waves and calculate measurements
            ecg_data = self._process_waves(primary_lead_name)

            return {
                "AnalysisLead": primary_lead_name,
                "ecgData": ecg_data,
                "quality": self.quality_scores[primary_lead_name],
            }

        except Exception as err:
            raise RuntimeError(f"ECG analysis failed: {str(err)}") from err

    def compute_quality(self) -> Dict[str, Dict[str, Any]]:
        """
        Compute quality metrics for each ECG lead.
        :return: Dictionary containing quality analysis results for all leads.
        :raises RuntimeError: If no decoded lead signals are available.
        """
        if not self.lead_signals:
            raise RuntimeError(
                "No decoded lead signals available. Call decode_data first."
            )

        # Analyze all leads using the quality processor
        results = self.quality_processor.analyze_all_leads(
            {
                "lead_signals": self.lead_signals,
                "cleaned_signals": self.cleaned_signals,
            }
        )

        # Store quality scores
        self.quality_scores = {
            lead: results["lead_quality"][lead] for lead in self.lead_signals.keys()
        }

        return cast(Dict[str, Dict[str, Any]], results)

    def save_leads_to_csv(self, filename: str) -> None:
        """
        Save the decoded lead signals to a CSV file.

        Args:
            filename: Name of the file to save the lead signals.
        """
        if not self.lead_signals:
            raise RuntimeError(
                "No decoded lead signals available. Call decode_data first."
            )

        # Convert lead signals to a DataFrame
        import pandas as pd

        df = pd.DataFrame(self.raw_signals)
        # Rename columns to required format
        column_mapping = {
            "I": "Lead 1",
            "III": "Lead 2",
            "V1": "Lead 3",
            "V2": "Lead 4",
            "V3": "Lead 5",
            "V4": "Lead 6",
            "V5": "Lead 7",
            "V6": "Lead 8",
        }

        # remove v1 - v4 and sort the columns
        df = df[[col for col in column_mapping.keys() if col in df.columns]]
        df = df[list(column_mapping.keys())]  # Ensure correct order
        # Rename columns
        df = df.rename(columns=column_mapping)
        df.to_csv(filename + " py.csv", index=False)
        print(f"Lead signals saved to {filename}.csv")

    def _filter_signal(self, raw: np.ndarray) -> np.ndarray:
        """
        Enhanced filtering pipeline with configurable components:
          1) Baseline correction (optional)
          2) Notch filtering (single or multiple frequencies)
          3) High-pass filtering (morphology-based or IIR)
          4) Smoothing (optional)
        """
        out = []
        for x in raw.tolist():  # iterate sample‐by‐sample
            y = x

            # 1) Optional baseline correction
            if self.enable_baseline_correction and hasattr(self, "baseline_filter"):
                y = self.baseline_filter.get_new_val(y)

            # 2) Notch filtering for power‐line interference
            y = self.notch.get_new_val(y)

            # 3) High-pass filtering
            if self.spike_removal:
                y = self.morph.compute_hpf(int(y))
            else:
                y = self.hp_filter.get_new_val(y)

            # 4) Optional smoothing
            if self.enable_smoothing and hasattr(self, "smoothing_filter"):
                y = self.smoothing_filter.get_new_val(y)

            out.append(y)
        return np.array(out, dtype=np.float64)

    def _validate_signal_data(self) -> None:
        """
        Validate that signal data exists and is valid.

        Raises:
            RuntimeError: If no decoded lead signals are available.
        """
        if not self.lead_signals:
            raise RuntimeError(
                "No decoded lead signals available. Call decode_data first."
            )

    def _calculate_interval(
        self,
        start_points: NDArray[np.float64],
        end_points: NDArray[np.float64],
        description: str,
    ) -> Optional[float]:
        """
        Calculate the mean interval between two sets of points.

        Args:
            start_points: Array of starting points
            end_points: Array of ending points
            description: Description of the interval for error messages

        Returns:
            Mean interval in milliseconds or None if calculation fails
        """
        if start_points.size == 0 or end_points.size == 0:
            return None

        clean_start, clean_end = self._clean_and_align_arrays(start_points, end_points)

        if clean_start.size == 0:
            return None

        return float((clean_end - clean_start).mean()) / self.sampling_rate * 1000

    def _clean_and_align_arrays(
        self, arr1: NDArray[np.float64], arr2: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Remove NaN values and align two arrays.

        Args:
            arr1: First array to clean and align
            arr2: Second array to clean and align

        Returns:
            Tuple of cleaned and aligned arrays
        """
        mask = ~np.isnan(arr1) & ~np.isnan(arr2)
        return arr1[mask], arr2[mask]

    def _calculate_qtc(self, qt_interval: float, rr_interval: float) -> float:
        """
        Calculate the corrected QT interval using Bazett's formula.

        Args:
            qt_interval: QT interval in milliseconds
            rr_interval: RR interval in milliseconds

        Returns:
            QTc interval in milliseconds
        """
        return qt_interval * np.sqrt(1000 / rr_interval)

    def _process_waves(self, primaryLeadName) -> Dict[str, Any]:
        """
        A fully-customizable ECG processing pipeline.

        Returns:
            A dict with keys:
              - "signals": raw and cleaned signals
              - "info": peak locations, heart rate, etc.
              - "waves": p/q/t onsets & offsets
              - "measurements": interval durations, QTc, etc.
        """
        self._validate_signal_data()

        # cleaned = self.cleaned_signals with I , III and primary lead
        cleaned = {
            lead: self.cleaned_signals[lead]
            for lead in ["I", "III", primaryLeadName]
            if lead in self.cleaned_signals
        }

        # 1. Detect peaks and compute heart rate
        #    We'll store per-lead results, but often just lead II is enough.
        peak_results = {}
        for lead, sig in cleaned.items():
            fn = self.peak_config["function"]  # Use get instead of pop
            kwargs = {
                "sampling_rate": self.sampling_rate,
                **{k: v for k, v in self.peak_config.items() if k != "function"},
            }
            signals, info = fn(sig, **kwargs)
            peak_results[lead] = (signals, info)

        # 2. Delineate waves
        waves_results = {}
        for lead, (signals, info) in peak_results.items():
            fn = self.delineate_config["function"]  # Use get instead of pop
            kwargs = {
                "sampling_rate": self.sampling_rate,
                "method": self.delineate_config.get("method", "dwt"),
            }
            _, waves = fn(cleaned[lead], info["ECG_R_Peaks"], **kwargs)
            waves_results[lead] = waves

        # 3. Compute measurements for your chosen lead
        #    (Or loop over all leads if you like.)
        #    Here, pick lead II if available:
        lead = "II" if "II" in cleaned else next(iter(cleaned))
        signals, info = peak_results[lead]
        waves = waves_results[lead]

        # extract wave points
        wave_points = {
            "p_onsets": np.array(waves.get("ECG_P_Onsets", []), dtype=float),
            "p_offsets": np.array(waves.get("ECG_P_Offsets", []), dtype=float),
            "q_peaks": np.array(waves.get("ECG_Q_Peaks", []), dtype=float),
            "r_peaks": np.array(info["ECG_R_Peaks"], dtype=float),
            "r_onsets": np.array(waves.get("ECG_R_Onsets", []), dtype=float),
            "r_offsets": np.array(waves.get("ECG_R_Offsets", []), dtype=float),
            "t_offsets": np.array(waves.get("ECG_T_Offsets", []), dtype=float),
        }

        # Calculate intervals and durations
        measurements = self._calculate_measurements(signals, info, wave_points)

        # Calculate electrical axes
        wave_axes = self._calculate_wave_axes(cleaned, wave_points)
        measurements.update(wave_axes)

        return {
            "raw_signal": self.lead_signals[lead],
            "cleaned_signal": cleaned[lead],
            "signals": signals,
            "info": info,
            "waves": waves,
            "measurements": measurements,
        }

    def _calculate_measurements(
        self,
        signals: Dict[str, Any],
        info: Dict[str, Any],
        wave_points: Dict[str, NDArray[np.float64]],
    ) -> Dict[str, Optional[float]]:
        """
        Calculate ECG measurements from wave points.

        Args:
            signals: Dictionary of processed signals
            info: Dictionary of wave information
            wave_points: Dictionary of wave point arrays

        Returns:
            Dictionary of calculated measurements
        """
        measurements = {
            "HeartRate_BPM": self._calculate_heart_rate(signals),
            "RR_Interval_ms": self._calculate_rr_interval(info),
            "P_Duration_ms": self._calculate_interval(
                wave_points["p_onsets"], wave_points["p_offsets"], "P wave duration"
            ),
            "PR_Interval_ms": self._calculate_interval(
                wave_points["p_onsets"], wave_points["r_peaks"], "PR interval"
            ),
            "QRS_Duration_ms": self._calculate_interval(
                wave_points["r_onsets"], wave_points["r_offsets"], "QRS duration"
            ),
            "QT_Interval_ms": self._calculate_interval(
                wave_points["q_peaks"], wave_points["t_offsets"], "QT interval"
            ),
            "QTc_Interval_ms": None,
        }

        # Calculate QTc if both QT and RR are present
        if measurements["QT_Interval_ms"] and measurements["RR_Interval_ms"]:
            measurements["QTc_Interval_ms"] = self._calculate_qtc(
                measurements["QT_Interval_ms"], measurements["RR_Interval_ms"]
            )

        return measurements

    def _calculate_heart_rate(self, signals: Dict[str, Any]) -> Optional[float]:
        """Calculate mean heart rate from signals."""
        return float(np.mean(signals["ECG_Rate"])) if "ECG_Rate" in signals else None

    def _calculate_rr_interval(self, info: Dict[str, Any]) -> Optional[float]:
        """Calculate mean RR interval from R peaks."""
        if "ECG_R_Peaks" not in info or len(info["ECG_R_Peaks"]) <= 1:
            return None
        return float(np.mean(np.diff(info["ECG_R_Peaks"]))) / self.sampling_rate * 1000

    def _calculate_electrical_axis(
        self, wave_type: str, lead_i_amp: float, lead_iii_amp: float
    ) -> Optional[float]:
        """
        Calculate electrical axis using Lead I and Lead III amplitudes.

        Args:
            wave_type: Type of wave ('P', 'QRS', or 'T')
            lead_i_amp: Amplitude in Lead I
            lead_iii_amp: Amplitude in Lead III

        Returns:
            Axis in degrees or None if calculation fails
        """
        try:
            # Calculate angle using arctangent of Lead III / Lead I
            angle_rad = np.arctan2(lead_iii_amp, lead_i_amp)
            angle_deg = np.degrees(angle_rad)

            # Convert to range [-180, 180]
            if angle_deg > 180:
                angle_deg -= 360
            return float(angle_deg)
        except Exception as e:
            print(f"Error calculating {wave_type} axis: {str(e)}")
            return None

    def _calculate_wave_axes(
        self,
        cleaned_signals: Dict[str, NDArray[np.float64]],
        wave_points: Dict[str, NDArray[np.float64]],
    ) -> Dict[str, Optional[float]]:
        """
        Calculate P, QRS, and T wave axes using Lead I and Lead III.

        Returns:
            Dictionary containing P, QRS, and T wave axes in degrees
        """
        if "I" not in cleaned_signals or "III" not in cleaned_signals:
            return {"P_Axis": None, "QRS_Axis": None, "T_Axis": None}

        results = {}

        # Get signal windows for different waves
        try:
            # P wave
            p_i = float(
                np.mean(
                    [
                        cleaned_signals["I"][int(idx)]
                        for idx in wave_points["p_onsets"]
                        if not np.isnan(idx)
                    ]
                )
            )
            p_iii = float(
                np.mean(
                    [
                        cleaned_signals["III"][int(idx)]
                        for idx in wave_points["p_onsets"]
                        if not np.isnan(idx)
                    ]
                )
            )
            results["P_Axis"] = self._calculate_electrical_axis("P", p_i, p_iii)

            # QRS complex
            qrs_i = float(
                np.mean(
                    [
                        cleaned_signals["I"][int(idx)]
                        for idx in wave_points["r_peaks"]
                        if not np.isnan(idx)
                    ]
                )
            )
            qrs_iii = float(
                np.mean(
                    [
                        cleaned_signals["III"][int(idx)]
                        for idx in wave_points["r_peaks"]
                        if not np.isnan(idx)
                    ]
                )
            )
            results["QRS_Axis"] = self._calculate_electrical_axis("QRS", qrs_i, qrs_iii)

            # T wave
            t_i = float(
                np.mean(
                    [
                        cleaned_signals["I"][int(idx)]
                        for idx in wave_points["t_offsets"]
                        if not np.isnan(idx)
                    ]
                )
            )
            t_iii = float(
                np.mean(
                    [
                        cleaned_signals["III"][int(idx)]
                        for idx in wave_points["t_offsets"]
                        if not np.isnan(idx)
                    ]
                )
            )
            results["T_Axis"] = self._calculate_electrical_axis("T", t_i, t_iii)

        except Exception as e:
            print(f"Error calculating wave axes: {str(e)}")
            results = {
                "P_Axis": cast(Optional[float], None),
                "QRS_Axis": cast(Optional[float], None),
                "T_Axis": cast(Optional[float], None),
            }

        return results


if __name__ == "__main__":
    # Example usage
    ecg_glove = EcgGlove()

    filepath = "data/220209015248248"
    # load byte data from a file or other source
    with open(filepath + ".ret", "rb") as f:
        data_bytes = f.read()
    try:
        ecg_glove.decode_data(data_bytes)
        # save the decoded leads to csv file
        ecg_glove.save_leads_to_csv(filepath)
        results = ecg_glove.process()
        print("ECG Analysis Results:", results)
    except Exception as e:
        print(f"Error processing ECG data: {str(e)}")
