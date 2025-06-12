from typing import Dict, TypeVar, Any, cast
import numpy as np
from numpy.typing import NDArray
import neurokit2 as nk
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
from ecg_processor import EcgQualityProcessor


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
        self.quality_scores: Dict[str, Dict[str, Any]] = {}
        self.quality_processor = EcgQualityProcessor(sampling_rate=sampling_rate)

    def decode_data(self, data_bytes: bytes) -> None:
        """
        Decode raw byte data from the ECG glove into individual lead signals.
        Populates self.lead_signals with keys for each lead.
        :param data_bytes: Byte string from ECG glove.
        :raises ValueError: If no valid ECG data is found in the byte stream.
        """
        # Initialize lists for each of the 8 channels (leads) captured by the glove
        channels: Dict[int, list[int]] = {i: [] for i in range(8)}
        i = 0
        size = len(data_bytes)

        while i < size:
            # Look for the start of a packet (0x80 marker) and a valid header
            if data_bytes[i] == 0x80 and i + 7 < size:
                # Check for specific header pattern 0x80 0x17 0x00 and verify header checksum
                if data_bytes[i + 1] == 0x17 and data_bytes[i + 2] == 0x00:
                    # Compute header checksum over 7 bytes (from i to i+6)
                    header_sum = sum(data_bytes[i : i + 7]) & 0xFF  # sum mod 256
                    packet_type = data_bytes[i + 5]  # packet type indicator in header
                    if header_sum == 0:  # valid header
                        # Inside the data packet processing section, update the value calculation:
                        if packet_type == 0x51:  # ECG data packet
                            # Ensure we have enough bytes for the data packet (81 bytes: 80 data + 1 checksum)
                            if i + 7 + 81 <= size:
                                data_start = i + 7
                                data_chunk = data_bytes[data_start : data_start + 81]
                                # Verify data packet checksum (sum of 81 bytes should mod 256 == 0)
                                if sum(data_chunk) & 0xFF == 0:
                                    # Decode 5 frames of 8 channels (16 bytes per frame)
                                    for frame_start in range(0, 80, 16):
                                        frame = data_chunk[
                                            frame_start : frame_start + 16
                                        ]
                                        # Each frame contains 8 little-endian 16-bit samples (one per channel)
                                        for ch in range(8):
                                            # Get LSB and MSB
                                            lsb = frame[2 * ch]
                                            msb = frame[2 * ch + 1]
                                            # Combine bytes matching Java implementation
                                            value = (msb << 8) | (lsb & 0xFF)
                                            # Convert to signed 16-bit if needed
                                            if value > 32767:
                                                value -= 65536
                                            channels[ch].append(value)
                                # Move index past this data packet
                                i = (
                                    data_start + 81 - 1
                                )  # -1 because loop will increment i
                            else:
                                # Incomplete packet at end of data, break out
                                break
                        elif packet_type == 0x03:
                            # Fault packet of 10 bytes payload (after header)
                            if i + 7 + 10 <= size:
                                # (Potentially handle device fault codes here)
                                i += 7 + 10 - 1  # skip the fault packet payload
                            else:
                                break
                        # (Other packet types could be handled here if needed)
                    # After processing header (valid or not), increment i to continue scanning
            i += 1

        # Convert channel lists to numpy arrays and store in lead_signals dict with lead names
        channel_arrays = {
            ch: np.array(data) for ch, data in channels.items() if len(data) > 0
        }
        if not channel_arrays:
            raise ValueError("No ECG data found in the provided byte stream.")

        # Map channels to lead names
        self.lead_signals = {
            "I": channel_arrays.get(0, np.array([])),
            "II": channel_arrays.get(1, np.array([])),
            "V1": channel_arrays.get(2, np.array([])),
            "V2": channel_arrays.get(3, np.array([])),
            "V3": channel_arrays.get(4, np.array([])),
            "V4": channel_arrays.get(5, np.array([])),
            "V5": channel_arrays.get(6, np.array([])),
            "V6": channel_arrays.get(7, np.array([])),
        }

        # Derive the remaining standard leads (III, aVR, aVL, aVF) if limb leads are present
        if self.lead_signals["I"].size > 0 and self.lead_signals["II"].size > 0:
            lead_I = self.lead_signals["I"]
            lead_II = self.lead_signals["II"]

            # Ensure Lead I and II are same length
            length = min(len(lead_I), len(lead_II))
            lead_I, lead_II = lead_I[:length], lead_II[:length]

            # Update the signals with trimmed versions
            self.lead_signals["I"] = lead_I
            self.lead_signals["II"] = lead_II

            # Derive III and augmented leads using the consistent variable names
            self.lead_signals["III"] = lead_II - lead_I
            self.lead_signals["aVR"] = -(lead_I + lead_II) / 2
            self.lead_signals["aVL"] = lead_I - lead_II / 2
            self.lead_signals["aVF"] = lead_II - lead_I / 2
        else:
            # If limb leads not present, we won't derive augmented leads
            self.lead_signals["III"] = np.array([])
            self.lead_signals["aVR"] = np.array([])
            self.lead_signals["aVL"] = np.array([])
            self.lead_signals["aVF"] = np.array([])

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
            ecg_signal = self.lead_signals[primary_lead_name]
            cleaned = nk.ecg_clean(
                ecg_signal, sampling_rate=self.sampling_rate, method=clean_method
            )

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

            # Turn each wave list into an array so we can subtract
            p_onsets = np.array(waves.get("ECG_P_Onsets", []), dtype=float)
            p_offsets = np.array(waves.get("ECG_P_Offsets", []), dtype=float)
            r_onsets = np.array(waves.get("ECG_R_Onsets", []), dtype=float)
            r_offsets = np.array(waves.get("ECG_R_Offsets", []), dtype=float)
            r_peaks_w = np.array(waves.get("ECG_R_Peaks", []), dtype=float)
            q_peaks = np.array(waves.get("ECG_Q_Peaks", []), dtype=float)
            t_offsets = np.array(waves.get("ECG_T_Offsets", []), dtype=float)

            # Build measurements
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
                "P_Duration_ms": (
                    float((p_offsets - p_onsets).mean()) / self.sampling_rate * 1000
                    if p_offsets.size and p_onsets.size
                    else None
                ),
                "PR_Interval_ms": (
                    float((r_peaks_w - p_onsets).mean()) / self.sampling_rate * 1000
                    if r_peaks_w.size and p_onsets.size
                    else None
                ),
                "QRS_Duration_ms": (
                    float((r_offsets - r_onsets).mean()) / self.sampling_rate * 1000
                    if r_offsets.size and r_onsets.size
                    else None
                ),
                "QT_Interval_ms": (
                    float((t_offsets - q_peaks).mean()) / self.sampling_rate * 1000
                    if t_offsets.size and q_peaks.size
                    else None
                ),
                "QTc_Interval_ms": None,  # computed below
            }

            # Calculate QTc if both QT and RR are present
            if measurements["QT_Interval_ms"] and measurements["RR_Interval_ms"]:
                measurements["QTc_Interval_ms"] = measurements[
                    "QT_Interval_ms"
                ] * np.sqrt(1000 / measurements["RR_Interval_ms"])

            results["measurements"] = measurements
            results["R_Peaks"] = (
                info["ECG_R_Peaks"].tolist()
                if isinstance(info["ECG_R_Peaks"], np.ndarray) and "ECG_R_Peaks" in info
                else []
            )

        except Exception as err:
            raise RuntimeError(f"ECG analysis failed: {str(err)}")

        return results
