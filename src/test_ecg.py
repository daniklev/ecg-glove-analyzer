import os
import numpy as np
import neurokit2 as nk
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import math
import argparse


class EcgGlove:
    """Interface for decoding and analyzing data from an ECG12 Glove device."""

    def __init__(self, sampling_rate=500):
        """
        Initialize the ECG glove processor.
        :param sampling_rate: Sampling rate of the ECG device (Hz). Default 500 Hz.
        """
        self.sampling_rate = sampling_rate
        # This dictionary will hold the decoded ECG lead signals (as numpy arrays)
        self.lead_signals = {}

    def decode_data(self, data_bytes: bytes) -> None:
        """
        Decode raw byte data from the ECG glove into individual lead signals.
        Populates self.lead_signals with keys for each lead.
        :param data_bytes: Byte string from ECG glove.
        """
        # Initialize lists for each of the 8 channels (leads) captured by the glove
        channels = {i: [] for i in range(8)}
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
                        if packet_type == 0x51:  # ECG data packet
                            # Ensure we have enough bytes for the data packet (81 bytes: 80 data + 1 checksum)
                            if i + 7 + 81 <= size:
                                # Data payload starts at i+7 and spans 81 bytes (including its checksum)
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
                                            # Little-endian int16: byte order [LSB, MSB]
                                            lsb = frame[2 * ch]
                                            msb = frame[2 * ch + 1]
                                            # Combine bytes and interpret as signed 16-bit
                                            value = (msb << 8) | lsb
                                            if value & 0x8000:  # if sign bit is set
                                                value = (
                                                    value - 0x10000
                                                )  # two's complement conversion
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
        # Assuming channel mapping: 0->Lead I, 1->Lead II, 2->V1, 3->V2, 4->V3, 5->V4, 6->V5, 7->V6
        # These assumptions are based on the device capturing 2 limb leads and 6 chest leads to form a 12-lead ECG.
        # If the actual channel mapping differs, this section should be adjusted accordingly.
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
        self,
        clean_method: str = "neurokit",
        quality_method: str = "averageQRS",
    ) -> dict:
        """
        Compute a quality score for each ECG lead using NeuroKit's ecg_quality.
        Returns a dictionary of {lead_name: quality_score}.

        Args:
            clean_method (str): Method for cleaning ECG signal ('neurokit', 'biosppy', etc.)
            quality_method (str): Method for quality assessment ('averageQRS' or 'zhao2018')
        Returns:
            dict: Dictionary mapping lead names to quality scores (0-1 range for averageQRS, or string for zhao2018)
        """
        if not self.lead_signals:
            raise RuntimeError(
                "No decoded lead signals available. Call decode_data first."
            )

        quality_scores = {}
        for lead, signal in self.lead_signals.items():
            if signal.size == 0:
                continue  # skip empty leads

            try:
                # Convert signal to float64 to avoid dtype issues
                signal = np.array(signal, dtype=np.float64)

                # Clean the ECG signal first
                cleaned = nk.ecg_clean(
                    signal, sampling_rate=self.sampling_rate, method=clean_method
                )

                # Ensure cleaned signal is float64
                cleaned = np.array(cleaned, dtype=np.float64)

                # Compute quality index based on method
                quality_idx = nk.ecg_quality(
                    cleaned, sampling_rate=self.sampling_rate, method=quality_method
                )

                if quality_method == "zhao2018":
                    # For zhao2018, quality_idx is already a string value
                    # Just take the first value since they should all be the same
                    quality_scores[lead] = (
                        quality_idx if len(quality_idx) > 0 else "Unknown"
                    )
                else:
                    # For averageQRS and other numeric methods
                    quality_idx = np.array(quality_idx, dtype=np.float64)
                    quality_score = float(np.nanmean(quality_idx))
                    # Ensure the score is between 0 and 1
                    quality_score = max(0.0, min(1.0, quality_score))
                    quality_scores[lead] = quality_score

            except Exception as e:
                print(f"Warning: Could not compute quality for lead {lead}: {str(e)}")
                quality_scores[lead] = (
                    "Unknown" if quality_method == "zhao2018" else float("nan")
                )

        return quality_scores

    def analyze(
        self,
        clean_method: str = "neurokit",
        peak_method: str = "neurokit",
        # quality_method: str = "averageQRS",
    ) -> dict:
        """
        Analyze the ECG signals to detect R-peaks and compute basic metrics using neurokit.
        :param clean_method: Method for cleaning ECG ('neurokit', 'biosppy', 'pantompkins', etc.)
        :param peak_method: Method for R-peak detection ('neurokit', 'pantompkins1985', 'hamilton2002', etc.)
        :param quality_method: Method for signal quality assessment ('averageQRS', 'zhao2018', etc.)
        :return: Dictionary containing analysis results (e.g., R-peak indices, heart rate).
        """
        if not self.lead_signals:
            raise RuntimeError(
                "No decoded lead signals available. Call decode_data first."
            )
        results = {}
        # Select a primary lead for analysis (the original code chose the lead with more detections between I and II).
        # As a simple approach, we'll use Lead II if available, otherwise fall back to Lead I.
        primary_lead_name = (
            "II" if self.lead_signals.get("II", np.array([])).size > 0 else "I"
        )
        ecg_signal = self.lead_signals.get(primary_lead_name)
        if ecg_signal is None or ecg_signal.size == 0:
            # If neither I nor II is present, pick the first available lead
            for ld, sig in self.lead_signals.items():
                if sig.size > 0:
                    ecg_signal = sig
                    primary_lead_name = ld
                    break
        # Ensure we have a valid signal to analyze
        if ecg_signal is None or ecg_signal.size == 0:
            raise RuntimeError("No valid ECG signal available for analysis.")

        try:
            # Clean the ECG signal
            cleaned = nk.ecg_clean(
                ecg_signal, sampling_rate=self.sampling_rate, method=clean_method
            )

            # Detect R-peaks
            peaks, info = nk.ecg_peaks(
                cleaned, sampling_rate=self.sampling_rate, method=peak_method
            )
            rpeaks = np.where(peaks["ECG_R_Peaks"] == 1)[0]

            # Process the full ECG signal
            signals, process_info = nk.ecg_process(
                cleaned, sampling_rate=self.sampling_rate
            )

            # Store results
            results["R_peaks_indices"] = rpeaks.tolist()
            results["AnalysisLead"] = primary_lead_name

            # Compute heart rate from R-peaks (if at least two peaks are found)
            if len(rpeaks) >= 2:
                # Calculate RR intervals in seconds
                rr_intervals = np.diff(rpeaks) / float(self.sampling_rate)
                hr_bpm = 60.0 / np.mean(rr_intervals)
                results["HeartRate_BPM"] = hr_bpm
            else:
                results["HeartRate_BPM"] = None

            # # Add quality metrics
            # quality_idx = nk.ecg_quality(
            #     cleaned, sampling_rate=self.sampling_rate, method=quality_method
            # )
            # results["SignalQuality"] = float(np.nanmean(quality_idx))

            # Add additional processed signal info
            results["ProcessedSignals"] = {
                "Cleaned": cleaned,
                "R_Peaks": peaks["ECG_R_Peaks"],
                "HeartRate": signals["ECG_Rate"] if "ECG_Rate" in signals else None,
                # "Quality": quality_idx,
            }

        except Exception as err:
            raise RuntimeError(f"ECG analysis failed: {str(err)}")
        return results


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Process ECG12 glove .ret file")
#     parser.add_argument("file_path", help="Path to .ret file")
#     args = parser.parse_args()
#     file_path = args.file_path

#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"File {file_path} does not exist.")
#     with open(file_path, "rb") as f:
#         data_bytes = f.read()

#     glove = EcgGlove(sampling_rate=500)

#     try:
#         glove.decode_data(data_bytes)
#     except ValueError as e:
#         print("Decoding error:", e)
#         exit(1)

#     # Compute quality for each lead
#     quality = glove.compute_quality()
#     print("Lead quality scores:")
#     for lead, q in quality.items():
#         print(f"  {lead}: {q:.3f}")

#     # Perform analysis (using NeuroKit algorithm by default)
#     analysis_results = glove.analyze(algorithm="neurokit")
#     print("\nAnalysis results:")
#     print(f"  AnalysisLead: {analysis_results.get('AnalysisLead')}")
#     print(f"  R_peaks_indices: {analysis_results.get('R_peaks_indices')}")
#     hr = analysis_results.get("HeartRate_BPM")
#     if hr is not None:
#         print(f"  Estimated Heart Rate: {hr:.1f} BPM")
#     else:
#         print("  Estimated Heart Rate: None")

#     # Optional: plot decoded leads if matplotlib is available
#     try:
#         leads = list(glove.lead_signals.keys())
#         num_leads = len(leads)
#         cols = 3
#         rows = math.ceil(num_leads / cols)
#         fig, axes = plt.subplots(rows, cols, sharex=True, figsize=(15, 8))
#         for idx, lead in enumerate(leads):
#             ax = axes.flat[idx]
#             signal = glove.lead_signals[lead]
#             times = np.arange(signal.size) / glove.sampling_rate
#             ax.plot(times, signal)
#             ax.set_title(f"Lead {lead}")
#             ax.set_xlabel("Time (s)")
#             ax.set_ylabel("Amplitude")
#         for idx in range(len(leads), rows * cols):
#             fig.delaxes(axes.flat[idx])
#         plt.tight_layout()
#         plt.show()
#     except Exception:
#         pass
