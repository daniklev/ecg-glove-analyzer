import numpy as np
from collections import deque
from typing import Dict, Optional, Any
import logging

from ECGFiltersV2 import (
    HiPassFilter,
    HPFilterType,
    NotchEcgFilter,
    MorphologyFilter,
    FilterConfig,
    VectorizedECGProcessor,
)
from glove_decoder import ECGPacketDecoder
from ecg_processor import EcgQualityProcessor

BAD_VALUE = 9999


def spike_remove(data: np.ndarray, threshold: int = 1000) -> np.ndarray:
    """
    Simple spike removal: any sample that differs from both neighbors
    by more than `threshold` is replaced by their average.
    """
    out = data.copy()
    for i in range(1, len(data) - 1):
        if (
            abs(int(data[i]) - int(data[i - 1])) > threshold
            and abs(int(data[i]) - int(data[i + 1])) > threshold
        ):
            out[i] = np.int16((int(data[i - 1]) + int(data[i + 1])) // 2)
    return out


class ECGLeadProcessor:
    """Enhanced ECG lead processor with integrated filtering and quality assessment."""

    def __init__(
        self,
        sample_rate: int = 500,
        max_raw_size: int = 15000,
        gain_1mv: float = 200.0,
        use_vectorized_processing: bool = True,
    ):
        self.sample_rate = sample_rate
        self.max_raw_size = max_raw_size
        self.gain_1mv = gain_1mv
        self.use_vectorized_processing = use_vectorized_processing

        # Raw data buffer
        self.row_lead: deque[int] = deque()
        self.count_baseline = 0

        # Filter configuration
        self.power_line_freq = 0  # 0=no notch, 50 or 60 Hz
        self.filter_type = 0  # 0=morphology, 1=HP015
        self.spike_removal = True
        self.ecg_range = 1.0
        self.rec_time = 0

        # Initialize processors
        self.decoder = ECGPacketDecoder()
        self.quality_processor = EcgQualityProcessor()

        if use_vectorized_processing:
            self.vectorized_processor = VectorizedECGProcessor(sample_rate)

        # Filter configuration
        self.filter_config = FilterConfig()
        self._configure_default_filters()

        # Logging
        self.logger = logging.getLogger(__name__)

    def _configure_default_filters(self):
        """Configure default filter settings for optimal ECG processing."""
        self.filter_config.enable_hpf = True
        self.filter_config.hpf_type = HPFilterType.HP015
        self.filter_config.enable_notch = True
        self.filter_config.notch_frequencies = [60]  # Default for US
        self.filter_config.enable_morphology = True
        self.filter_config.spike_removal = True

    def set_power_line_freq(self, freq: int):
        """Set power line frequency for notch filtering."""
        self.power_line_freq = freq
        self.filter_config.notch_frequencies = [freq] if freq != 0 else []

    def set_filter_type(self, filter_type: int):
        """Set filter type: 0=morphology, 1=HP015."""
        self.filter_type = filter_type
        if filter_type == 0:
            self.filter_config.enable_morphology = True
            self.filter_config.hpf_type = HPFilterType.HP015
        else:
            self.filter_config.enable_morphology = False
            self.filter_config.hpf_type = HPFilterType.HP015

    def set_spike_removal(self, flag: bool):
        """Enable/disable spike removal."""
        self.spike_removal = flag
        self.filter_config.spike_removal = flag

    def set_rec_time(self, seconds: int):
        """Set recording time in seconds."""
        self.rec_time = seconds

    def add_new_val(self, new_val: int):
        """Add a new sample to the buffer."""
        # Maintain fixed buffer length
        if len(self.row_lead) >= self.max_raw_size:
            self.row_lead.popleft()
        self.row_lead.append(new_val)

        # Update baseline counter
        if abs(new_val) > self.get_trash_level():
            self.count_baseline = 0
        else:
            self.count_baseline += 1

    def get_trash_level(self) -> int:
        """Get threshold level for baseline detection."""
        return int(round(200 * self.ecg_range))

    def process_packet_data(
        self, packet_data: bytes
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Process raw packet data and return filtered ECG leads.

        Args:
            packet_data: Raw packet data from ECG glove

        Returns:
            Dictionary of filtered ECG leads or None if processing failed
        """
        try:
            # Decode packet data into leads
            leads = self.decoder.decode(packet_data)

            if not leads or len(next(iter(leads.values()))) == 0:
                self.logger.warning("No valid lead data decoded from packet")
                return None

            # Process leads with filtering
            return self._process_leads(leads)

        except Exception as e:
            self.logger.error(f"Error processing packet data: {e}")
            return None

    def _process_leads(self, leads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Process leads using vectorized operations for better performance."""
        config = {
            "spike_removal": self.filter_config.spike_removal,
            "filters": self.filter_config.notch_frequencies,
            "hp_filter_type": self.filter_config.hpf_type.value,
        }

        # Use vectorized processor
        processed_leads = self.vectorized_processor.process_leads_batch(leads, config)

        return processed_leads

    def _process_leads_sequential(
        self, leads: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Process leads sequentially using individual filters."""
        processed_leads = {}

        for lead_name, lead_data in leads.items():
            try:
                processed_leads[lead_name] = self._process_single_lead(lead_data)
            except Exception as e:
                self.logger.error(f"Error processing lead {lead_name}: {e}")
                processed_leads[lead_name] = lead_data  # Return unprocessed data

        return processed_leads

    def _process_single_lead(self, lead_data: np.ndarray) -> np.ndarray:
        """Process a single lead with the configured filters."""
        # Convert to int16 and invert polarity if needed
        ftr_data = (-lead_data).astype(np.int16)

        # 1) Notch filter (50/60 Hz)
        if self.power_line_freq != 0:
            nf = NotchEcgFilter(self.power_line_freq)
            for i, v in enumerate(ftr_data):
                if v != BAD_VALUE:
                    ftr_data[i] = np.int16(round(nf.get_new_val(int(v))))

        # 2) Morphology HPF or 0.15 Hz HPF
        if self.filter_type == 0:
            morf = MorphologyFilter()
            for i, v in enumerate(ftr_data):
                if v != BAD_VALUE:
                    ftr_data[i] = np.int16(morf.compute_hpf(int(v)))
        else:
            hpf = HiPassFilter(HPFilterType.HP015)
            for i, v in enumerate(ftr_data):
                if v != BAD_VALUE:
                    ftr_data[i] = np.int16(hpf.get_new_val(int(v)))

        # 3) Spike removal (only for morphology)
        if self.filter_type == 0 and self.spike_removal:
            ftr_data = spike_remove(ftr_data)

        return ftr_data

    def get_saved_filtered_data(self) -> list[int]:
        """
        Legacy method for compatibility with original ECGGloveV2.
        Get filtered data from the internal buffer.
        """
        # Prepare raw buffer
        raw = list(self.row_lead)
        n_samples = min(len(raw), self.rec_time * self.sample_rate)
        start_idx = max(0, len(raw) - n_samples - 1)

        # Invert polarity
        rec_data = np.array(
            [-raw[start_idx + i] for i in range(n_samples)],
            dtype=np.int16,
        )

        # Process the data
        processed_data = self._process_single_lead(rec_data)
        return processed_data.tolist()

    def assess_signal_quality(
        self, leads: Dict[str, np.ndarray], context: str = "clinical"
    ) -> Dict[str, Any]:
        """
        Assess the quality of ECG signals.

        Args:
            leads: Dictionary of ECG lead signals
            context: "clinical" or "ambulance" for different weighting schemes

        Returns:
            Dictionary containing quality metrics
        """
        try:
            # Use appropriate weights based on context
            if context == "ambulance":
                _ = self.quality_processor.AMBULANCE_LEAD_WEIGHTS
            else:
                _ = self.quality_processor.CLINICAL_LEAD_WEIGHTS

            # Calculate quality metrics (this would need to be implemented in the quality processor)
            quality_metrics = {
                "overall_quality": 0.8,  # Placeholder
                "lead_qualities": {lead: 0.8 for lead in leads.keys()},
                "snr_db": 20.0,  # Placeholder
                "baseline_stability": 0.9,  # Placeholder
            }

            return quality_metrics

        except Exception as e:
            self.logger.error(f"Error assessing signal quality: {e}")
            return {
                "overall_quality": 0.0,
                "lead_qualities": {lead: 0.0 for lead in leads.keys()},
                "snr_db": 0.0,
                "baseline_stability": 0.0,
            }

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about the current processing state."""
        return {
            "buffer_size": len(self.row_lead),
            "max_buffer_size": self.max_raw_size,
            "buffer_utilization": len(self.row_lead) / self.max_raw_size,
            "baseline_count": self.count_baseline,
            "trash_level": self.get_trash_level(),
            "sample_rate": self.sample_rate,
            "filter_config": {
                "power_line_freq": self.power_line_freq,
                "filter_type": self.filter_type,
                "spike_removal": self.spike_removal,
                "vectorized_processing": self.use_vectorized_processing,
            },
        }


class ECGGloveAnalyzer:
    """
    Main analyzer class that orchestrates the entire ECG processing pipeline.
    This connects the decoder, processor, and quality assessment components.
    """

    def __init__(
        self,
        sample_rate: int = 500,
        max_raw_size: int = 15000,
        enable_real_time: bool = True,
    ):
        self.sample_rate = sample_rate
        self.enable_real_time = enable_real_time

        # Initialize the lead processor
        self.lead_processor = ECGLeadProcessor(
            sample_rate=sample_rate,
            max_raw_size=max_raw_size,
            use_vectorized_processing=True,
        )

        # Processing state
        self.is_recording = False
        self.total_packets_processed = 0
        self.last_processing_time = 0.0

        # Results storage
        self.latest_leads = {}
        self.latest_quality_metrics = {}

        # Logging
        self.logger = logging.getLogger(__name__)

    def configure_filters(
        self,
        power_line_freq: int = 60,
        filter_type: int = 0,
        spike_removal: bool = True,
    ):
        """Configure the filtering parameters."""
        self.lead_processor.set_power_line_freq(power_line_freq)
        self.lead_processor.set_filter_type(filter_type)
        self.lead_processor.set_spike_removal(spike_removal)

    def start_recording(self, duration_seconds: int = 10):
        """Start ECG recording session."""
        self.lead_processor.set_rec_time(duration_seconds)
        self.is_recording = True
        self.total_packets_processed = 0
        self.logger.info(f"Started ECG recording for {duration_seconds} seconds")

    def stop_recording(self):
        """Stop ECG recording session."""
        self.is_recording = False
        self.logger.info(
            f"Stopped ECG recording after processing {self.total_packets_processed} packets"
        )

    def process_realtime_packet(self, packet_data: bytes) -> Optional[Dict[str, Any]]:
        """
        Process a real-time packet and return results.

        Args:
            packet_data: Raw packet data from ECG device

        Returns:
            Dictionary containing processed leads and quality metrics
        """
        if not self.is_recording:
            return None

        import time

        start_time = time.time()

        try:
            # Process the packet
            leads = self.lead_processor.process_packet_data(packet_data)

            if leads is None:
                return None

            # Store latest results
            self.latest_leads = leads

            # Assess quality
            self.latest_quality_metrics = self.lead_processor.assess_signal_quality(
                leads
            )

            # Update stats
            self.total_packets_processed += 1
            self.last_processing_time = time.time() - start_time

            # Return comprehensive results
            result = {
                "leads": leads,
                "quality_metrics": self.latest_quality_metrics,
                "processing_stats": self.lead_processor.get_processing_stats(),
                "timestamp": time.time(),
                "packet_number": self.total_packets_processed,
                "processing_time_ms": self.last_processing_time * 1000,
            }

            return result

        except Exception as e:
            self.logger.error(f"Error processing real-time packet: {e}")
            return None

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get a summary of the current analysis session."""
        return {
            "session_stats": {
                "is_recording": self.is_recording,
                "total_packets_processed": self.total_packets_processed,
                "last_processing_time_ms": self.last_processing_time * 1000,
                "sample_rate": self.sample_rate,
            },
            "latest_leads": list(self.latest_leads.keys()) if self.latest_leads else [],
            "latest_quality": self.latest_quality_metrics,
            "processor_stats": self.lead_processor.get_processing_stats(),
        }


# Legacy compatibility class
class CommwellEcgLead(ECGLeadProcessor):
    """
    Legacy compatibility class that maintains the original interface
    while using the enhanced processing capabilities.
    """

    pass
