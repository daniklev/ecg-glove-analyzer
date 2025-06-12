"""ECG Glove data decoder.

This module provides functionality to decode binary data from the ECG glove device.
It handles packet framing, checksum verification, and extraction of ECG channel data.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray


@dataclass
class GlovePacket:
    """Represents a decoded glove data packet."""

    packet_type: int
    payload: bytes
    checksum_valid: bool


class ECGPacketDecoder:
    """Decoder for ECG glove data packets."""

    # Packet constants
    SYNC_BYTE = 0x80
    HEADER_SIZE = 7
    ECG_PACKET_TYPE = 0x51
    FAULT_PACKET_TYPE = 0x03
    ECG_PAYLOAD_SIZE = 81  # 80 bytes data + 1 byte checksum
    FAULT_PAYLOAD_SIZE = 10

    def __init__(self) -> None:
        """Initialize the decoder."""
        self.buffer = bytearray()
        self.channels: Dict[int, List[int]] = {i: [] for i in range(8)}

    def _verify_checksum(self, data: bytes) -> bool:
        """Verify packet checksum (sum of all bytes should be 0 mod 256)."""
        return sum(data) & 0xFF == 0

    def _decode_header(self, header: bytes) -> Optional[int]:
        """Decode packet header and return packet type if valid."""
        if (
            len(header) == self.HEADER_SIZE
            and header[0] == self.SYNC_BYTE
            and header[1] == 0x17
            and header[2] == 0x00
            and self._verify_checksum(header)
        ):
            return header[5]  # Packet type
        return None

    def _decode_ecg_frame(self, frame: bytes) -> None:
        """Decode a single ECG frame (16 bytes) containing 8 channels."""
        for ch in range(8):
            # Get LSB and MSB
            lsb = frame[2 * ch]
            msb = frame[2 * ch + 1]
            # Combine bytes to form 16-bit value
            value = (msb << 8) | (lsb & 0xFF)
            # Convert to signed 16-bit
            if value > 32767:
                value -= 65536
            self.channels[ch].append(value)

    def _decode_ecg_packet(self, payload: bytes) -> bool:
        """Decode ECG data packet payload."""
        if not self._verify_checksum(payload):
            return False

        # Process 5 frames of 8 channels (16 bytes per frame)
        data = payload[:-1]  # Skip checksum byte
        for frame_start in range(0, len(data), 16):
            frame = data[frame_start : frame_start + 16]
            if len(frame) == 16:  # Ensure complete frame
                self._decode_ecg_frame(frame)
        return True

    def _find_next_packet(
        self, data: bytes, start: int = 0
    ) -> Tuple[int, Optional[GlovePacket]]:
        """Find and decode the next packet in the data stream."""
        size = len(data)
        i = start

        while i < size:
            # Look for sync byte
            if data[i] != self.SYNC_BYTE:
                i += 1
                continue

            # Check if we have enough bytes for a header
            if i + self.HEADER_SIZE > size:
                break

            # Try to decode header
            header = data[i : i + self.HEADER_SIZE]
            packet_type = self._decode_header(header)
            if packet_type is None:
                i += 1
                continue

            # Handle different packet types
            payload_size = 0
            if packet_type == self.ECG_PACKET_TYPE:
                payload_size = self.ECG_PAYLOAD_SIZE
            elif packet_type == self.FAULT_PACKET_TYPE:
                payload_size = self.FAULT_PAYLOAD_SIZE

            # Check if we have complete packet
            packet_end = i + self.HEADER_SIZE + payload_size
            if packet_end > size:
                break

            # Extract and verify payload
            payload = data[i + self.HEADER_SIZE : packet_end]
            checksum_valid = self._verify_checksum(payload)

            return packet_end, GlovePacket(
                packet_type=packet_type, payload=payload, checksum_valid=checksum_valid
            )

            i = packet_end

        return i, None

    def decode(self, data: bytes) -> Dict[str, NDArray[np.float64]]:
        """Decode ECG data from bytes and return channel data.

        Args:
            data: Raw bytes from the ECG glove device

        Returns:
            Dictionary mapping lead names to numpy arrays of signal values

        Raises:
            ValueError: If no valid ECG data is found
        """
        # Reset channel buffers
        self.channels = {i: [] for i in range(8)}

        # Process all packets
        pos = 0
        while pos < len(data):
            next_pos, packet = self._find_next_packet(data, pos)
            if packet is None:
                break

            if packet.packet_type == self.ECG_PACKET_TYPE and packet.checksum_valid:
                self._decode_ecg_packet(packet.payload)

            pos = next_pos

        # Convert channel lists to numpy arrays
        if not any(self.channels.values()):
            raise ValueError("No valid ECG data found in input")

        # Map channels to lead names and convert to numpy arrays
        leads = {
            "I": np.array(self.channels[0], dtype=np.float64),
            "II": np.array(self.channels[1], dtype=np.float64),
            "V1": np.array(self.channels[2], dtype=np.float64),
            "V2": np.array(self.channels[3], dtype=np.float64),
            "V3": np.array(self.channels[4], dtype=np.float64),
            "V4": np.array(self.channels[5], dtype=np.float64),
            "V5": np.array(self.channels[6], dtype=np.float64),
            "V6": np.array(self.channels[7], dtype=np.float64),
        }

        # Ensure all leads have the same length
        min_length = min(len(arr) for arr in leads.values())
        leads = {name: arr[:min_length] for name, arr in leads.items()}

        # Calculate derived leads
        if len(leads["I"]) > 0 and len(leads["II"]) > 0:
            leads["III"] = leads["II"] - leads["I"]
            leads["aVR"] = -(leads["I"] + leads["II"]) / 2
            leads["aVL"] = leads["I"] - leads["II"] / 2
            leads["aVF"] = leads["II"] - leads["I"] / 2

        return leads
