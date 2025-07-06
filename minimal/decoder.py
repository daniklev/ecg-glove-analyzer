import base64
from typing import Dict
import numpy as np
from numpy.typing import NDArray


class ECGPacketDecoder:
    """
    Decode raw ECG packet bytes into 12 standard lead signals.
    """

    PC_ADDR = 0x80
    UNIT_ADDR = 0x17
    TYPE_DATA = 0x00
    DATA_SUBTYPE = 0x51
    HEADER_LEN = 7

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Clear any previously-decoded samples."""
        self._raw_leads = {ch: [] for ch in range(8)}

    def decode_bytes(self, buf: bytes) -> Dict[str, NDArray[np.float64]]:
        """
        Decode a bytes buffer of ECG packets into a dict of leads.
        Returns 12 NumPy arrays: I, II, III, aVR, aVL, aVF, and V1–V6.
        """
        self.reset()
        self._feed(buf)
        return self._get_leads()

    def decode_base64(self, b64_str: str) -> Dict[str, NDArray[np.float64]]:
        """
        Decode a base64-encoded ECG packet string.
        """
        raw = base64.b64decode(b64_str)
        return self.decode_bytes(raw)

    def _feed(self, buf: bytes) -> None:
        size = len(buf)
        i = 0
        packet_type = 0

        while i < size:
            # Frame sync: look for PC_ADDR
            while i < size and buf[i] != self.PC_ADDR:
                i += 1
                if i > size - 11:
                    return
            if i > size - 11:
                return

            # Check header (unit addr, type)
            if buf[i + 1] == self.UNIT_ADDR and buf[i + 2] == self.TYPE_DATA:
                hdr_sum = sum(buf[i + k] for k in range(self.HEADER_LEN)) & 0xFF
                packet_type = buf[i + 5] if hdr_sum == 0 else 0

            # Decode ECG data packets
            if packet_type == self.DATA_SUBTYPE:
                start = i + self.HEADER_LEN
                if start + packet_type < size:
                    block = buf[start : start + packet_type]
                    if sum(block) & 0xFF == 0:
                        data_len = packet_type - 1  # drop checksum
                        if data_len % 16 == 0:
                            for base in range(start, start + data_len, 16):
                                for ch in range(8):
                                    lo = buf[base + ch * 2]
                                    hi = buf[base + ch * 2 + 1]
                                    val = (hi << 8) | lo
                                    if val & 0x8000:
                                        val -= 0x10000
                                    self._raw_leads[ch].append(val)
                i += self.HEADER_LEN + packet_type

            elif packet_type == 3:
                # Fault packet: skip
                i += 1
            else:
                # Other packet types: advance one byte
                i += 1

    def _get_leads(self) -> Dict[str, NDArray[np.float64]]:
        """
        Convert 8 decoded channel arrays into 12 standard ECG leads.
        """
        raw = {
            "I": np.array(self._raw_leads[0], dtype=float),
            "III": np.array(self._raw_leads[1], dtype=float),
            "V1": np.array(self._raw_leads[2], dtype=float),
            "V2": np.array(self._raw_leads[3], dtype=float),
            "V3": np.array(self._raw_leads[4], dtype=float),
            "V4": np.array(self._raw_leads[5], dtype=float),
            "V5": np.array(self._raw_leads[6], dtype=float),
            "V6": np.array(self._raw_leads[7], dtype=float),
        }
        # If no data, return the eight channel arrays
        if raw["I"].size == 0:
            return raw

        # Truncate to shortest channel
        min_len = min(arr.size for arr in raw.values())
        for key in raw:
            raw[key] = raw[key][:min_len]

        # Derive the four additional leads
        leads = dict(raw)
        leads["II"] = leads["I"] + leads["III"]
        leads["aVR"] = -(leads["I"] + leads["II"]) / 2
        leads["aVL"] = leads["I"] - leads["II"] / 2
        leads["aVF"] = leads["II"] - leads["I"] / 2

        return leads
