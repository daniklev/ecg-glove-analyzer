from typing import Dict, List
import numpy as np
from numpy.typing import NDArray


class ECGPacketDecoder:
    PC_ADDR = 0x80  # Destination = PC
    UNIT_ADDR = 0x17  # Source = ECG unit
    TYPE_DATA = 0x00  # Transfer type = Data
    DATA_SUBTYPE = 0x51  # header[5] == 0x51 ⇒ 81-byte ECG payload
    HEADER_LEN = 7  # bytes in the header

    def __init__(self):
        # 8 channels: 0→Lead I, 1→Lead III, …, 7→Lead V6
        self.leads: Dict[int, List[int]] = {ch: [] for ch in range(8)}

    def reset(self):
        """Clear out any previously-decoded samples."""
        self.leads: Dict[int, List[int]] = {ch: [] for ch in range(8)}

    def decode(self, buf: bytes) -> Dict[str, NDArray[np.float64]]:
        """Decode ECG data and return lead signals."""
        self.reset()  # Clear previous data
        self.feed(buf)  # Process the data
        return self.get_leads()  # Return the processed leads

    def feed(self, buf: bytes):
        size = len(buf)
        i = 0
        packetType = 0

        while i < size:
            # 1) Frame-sync: find 0x80 (PC_ADDR)
            while i < size and buf[i] != self.PC_ADDR:
                i += 1
                if i > size - 11:  # fewer than 11 bytes left ⇒ stop
                    return
            if i > size - 11:
                return

            # 2) If this looks like a data header, verify its checksum
            if buf[i + 1] == self.UNIT_ADDR and buf[i + 2] == self.TYPE_DATA:
                hdr_sum = sum(buf[i + k] for k in range(self.HEADER_LEN)) & 0xFF
                if hdr_sum == 0:
                    packetType = buf[i + 5]
                else:
                    packetType = 0
            # else: packetType stays whatever it was

            # 3) Dispatch on packetType
            if packetType == self.DATA_SUBTYPE:
                # move past header
                start = i + self.HEADER_LEN

                # only decode if the full payload+cs fits
                if start + packetType < size:
                    block = buf[start : start + packetType]
                    # block includes (data + 1-byte checksum)
                    if sum(block) & 0xFF == 0:
                        data_len = packetType - 1  # drop the final cs byte
                        # must be whole 16-byte groups (8 leads×2 bytes)
                        if data_len % 16 == 0:
                            # decode exactly like computeLead does
                            for base in range(start, start + data_len, 16):
                                for ch in range(8):
                                    lo = buf[base + ch * 2]
                                    hi = buf[base + ch * 2 + 1]
                                    val = (hi << 8) | lo
                                    # two's-complement fix
                                    if val & 0x8000:
                                        val -= 0x10000
                                    self.leads[ch].append(val)

                # advance past header + entire payload
                i += self.HEADER_LEN + packetType

            elif packetType == 3:
                # fault packet
                i += 1

            else:
                # any other packetType ⇒ skip one byte
                i += 1

    def get_leads(self) -> Dict[str, NDArray[np.float64]]:
        """Convert raw channel data to standard ECG leads."""
        # — map raw channels → leads, truncate to shortest —
        raw = {
            "I": np.array(self.leads[0], dtype=float),
            "III": np.array(self.leads[1], dtype=float),
            "V1": np.array(self.leads[2], dtype=float),
            "V2": np.array(self.leads[3], dtype=float),
            "V3": np.array(self.leads[4], dtype=float),
            "V4": np.array(self.leads[5], dtype=float),
            "V5": np.array(self.leads[6], dtype=float),
            "V6": np.array(self.leads[7], dtype=float),
        }
        min_len = min(len(arr) for arr in raw.values()) if raw["I"].size > 0 else 0
        for lead in raw:
            raw[lead] = raw[lead][:min_len]

        # — derive remaining standard leads —
        leads = dict(raw)
        if min_len > 0:
            leads["II"] = leads["I"] + leads["III"]
            leads["aVR"] = -(leads["I"] + leads["II"]) / 2
            leads["aVL"] = leads["I"] - leads["II"] / 2
            leads["aVF"] = leads["II"] - leads["I"] / 2

        return leads