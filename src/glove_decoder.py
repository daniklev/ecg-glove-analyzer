from typing import Dict
import numpy as np
from numpy.typing import NDArray


class ECGPacketDecoder:
    """Decoder for ECG glove binary protocol (ES/ET v2.0.1.14)."""

    # Protocol constants (v2.0.1.14)
    PC_ADDRESS = 0x80  # Destination = PC  [oai_citation:1‡010 ES-ECG and PG ET Elec Unit_v 2.0.1.14_Communication Protocol_r1.0.pdf](file-service://file-7tXiGhWAh6BeznyzBo2HUH)
    DEVICE_ADDRESS = 0x17  # Source = ECG unit (500 Hz)  [oai_citation:2‡010 ES-ECG and PG ET Elec Unit_v 2.0.1.14_Communication Protocol_r1.0.pdf](file-service://file-7tXiGhWAh6BeznyzBo2HUH)
    DATA_TRANSFER_TYPE = 0x00  # Transfer Type = Data  [oai_citation:3‡010 ES-ECG and PG ET Elec Unit_v 2.0.1.14_Communication Protocol_r1.0.pdf](file-service://file-7tXiGhWAh6BeznyzBo2HUH)

    SYNC_BYTE = PC_ADDRESS  # Byte 0 of header
    HEADER_LENGTH = 7  # Bytes 0–6: header + checksum  [oai_citation:4‡010 ES-ECG and PG ET Elec Unit_v 2.0.1.14_Communication Protocol_r1.0.pdf](file-service://file-7tXiGhWAh6BeznyzBo2HUH)
    FRAME_SIZE = 16  # 8 channels × 2 bytes per sample

    def __init__(self) -> None:
        # Raw buffers for eight channels
        self.channels: Dict[int, list[int]] = {i: [] for i in range(8)}

    @staticmethod
    def _verify_header_checksum(header: bytes) -> bool:
        # Sum(header_bytes + checksum) mod 256 == 0
        return sum(header) & 0xFF == 0

    @staticmethod
    def _verify_payload_checksum(payload: bytes) -> bool:
        # Sum(data_bytes + checksum) mod 256 == 0
        return sum(payload) & 0xFF == 0

    def decode(self, data: bytes) -> Dict[str, NDArray[np.float64]]:
        buf = bytearray(data)
        n = len(buf)
        i = 0

        while i + self.HEADER_LENGTH <= n:
            # 1) Find packet start
            if buf[i] != self.SYNC_BYTE:
                i += 1
                continue

            # 2) Slice out header
            header = bytes(buf[i : i + self.HEADER_LENGTH])
            dest = header[0]
            src = header[1]
            pkt_type = header[2]
            data_len = header[5]

            # 3) Validate header
            if not (
                self._verify_header_checksum(header)
                and dest == self.PC_ADDRESS
                and src == self.DEVICE_ADDRESS
                and pkt_type == self.DATA_TRANSFER_TYPE
            ):
                i += 1
                continue

            # 4) Make sure full payload is present
            end = i + self.HEADER_LENGTH + data_len
            if end > n:
                break  # incomplete packet at buffer end

            payload = bytes(buf[i + self.HEADER_LENGTH : end])
            # 5) Validate payload checksum
            if len(payload) < 1 or not self._verify_payload_checksum(payload):
                i = end
                continue

            # 6) Strip off the checksum byte
            frame_data = payload[:-1]

            # 7) Decode each 16-byte frame into signed 16-bit samples
            for offset in range(0, len(frame_data), self.FRAME_SIZE):
                frame = frame_data[offset : offset + self.FRAME_SIZE]
                if len(frame) < self.FRAME_SIZE:
                    break
                for ch in range(8):
                    lsb = frame[2 * ch]
                    msb = frame[2 * ch + 1]
                    val = (msb << 8) | (lsb & 0xFF)
                    # sign-extend 16-bit
                    if val >= 0x8000:
                        val -= 0x10000
                    self.channels[ch].append(val)

            i = end  # move to next packet

        # 8) Map raw channels → leads
        #    (Channel numbering as per Table 8: Ch1=I, Ch2=III, Ch3=V1 … Ch8=V6)
        raw = {
            "I": np.array(self.channels[0], dtype=float),
            "III": np.array(self.channels[1], dtype=float),
            "V1": np.array(self.channels[2], dtype=float),
            "V2": np.array(self.channels[3], dtype=float),
            "V3": np.array(self.channels[4], dtype=float),
            "V4": np.array(self.channels[5], dtype=float),
            "V5": np.array(self.channels[6], dtype=float),
            "V6": np.array(self.channels[7], dtype=float),
        }

        # 9) Truncate to the shortest lead length
        min_len = min(arr.size for arr in raw.values())
        for lead in raw:
            raw[lead] = raw[lead][:min_len]

        # 10) Derive standard limb and augmented leads
        leads: Dict[str, NDArray[np.float64]] = dict(raw)
        if min_len > 0:
            leads["II"] = leads["I"] + leads["III"]
            leads["aVR"] = -(leads["I"] + leads["II"]) / 2
            leads["aVL"] = leads["I"] - leads["II"] / 2
            leads["aVF"] = leads["II"] - leads["I"] / 2

        return leads
