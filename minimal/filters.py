from numpy.typing import NDArray
import numpy as np
from typing import List
from scipy.signal import sosfilt, iirnotch, butter, tf2sos, sosfiltfilt, iirfilter
from collections import deque
from numba import njit


class FilterConfig:
    """
    Configuration for ECG filter chain following Commwell_Ecg_Lead.java.
    filter_type: 0=morphology, 1=HP015, 2=HP005, 3=HP05
    notch_frequencies: list of powerline frequencies to notch (e.g., [50,100])
    spike_removal: secondary spike removal (Morphology)
    enable_baseline_correction: apply recursive baseline filter
    """

    def __init__(
        self,
        sampling_rate: int = 500,
        notch_frequencies: List[int] = [],
        filter_type: int = 0,
        spike_removal: bool = False,
        baseline_correction: bool = False,
        human_filter: bool = False,
    ):
        self.sampling_rate = sampling_rate
        self.notch_frequencies = notch_frequencies or []
        self.filter_type = filter_type
        self.spike_removal = spike_removal
        self.baseline_correction = baseline_correction
        self.human_filter = human_filter


class HiPassFilter:
    _COEFFS = {
        1: {"HP0": -0.9963349287, "HP1": 1.9963282000, "GAIN": 1.001837588},  # HP015
        2: {"HP0": -0.9987734371, "HP1": 1.9987726844, "GAIN": 1.00061384},  # HP005
        3: {"HP0": -0.9878018507, "HP1": 1.9877269954, "GAIN": 1.006155446},  # HP05
    }

    def __init__(self, filter_type: int):
        coeffs = self._COEFFS.get(filter_type)
        if coeffs is None:
            raise ValueError(f"Unsupported filter_type {filter_type}")
        self.HP0 = coeffs["HP0"]
        self.HP1 = coeffs["HP1"]
        self.GAIN = coeffs["GAIN"]
        self.xv = [0.0, 0.0, 0.0]
        self.yv = [0.0, 0.0, 0.0]

    def filter_sample(self, val: float) -> float:
        self.xv[0], self.xv[1], self.xv[2] = self.xv[1], self.xv[2], val / self.GAIN
        self.yv[0], self.yv[1] = self.yv[1], self.yv[2]
        self.yv[2] = (
            (self.xv[0] + self.xv[2])
            - 2 * self.xv[1]
            + self.HP0 * self.yv[0]
            + self.HP1 * self.yv[1]
        )
        return self.yv[2]


class Buffer:
    def __init__(self, size: int):
        self.buf = deque(maxlen=size)

    def add(self, x: float):
        self.buf.append(x)

    def is_full(self) -> bool:
        return len(self.buf) == self.buf.maxlen

    def median(self) -> float:
        vals = sorted(self.buf)
        return vals[len(vals) // 2]


class MorphologyFilter:
    def __init__(self, window_size: int = 8):
        self.buffer = Buffer(window_size)
        self.prev_out = 0.0

    def filter_sample(self, data: float) -> float:
        self.buffer.add(data)
        if not self.buffer.is_full():
            return data
        med = self.buffer.median()
        out = data - med
        sm = 0.7 * out + 0.3 * self.prev_out
        self.prev_out = sm
        return sm


class NotchEcgFilter:
    _AR_COEFS: dict = {50: [], 60: [], 100: [], 120: []}
    # Кеш для уже рассчитанных SOS-коэффициентов, ключ = частота нотча
    _SOS: dict[int, np.ndarray] = {}

    def __init__(self, freq: int):       
        self.ar = self._AR_COEFS.get(freq, []).copy()
        self.N = len(self.ar)
        self.buf = deque([0.0] * self.N, maxlen=self.N)
        Q: float = 30.0
        self._SOS[60] = iirnotch(w0=60, Q=Q, fs=self.sampling_rate, output='sos')
         # Берём готовые коэффициенты из кеша
        self.sos = self._SOS[60]
        # Инициализируем внутреннее состояние sosfilt нулями
        self.zi = np.zeros((self.sos.shape[0], 2))

    def filter_sample(self, val: float) -> float:

        if self.N == 0:
            return val
        
      #  self.buf.append(val)
      #  res = 0.0
      #  for coef, past in zip(self.ar, reversed(self.buf)):
      #      res += coef * past
      #  return res
      # sosfilt требует массив, поэтому упаковываем x в список длиной 1
        y, self.zi = sosfilt(self.sos, [val], zi=self.zi)
        # Возвращаем скаляр (первый и единственный элемент)
        return float(y[0])


class MultiNotchFilter:
    def __init__(self, freqs: List[int]):
        self.filters = [NotchEcgFilter(f) for f in freqs]

    def filter_sample(self, val: float) -> float:
        out = val
        for f in self.filters:
            out = f.filter_sample(out)
        return out


class BaselineFilter:
    def __init__(self, cutoff: float = 0.5, fs: int = 500):
        rc = 1.0 / (2 * np.pi * cutoff)
        dt = 1.0 / fs
        self.alpha = rc / (rc + dt)
        self.prev_in = 0.0
        self.prev_out = 0.0

    def filter_sample(self, val: float) -> float:
        out = self.alpha * (self.prev_out + val - self.prev_in)
        self.prev_in = val
        self.prev_out = out
        return out


@njit
def fast_morphology_filter(arr: np.ndarray) -> np.ndarray:
    n = arr.shape[0]
    out = np.empty_like(arr)
    for i in range(n):
        lo = max(0, i - 2)
        hi = min(n, i + 3)
        # simple median
        window = arr[lo:hi]
        out[i] = np.median(window)
    return out


# Cache SOS coefficients to avoid recomputation
_sos_notch_cache = {}
_sos_hpf_cache = {}


def _sos_notch_coeffOrig(freq: int, fs: int):  #  0.977
    key = (freq, fs)
    if key not in _sos_notch_cache:
        w0 = freq / (fs / 2)  
        Q = 10 if freq == 60 else 30
        # b, a = iirnotch(w0, Q=30)
        b, a = iirnotch(freq, Q, fs)
        _sos_notch_cache[key] = tf2sos(b, a)
    return _sos_notch_cache[key]

def _sos_notch_coeff(freq: int, fs: int):   # 0.966
    key = (freq, fs)
    if key not in _sos_notch_cache:
        bw = 4.0  # ширина выреза ±2 Гц
        low = (freq - bw/2) / (fs/2)
        high = (freq + bw/2) / (fs/2)
        _sos_notch_cache[key] = butter(4, [low, high], btype="bandstop", output="sos")
    return _sos_notch_cache[key]


def _sos_notch_coeffAlex2(freq: int, fs: int): # 0.952
    key = (freq, fs)
    if key not in _sos_notch_cache:
        bw   = 2.0          # ширина полосы ±1 Гц
        low  = (freq - bw) / (fs / 2)
        high = (freq + bw) / (fs / 2)
        sos = iirfilter(
            N=4,
            Wn=[low, high],
            btype="bandstop",
            ftype="ellip",
            rp=1,             # 1 dB пульсации в полосе
            rs=60,            # 60 dB затухания в стоп-полосе
            output="sos"
        )
        _sos_notch_cache[key] = sos
    return _sos_notch_cache[key]

def _sos_hpf_coeff(filter_type: int, fs: int):
    key = (filter_type, fs)
    if key not in _sos_hpf_cache:
        cutoff_map = {1: 0.15, 2: 0.05, 3: 0.5}
        cutoff = cutoff_map[filter_type]
        _sos_hpf_cache[key] = butter(1, cutoff, btype="highpass", fs=fs, output="sos")
    return _sos_hpf_cache[key]


def apply_filters(signal: np.ndarray, config: FilterConfig) -> NDArray[np.float64]:
    out = signal.astype(float)

    # 1) Primary filter
    if config.filter_type == 0:
        # morphology
        out = fast_morphology_filter(np.asarray(out, dtype=np.float64))
    elif config.filter_type in (1, 2, 3):
        sos = _sos_hpf_coeff(config.filter_type, config.sampling_rate)
        out = sosfilt(sos, out)

    # 2) Notch filtering (vectorized SOS)
    if config.notch_frequencies:
        for freq in config.notch_frequencies:
            sos = _sos_notch_coeff(freq, config.sampling_rate)
            # out = sosfilt(sos, out)
            out = sosfiltfilt(sos, out)

    # 3) Use High-Pass Filter with 0 Hz cutoff to 40 Hz
    if config.human_filter:
        # Human filter: 0-40 Hz
        sos = butter(2, (0.05, 40), btype="bandpass", fs=config.sampling_rate, output="sos")
        out = sosfilt(sos, out)

    # 4) Secondary spike removal
    if config.spike_removal:
        out = fast_morphology_filter(np.asarray(out, dtype=np.float64))

    # 4) Baseline correction
    if config.baseline_correction:
        bf = BaselineFilter(cutoff=0.15, fs=config.sampling_rate)
        out = np.array([bf.filter_sample(float(x)) for x in out])

    #remove first 3 seconds of samples
    """skip_samples = 3 * config.sampling_rate
    if len(out) > skip_samples:
        out = out[skip_samples:]
    """

    return np.asarray(out, dtype=np.float64)
