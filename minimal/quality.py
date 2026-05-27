import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy.signal import welch, butter, sosfiltfilt, iirnotch, tf2sos

# Clinical lead weights for ECG Quality aggregation
CLINICAL_WEIGHTS: Dict[str, float] = {
    "I": 0.08,
    "II": 0.1,
    "III": 0.08,
    "aVR": 0.06,
    "aVL": 0.06,
    "aVF": 0.08,
    "V1": 0.10,
    "V2": 0.10,
    "V3": 0.08,
    "V4": 0.08,
    "V5": 0.09,
    "V6": 0.09,
}

AMBULANCE_WEIGHTS: Dict[str, float] = {
    "I": 0.06,
    "II": 0.20,
    "III": 0.07,
    "aVR": 0.03,
    "aVL": 0.05,
    "aVF": 0.10,
    "V1": 0.12,
    "V2": 0.10,
    "V3": 0.08,
    "V4": 0.07,
    "V5": 0.06,
    "V6": 0.06,
}

FLAGS_WEIGHTS: Dict[str, float] = {
    "Muscle_Artifact": 0.2,
    "Bad_Electrode_Contact": 0.2,
    "Powerline_Interference": 0.2,
    "Baseline_Drift": 0.2,
    "Low_SNR": 0.2,
}


# Thresholds for flagging issues for (T_good,T_bad)
T_GRADES: Dict[str, Tuple[float, float]] = {
    "Muscle_Artifact": (0.035, 0.088),   # prev value  (0.05, 0.1),
    "Bad_Electrode_Contact": (10, 800),
    "Powerline_Interference": (0.002, 0.02), # Плохо реагирует на шум  0.01 - 0.05 good detection
    "Baseline_Drift": (0.02, 0.85), # old (0.02, 0.1),
    "Low_SNR": (30, 17),   # prev range 20-10 - 25, 15 
}

# Mapping of flag names to user-friendly messages
FLAG_MESSAGES: Dict[str, str] = {
    "Muscle_Artifact": "Excess muscle noise",
    "Bad_Electrode_Contact": "Poor electrode contact",
    "Powerline_Interference": "Power-line interference detected",
    "Baseline_Drift": "Baseline drift present",
    "Low_SNR": "Low signal-to-noise ratio",
}

def _band_power(psd, freqs, f0, width):
    lo, hi = f0 - width, f0 + width
    m = (freqs >= lo) & (freqs <= hi)
    if not np.any(m):
        return 0.0
    return psd[m].sum()

def _band_power_trapz(freqs, psd, f0, halfwidth):
    lo, hi = f0 - halfwidth, f0 + halfwidth
    m = (freqs >= lo) & (freqs <= hi)
    if not np.any(m):
        return 0.0
    # интеграл по частоте (корректнее, чем простая сумма)
    return float(np.trapz(psd[m], freqs[m]))

def _peak_in_window(freqs, psd, fmin, fmax, fallback):
    m = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(m):
        return float(fallback)
    sub_f = freqs[m]
    sub_p = psd[m]
    return float(sub_f[np.argmax(sub_p)])

def analyze_lead_quality(
    signal: np.ndarray,
    sampling_rate: int = 500,
    next_window_signal: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute signal-quality metrics for a single ECG segment.
    Returns:
      - flags: Dict of boolean flags
      - values: Dict of computed values
      - QRS_Amplitude (float)
      - SNR_dB (float)
    """
    # Remove DC offset
    sig = signal - np.mean(signal)

    # Initialize flags
    flags: Dict[str, float] = {}

    # Muscle artifact: 35–100 Hz excluding 50/60 Hz >10%
    # hf = np.sum(psd[(freqs > 40) & (freqs < 100)])
    # Daniel Code hf = np.sum(psd[(freqs > 40) & (freqs < 100) & ~((freqs > 49) & (freqs < 51))])
   
    # 1) Notch-фильтры 50 Гц и 60 Гц в SOS
    # создаёт коэффициенты IIR-фильтра-«заглушки» на 50 Гц. Q=80: добротность фильтра — высокая Q даёт очень узкий «выкол», чтобы не задеть соседние частоты.
    # Результат: два массива b50 и a50 — числитель и знаменатель передаточной функции фильтра второго порядка.
    # Q=50 → средний вырез (~49,5–50,5 Гц) → оптимальный баланс.
    b50, a50 = iirnotch(50, Q=50, fs=sampling_rate)
    # конвертирует парочку (b50, a50) в формат SOS (second-order sections). - sosfiltfilt умеет работать только с SOS, а не с обычными коэффициентами.
    sos50    = tf2sos(b50, a50)
    b60, a60 = iirnotch(60, Q=50, fs=sampling_rate)
    sos60    = tf2sos(b60, a60)


    # 2) Применяем notch-фильтры sosfiltfilt: фильтрует сигнал двухпроходно (прямой + обратный проход), чтобы убрать фазовые искажения.
    sig_notched = sosfiltfilt(sos50, sig) 
    # sig_notched — уже без 50 -60 Гц-компоненты.
    sig_notched = sosfiltfilt(sos60, sig_notched)

    # 3) nperseg для всех Welch-оценок: окно ~2.5 с → Δf ≈ 0.4 Гц
    nperseg = int(2.5 * sampling_rate)

    # 4) Флаг сетевого шума (48–52 и 58–62 Гц 100 120)  маска, которая истинна для частот в ±2 Гц вокруг 50 Гц и 60 Гц.
    
    x_raw = signal - np.mean(signal)
    freqs_raw, psd_raw = welch(x_raw, fs=sampling_rate, nperseg=nperseg, window="hann", detrend="constant")
    total_raw = psd_raw.sum() + 1e-12
    # адаптивный поиск пиков 50/60 Гц (учтёт дрейф частоты)
    win50 = (freqs_raw >= 45) & (freqs_raw <= 55)
    win60 = (freqs_raw >= 55) & (freqs_raw <= 65)
    f50_peak = freqs_raw[win50][np.argmax(psd_raw[win50])] if np.any(win50) else 50.0
    f60_peak = freqs_raw[win60][np.argmax(psd_raw[win60])] if np.any(win60) else 60.0

    # мощность вокруг пиков и их гармоник
    p50  = _band_power(psd_raw, freqs_raw, f50_peak, width=1.5)
    p60  = _band_power(psd_raw, freqs_raw, f60_peak, width=1.5)
    p100 = _band_power(psd_raw, freqs_raw, 2.0 * f50_peak, width=1.0) if 2.0 * f50_peak <= freqs_raw[-1] else 0.0
    p120 = _band_power(psd_raw, freqs_raw, 2.0 * f60_peak, width=1.0) if 2.0 * f60_peak <= freqs_raw[-1] else 0.0

    pi_pre = (p50 + p60 + p100 + p120) / total_raw

    # --- 2) PSD после фильтра (остаток) ---
    x_post = sig_notched - np.mean(sig_notched)
    freqs_post, psd_post = welch(x_post, fs=sampling_rate, nperseg=nperseg, window="hann", detrend="constant")
    total_post = psd_post.sum() + 1e-12

    # измеряем остаток вокруг тех же пиков (чтобы сравнение было честным)
    p50_r  = _band_power(psd_post, freqs_post, f50_peak, width=1.5)
    p60_r  = _band_power(psd_post, freqs_post, f60_peak, width=1.5)
    p100_r = _band_power(psd_post, freqs_post, 2.0 * f50_peak, width=1.0) if 2.0 * f50_peak <= freqs_post[-1] else 0.0
    p120_r = _band_power(psd_post, freqs_post, 2.0 * f60_peak, width=1.0) if 2.0 * f60_peak <= freqs_post[-1] else 0.0

    pi_post = (p50_r + p60_r + p100_r + p120_r) / total_post
    # E1: оцениваем флаг по pi_pre (уровень помехи ДО notch), а не по остатку после фильтра —
    # иначе флаг всегда около нуля и бесполезен.
    pi = pi_pre
    flags["Powerline_Interference"] = np.clip(
        (pi_pre - T_GRADES["Powerline_Interference"][0])
        / (T_GRADES["Powerline_Interference"][1] - T_GRADES["Powerline_Interference"][0]),
        0,
        1,
    )
    

    """  очень тяжелая
    nperseg = int(2.5 * sampling_rate)  # всё окно → Δf ~ 0.4 Гц
    x_raw = signal - np.mean(signal)
    freqs_raw, psd_raw = welch(x_raw, fs=sampling_rate, nperseg=nperseg,
                            window="hann", detrend="constant")
    total_raw = float(np.trapz(psd_raw, freqs_raw) + 1e-12)

    # адаптивные пики 50/60 Гц
    f50_peak = _peak_in_window(freqs_raw, psd_raw, 45.0, 55.0, 50.0)
    f60_peak = _peak_in_window(freqs_raw, psd_raw, 55.0, 65.0, 60.0)

    # интеграция вокруг пиков и гармоник
    fund_hw  = 1.5  # Δ1
    harm_hw  = 1.0  # Δk, k>=2
    harmonics = (1, 2)  # можно (1,2,3) при необходимости

    def _pl_power(freqs, psd, f50, f60):
        p = 0.0
        for k in harmonics:
            hw = fund_hw if k == 1 else harm_hw
            f50k = k * f50
            f60k = k * f60
            if f50k <= freqs[-1]:
                p += _band_power_trapz(freqs, psd, f50k, hw)
            if f60k <= freqs[-1]:
                p += _band_power_trapz(freqs, psd, f60k, hw)
        return p

    p_raw = _pl_power(freqs_raw, psd_raw, f50_peak, f60_peak)
    pi_pre = float(p_raw / total_raw)

    # --- 2) PSD ПОСЛЕ фильтра (остаток) ---
    x_post = sig_notched - np.mean(sig_notched)
    freqs_post, psd_post = welch(x_post, fs=sampling_rate, nperseg=nperseg,
                                window="hann", detrend="constant")
    total_post = float(np.trapz(psd_post, freqs_post) + 1e-12)

    # измеряем остаток вокруг ТЕХ ЖЕ адаптивных пиков
    p_post = _pl_power(freqs_post, psd_post, f50_peak, f60_peak)
    pi_post = float(p_post / total_post)
    pi = pi_pre
    flags["Powerline_Interference"] = np.clip(
        (pi_post - T_GRADES["Powerline_Interference"][0])
        / (T_GRADES["Powerline_Interference"][1] - T_GRADES["Powerline_Interference"][0]),
        0,
        1,
    )
    """
    
    """     
    # Old version 4) Флаг сетевого шума (48–52 и 58–62 Гц)  маска, которая истинна для частот в ±2 Гц вокруг 50 Гц и 60 Гц.
    mains_bins = ((freqs2 >= 48) & (freqs2 <= 52)) | ((freqs2 >= 58) & (freqs2 <= 62))
    # Суммируем энергию в этих бинax — это сколько сетевого «гудения» ещё осталось.
    mains_power = np.sum(psd2[mains_bins])
    # pi (powerline_interference): отношение сетевого шума ко всей энергии — показатель «грязи» от сети.
    pi = mains_power / total_power2
    flags["Powerline_Interference"] = np.clip(
        (pi - T_GRADES["Powerline_Interference"][0])
        / (T_GRADES["Powerline_Interference"][1] - T_GRADES["Powerline_Interference"][0]),
        0,
        1,
    ) 
    """

    """ 
    # 5) Флаг мышечного артефакта (35–100 Гц) hf_bins: маска для диапазона высокой частоты, где живёт EMG-шум.
    hf_bins = (freqs2 >= 35) & (freqs2 <= 100)
    # Суммарная энергия в этом диапазоне.
    hf_power = np.sum(psd2[hf_bins])
    # ma_ratio: доля мышечного шума от общей энергии.
    ma_ratio = hf_power / total_power2

    flags["Muscle_Artifact"] = np.clip(
       (ma_ratio - T_GRADES["Muscle_Artifact"][0])
        / (T_GRADES["Muscle_Artifact"][1] - T_GRADES["Muscle_Artifact"][0]),
        0,
        1,
    ) 
    """ 
  
    # 5) Флаг мышечного артефакта (35–100 Гц) hf_bins: маска для диапазона высокой частоты, где живёт EMG-шум.
    bw_ma = 1.5

    # маски в PSD после фильтра
    hf_mask  = (freqs_post >= 35.0) & (freqs_post <= 100.0)
    ex50     = (freqs_post >= (f50_peak - bw_ma)) & (freqs_post <= (f50_peak + bw_ma))
    ex60     = (freqs_post >= (f60_peak - bw_ma)) & (freqs_post <= (f60_peak + bw_ma))
    ex100    = (freqs_post >= 99.0) & (freqs_post <= 101.0)  # 2-я гармоника 50 Гц на границе диапазона

    hf_power_psd = psd_post[hf_mask & ~(ex50 | ex60 | ex100)].sum()
    ma_ratio_psd = hf_power_psd / (total_post + 1e-12)

    # тайм-домен фолбэк: полоса 35–100 Гц (после notched сигнала)
    # from scipy.signal import butter, sosfiltfilt
    sos_ma = butter(4, [35.0, 100.0], btype="bandpass", fs=sampling_rate, output="sos")
    x_hf   = sosfiltfilt(sos_ma, sig_notched)
    rms_hf = np.sqrt(np.mean(x_hf**2))
    rms_all = np.sqrt(np.mean((sig_notched - np.mean(sig_notched))**2)) + 1e-12
    ma_ratio_td = (rms_hf / rms_all)**2

    # финальная метрика как максимум из PSD- и TD-оценок
    ma_ratio = max(ma_ratio_psd, ma_ratio_td)        

    flags["Muscle_Artifact"] = np.clip(
       (ma_ratio - T_GRADES["Muscle_Artifact"][0])
        / (T_GRADES["Muscle_Artifact"][1] - T_GRADES["Muscle_Artifact"][0]),
        0,
        1,
    ) 
    

    
    # Old Code     
    """
    mask = (freqs > 35) & (freqs < 99)
    bad50 = (freqs >= 48) & (freqs <= 52)
    bad60 = (freqs >= 58) & (freqs <= 62)
    hf = np.sum(psd[mask & ~(bad50 | bad60)])
    ma = hf / total_power
    flags["Muscle_Artifact"] = np.clip(
       (ma - T_GRADES["Muscle_Artifact"][0])
        / (T_GRADES["Muscle_Artifact"][1] - T_GRADES["Muscle_Artifact"][0]),
        0,
        1,
    )
    flags["Muscle_Artifact"] = 1.0 if ma > 0.1 else -np.log10(1 - 9 * ma)
    ma_ratio = ma
    
    # Powerline interference: 49–51 & 59–61 Hz
    #old code
    
    p50 = np.sum(psd[(freqs > 49) & (freqs < 51)])
    p60 = np.sum(psd[(freqs > 59) & (freqs < 61)])
    pi = (p50 + p60) / total_power
    flags["Powerline_Interference"] = np.clip(
        (pi - T_GRADES["Powerline_Interference"][0])
        / (T_GRADES["Powerline_Interference"][1] - T_GRADES["Powerline_Interference"][0]),
        0,
        1,
    )
    # flags["Powerline_Interference"] = 1.0 if pi > 0.05 else -np.log10(1 - 20 * pi)
    """


    # Baseline drift: <0.5 Hz - use 2 windows if next_window_signal is available
    # Code Alex 1
    """
    if next_window_signal is not None:
        # Concatenate current and next window for baseline drift analysis
        combined_sig = np.concatenate([signal, next_window_signal])
        # Снимаем DC-компоненту
        combined_sig -= np.mean(combined_sig)
        nperseg = len(combined_sig)
        # Берём весь сигнал одним сегментом, чтобы Δf ≈ 1 / T_total
        nfft = 2**int(np.ceil(np.log2(nperseg)))
        freqs_bd, psd_bd = welch(
            combined_sig,
            fs=sampling_rate,
            nperseg=nperseg,
            nfft=nfft,  # nfft с нулевым дополнением даёт ещё более «мелкий» шаг, не меняя стабильность оценки.
            window="hann",
            detrend="constant"  # detrend="constant" убирает токовую составляющую перед оценкой.
        )
        total_power_bd = np.sum(psd_bd) + 1e-12
        lf = np.sum(psd_bd[freqs_bd < 0.5])
        bd = lf / total_power_bd
    else:
        lf = np.sum(psd[freqs < 0.5])
        bd = lf / total_power
    """
    # Baseline drift: <0.5 Hz - use 2 windows if next_window_signal is available
    # Code Alex 2
    if next_window_signal is not None:
        combined_sig = np.concatenate([signal, next_window_signal])
        combined_sig -= combined_sig.mean()
        freqs_bd, psd_bd = welch(combined_sig, fs=sampling_rate, nperseg=int(1.25*sampling_rate))
        total_power_bd = psd_bd.sum() + 1e-12
        lf = psd_bd[freqs_bd < 0.5].sum()
        bd = lf / total_power_bd
    else:
        # E2: fallback на PSD после notch (с nperseg=2.5·fs → Δf ≈ 0.4 Гц, есть бины < 0.5 Гц).
        # Раньше использовался welch без nperseg, где ни один бин не попадал < 0.5 Гц → bd ≈ 0.
        lf = psd_post[freqs_post < 0.5].sum()
        bd = lf / total_post

    flags["Baseline_Drift"] = np.clip(
            (bd - T_GRADES["Baseline_Drift"][0])
            / (T_GRADES["Baseline_Drift"][1] - T_GRADES["Baseline_Drift"][0]),
            0,
            1,
        )


    # Baseline drift: <0.5 Hz - use 2 windows if next_window_signal is available
    # old Daniel code
    """
    if next_window_signal is not None:
        # Concatenate current and next window for baseline drift analysis
        combined_sig = np.concatenate(
            [signal, next_window_signal]
        )
        combined_sig -= np.mean(combined_sig)
        freqs_bd, psd_bd = welch(combined_sig, fs=sampling_rate)
        total_power_bd = np.sum(psd_bd) + 1e-12
        lf = np.sum(psd_bd[(freqs_bd < 0.5)])
        bd = lf / total_power_bd
    else:
        # Fall back to single window for baseline drift
        lf = np.sum(psd[(freqs < 0.5)])
        bd = lf / total_power

    flags["Baseline_Drift"] = np.clip(
        (bd - T_GRADES["Baseline_Drift"][0])
        / (T_GRADES["Baseline_Drift"][1] - T_GRADES["Baseline_Drift"][0]),
        0,
        1,
    )
    """
    # flags["Baseline_Drift"] = 1.0 if bd > 0.1 else -np.log10(1 - 10 * bd)

    # E6: QRS-амплитуду меряем от сигнала с удалённым дрейфом (HP-фильтр 1 Гц),
    # иначе baseline drift раздувает ptp и Bad_Electrode_Contact даёт ложноположительные.
    sos_hp = butter(4, 1, btype="highpass", fs=sampling_rate, output="sos")
    clean_hp = sosfiltfilt(sos_hp, sig)
    amp = float(np.ptp(clean_hp))
    bc = amp  # в values["b_e_c"] сохраняем именно амплитуду (так осмысленнее, чем доля LF)

    # E3: Bad electrode contact — плавная оценка между порогами + явная нулевая ветка.
    t_lo, t_hi = T_GRADES["Bad_Electrode_Contact"]
    if amp < t_lo or amp > t_hi:
        flags["Bad_Electrode_Contact"] = 1.0
    else:
        flags["Bad_Electrode_Contact"] = 0.0

    # SNR in dB
    amp_threshold = 5  # порог "нет сигнала" (в единицах исходного сигнала)
    no_signal = amp < amp_threshold

    if no_signal:
        snr = 0.1
    else:
        try:
            sos = butter(2, [0.5, 40], btype="bandpass", fs=sampling_rate, output="sos")
            clean = sosfiltfilt(sos, sig)
            noise = sig - clean
            # E4: SNR = средняя мощность сигнала / средняя мощность шума.
            # Раньше использовался amp² (peak-to-peak)² — это смешивало пик и среднее
            # и давало смещённые значения SNR для каналов с разной QRS-амплитудой.
            signal_power = np.mean(clean ** 2) + 1e-12
            noise_power = np.mean(noise ** 2) + 1e-12
            snr = 10 * np.log10(signal_power / noise_power)
        except Exception:
            signal_power = np.mean(sig ** 2) + 1e-12
            noise_power = np.var(sig) + 1e-12
            snr = 10 * np.log10(signal_power / noise_power)

    flags["Low_SNR"] = np.clip(
        (snr - T_GRADES["Low_SNR"][0])
        / (T_GRADES["Low_SNR"][1] - T_GRADES["Low_SNR"][0]),
        0,
        1,
    )

    return {
        "flags": flags,
        "values": {
            "m_a": ma_ratio,
            "b_e_c": bc,
            "p_i": pi,
            "b_d": bd,
            "snr": snr,
            "qrs_amp": amp,
        },
    }


def compute_quality_score(flags: Dict[str, float]) -> float:
    """
    Derive quality score (0.0–1.0) where each true flag deducts 0.2.
    """
    # score = max(0.0, 1.0 - 0.2 * sum(flags.values()))
    score = 1.0
    for flag, value in flags.items():
        if value > 0.0:
            score -= FLAGS_WEIGHTS.get(flag, 0.2) * value
    score = max(0.0, min(1.0, score))
    return score


def analyze_ecg_all_leads(
    leads: Dict[str, np.ndarray],
    sampling_rate: int = 500,
    weights: Dict[str, float] = CLINICAL_WEIGHTS,
    window_sec: float = 2.5,
    result_window_sec: float = 5.0,
) -> Dict[str, Any]:
    """
    Analyze all 12 leads, segmenting into windows, computing per-window metrics,
    finding the best consecutive windows that form a 5-second result window,
    aggregating total, and returning problems for feedback.

    Returns:
      lead_quality: Dict of lead -> {QualityScore, Problems: List[str], SNR_dB, QRS_Amplitude}
      total_quality: float
      classification: str
      best_windows_used: List[int] - consecutive window indices used for result
    """
    # Window parameters
    wlen = int(window_sec * sampling_rate)
    n = len(next(iter(leads.values())))
    nwin = n // wlen if wlen > 0 else 0

    # Calculate how many consecutive windows we need for the result window
    windows_needed = int(result_window_sec / window_sec)  # Should be 2 for 5sec/2.5sec

    # Store all window metrics for finding best consecutive windows
    all_window_metrics: List[Dict[str, Any]] = []

    # Compute metrics for all windows
    for i in range(nwin):
        window_metrics = {}
        window_quality_scores = []

        for lead, sig in leads.items():
            seg = sig[i * wlen : (i + 1) * wlen]

            # Get next window for baseline drift calculation if available
            next_seg = None
            if i < nwin - 1:  # Not the last window
                next_seg = sig[(i + 1) * wlen : (i + 2) * wlen]

            metrics = analyze_lead_quality(seg, sampling_rate, next_seg)
            flags = metrics["flags"]
            quality_score = compute_quality_score(flags)

            window_metrics[lead] = {
                "quality_score": quality_score,
                "metrics": metrics,
                "flags": flags,
            }
            window_quality_scores.append(quality_score)

        # Calculate overall window quality (average across all leads)
        window_overall_quality = np.mean(window_quality_scores)
        all_window_metrics.append(
            {
                "window_index": i,
                "overall_quality": window_overall_quality,
                "lead_metrics": window_metrics,
            }
        )

    # Find the best consecutive windows that form the result window
    best_consecutive_windows = []
    best_consecutive_quality = -1.0

    # Try all possible consecutive window combinations
    for start_idx in range(nwin - windows_needed + 1):
        consecutive_windows = all_window_metrics[start_idx : start_idx + windows_needed]

        # Calculate average quality of this consecutive sequence
        avg_quality = np.mean([w["overall_quality"] for w in consecutive_windows])

        if avg_quality > best_consecutive_quality:
            best_consecutive_quality = avg_quality
            best_consecutive_windows = consecutive_windows

    lead_quality: Dict[str, Any] = {}
    total_quality = 0.0
    error_level = 0.0

    for lead in leads.keys():
        if best_consecutive_windows:
            # Use metrics from best consecutive windows for this lead
            best_metrics = [w["lead_metrics"][lead] for w in best_consecutive_windows]

            # Average quality scores and metrics from best windows
            quality_scores = [m["quality_score"] for m in best_metrics]
            qrs_amp_values = [m["metrics"]["values"]["qrs_amp"] for m in best_metrics]
            snr_values = [m["metrics"]["values"]["snr"] for m in best_metrics]
            ma_values = [m["metrics"]["values"]["m_a"] for m in best_metrics]
            bec_values = [m["metrics"]["values"]["b_e_c"] for m in best_metrics]
            pi_values = [m["metrics"]["values"]["p_i"] for m in best_metrics]
            bd_values = [m["metrics"]["values"]["b_d"] for m in best_metrics]

            avg_q = float(np.mean(quality_scores))
            avg_qrs_amp = float(np.mean(qrs_amp_values))
            avg_snr = float(np.mean(snr_values))
            avg_ma = float(np.mean(ma_values))
            avg_bec = float(np.mean(bec_values))
            avg_pi = float(np.mean(pi_values))
            avg_bd = float(np.mean(bd_values))

            # E12: проблема считается, если СРЕДНЕЕ значение флага по выбранным окнам >= 0.5.
            # Раньше требовалось, чтобы флаг был > 0.5 в >=50% окон — при 2 окнах это означало
            # "в обоих окнах сразу" и пропускало транзитные события.
            problems: List[str] = []
            for flag in FLAG_MESSAGES.keys():
                vals = [m["flags"].get(flag, 0.0) for m in best_metrics]
                if vals and float(np.mean(vals)) >= 0.5:
                    problems.append(FLAG_MESSAGES[flag])
        else:
            # Fallback if no windows available
            avg_q = 0.0
            avg_snr = 0.0
            avg_qrs_amp = 0.0
            avg_ma = 0.0
            avg_bec = 0.0
            avg_pi = 0.0
            avg_bd = 0.0
            problems = []

        lead_quality[lead] = {
            "QualityScore": avg_q,
            "Problems": problems,
            "SNR_dB": avg_snr,
            "QRS_Amplitude": avg_qrs_amp,
            "m_a": avg_ma,
            "b_e_c": avg_bec,
            "p_i": avg_pi,
            "b_d": avg_bd,
        }
        total_quality += avg_q * weights.get(lead, 0.0)

        if avg_q < 0.65:
            error_level += 0.2
        if avg_q < 0.85:
            error_level += 0.02

    total_quality = total_quality - error_level

    # Classification
    if total_quality > 0.85:
        classification = "Good"
    elif total_quality > 0.65:
        classification = "Questionable"
    else:
        classification = "Not usable"

    return {
        "lead_quality": lead_quality,
        "total_quality": total_quality,
        "classification": classification,
        "best_windows_used": (
            [w["window_index"] for w in best_consecutive_windows]
            if best_consecutive_windows
            else []
        ),
    }
