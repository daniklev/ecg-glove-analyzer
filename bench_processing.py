"""
Standalone timing benchmark for the ECG processing pipeline.

Measures ONLY the data-processing path used to produce statistics:
    1. raw decode            (ECGPacketDecoder.decode)
    2. decode_data total     (decode + per-sample filtering + cleaning copy)
       -> filtering          (decode_data total - raw decode)
    3. compute_quality       (the "statistics" -> analyze_all_leads)
    4. process               (R-peak detection + delineation + measurements)

It does NOT touch gui_ecg.py, does NOT draw any matplotlib figure and does
NOT fill any Qt table -- so the reported numbers are pure compute time.

Config mirrors the GUI defaults (clean='none', notch=[60], spike removal on,
HP 0.15 Hz, 500 Hz sampling).

Usage:
    python bench_processing.py                 # uses a default file
    python bench_processing.py data/X.ret ...  # one or more files
"""

import os
import sys
import time
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ecg_glove import EcgGlove  # noqa: E402
from glove_decoder import ECGPacketDecoder  # noqa: E402
from ecg_filters import HPFilterType  # noqa: E402

REPEATS = int(os.environ.get("BENCH_REPEATS", "5"))  # repetitions per phase


def _make_glove() -> EcgGlove:
    """EcgGlove configured exactly like the GUI's default checkboxes."""
    return EcgGlove(
        sampling_rate=500,
        clean_method="none",
        peak_method="neurokit",
        filters=[60],
        spike_removal=True,
        hp_filter_type=HPFilterType.HP015,
        powerline_freq=60,
        enable_baseline_correction=False,
        enable_smoothing=False,
        smoothing_window=5,
    )


def _stats(times):
    """Return (mean, min) in ms for a list of seconds."""
    return statistics.mean(times) * 1000.0, min(times) * 1000.0


def _time(fn, repeats=REPEATS, warmup=1):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return samples


def bench_file(path: str) -> None:
    with open(path, "rb") as f:
        data_bytes = f.read()

    # --- signal size info -------------------------------------------------
    g0 = _make_glove()
    g0.decode_data(data_bytes)
    n_leads = len(g0.raw_signals)
    n_samples = max((s.size for s in g0.raw_signals.values()), default=0)
    fs = g0.sampling_rate
    dur_s = n_samples / fs if fs else 0.0
    n_win = n_samples // int(2.5 * fs) if fs else 0

    print("=" * 70)
    print(f"File: {os.path.basename(path)}  ({len(data_bytes)/1024:.0f} KB)")
    print(
        f"Leads: {n_leads} | samples/lead: {n_samples} "
        f"| duration: {dur_s:.1f} s | 2.5s-windows: {n_win}"
    )
    print("-" * 70)

    # --- 1. raw decode only ----------------------------------------------
    decode_samples = _time(lambda: ECGPacketDecoder().decode(data_bytes))

    # --- 2. decode_data total (decode + filtering + cleaning copy) -------
    def _decode_data():
        g = _make_glove()
        g.decode_data(data_bytes)

    decode_data_samples = _time(_decode_data)

    # --- 3. compute_quality = the statistics -----------------------------
    g_stats = _make_glove()
    g_stats.decode_data(data_bytes)
    quality_samples = _time(lambda: g_stats.compute_quality())

    # --- 4. process (measurements) ---------------------------------------
    def _process():
        g_stats.quality_scores = {}  # force recompute path off
        g_stats.compute_quality()
        g_stats.process()

    # time process alone: pre-populate quality once, then time process only
    g_stats.compute_quality()

    def _process_only():
        g_stats.process()

    process_samples = _time(_process_only)

    # --- report -----------------------------------------------------------
    dec_mean, dec_min = _stats(decode_samples)
    dd_mean, dd_min = _stats(decode_data_samples)
    q_mean, q_min = _stats(quality_samples)
    p_mean, p_min = _stats(process_samples)
    filt_mean = dd_mean - dec_mean  # filtering ≈ decode_data - raw decode

    print(f"{'phase':<34}{'mean (ms)':>12}{'min (ms)':>12}")
    print(f"{'  raw decode':<34}{dec_mean:>12.1f}{dec_min:>12.1f}")
    print(f"{'  filtering':<34}{filt_mean:>12.1f}{'':>12}")
    print(f"{'decode_data TOTAL':<34}{dd_mean:>12.1f}{dd_min:>12.1f}")
    print(f"{'compute_quality (STATISTICS)':<34}{q_mean:>12.1f}{q_min:>12.1f}")
    print(f"{'process (measurements)':<34}{p_mean:>12.1f}{p_min:>12.1f}")
    print("-" * 70)
    total_stats = dd_mean + q_mean
    print(
        f"{'TOTAL to get statistics':<34}{total_stats:>12.1f} ms"
        f"   (decode_data + compute_quality)"
    )
    print(
        f"{'TOTAL incl. measurements':<34}{dd_mean + q_mean + p_mean:>12.1f} ms"
    )
    print("=" * 70)
    print()


def main():
    args = sys.argv[1:]
    if not args:
        args = [os.path.join("data", "220209015248248.ret")]
    for path in args:
        if not os.path.isfile(path):
            print(f"!! not found: {path}")
            continue
        bench_file(path)


if __name__ == "__main__":
    main()
