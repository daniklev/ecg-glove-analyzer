"""Standalone ECG 12-lead quality + measurements runner.

Self-contained drop-in for the microservice: imports only the sibling modules
in this folder (glove_decoder, ecg_filters, ecg_processor, ecg_glove), which are
verbatim copies of ecg-glove-analyzer/src/. Running this on a .ret recording
reproduces exactly what the GUI (src/gui_ecg.py) computes for the same input and
filter configuration, so the two can be diffed 1:1.

Usage:
    python run_quality.py path/to/file.ret
    python run_quality.py --base64 <BASE64>            # raw bytes as base64
    python run_quality.py --base64-file b64.txt         # base64 from a file
    python run_quality.py file.ret --json               # machine-readable output

Filter config defaults match the GUI defaults; override with the flags below to
reproduce a non-default GUI run.
"""

import argparse
import base64
import contextlib
import io
import json
import os
import sys

# Make the verbatim flat-import copies resolve regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from ecg_glove import EcgGlove  # noqa: E402
from ecg_filters import HPFilterType  # noqa: E402


# --- GUI-side derived metrics (gui_ecg.py:1278-1326) ----------------------
# These three category scores are computed in the GUI, not the engine, so they
# are reproduced here verbatim for full parity with the GUI results panel.
_CLINICAL_WEIGHTS = {
    "I": 0.07, "II": 0.12, "III": 0.06, "aVR": 0.04, "aVL": 0.06,
    "aVF": 0.09, "V1": 0.10, "V2": 0.10, "V3": 0.10, "V4": 0.08,
    "V5": 0.09, "V6": 0.09,
}


def _category_score(lead_quality_map, flag_names):
    if not lead_quality_map or not flag_names:
        return None
    total_w = 0.0
    weighted_penalty = 0.0
    n_flags = len(flag_names)
    for lead, lq_data in lead_quality_map.items():
        w = _CLINICAL_WEIGHTS.get(lead, 0.08)
        active = sum(1 for f in flag_names if lq_data.get(f))
        weighted_penalty += w * (active / n_flags)
        total_w += w
    if total_w <= 0:
        return None
    return float(max(0.0, min(1.0, 1.0 - weighted_penalty / total_w)))


_HP_MAP = {
    "0.05": HPFilterType.HP005,
    "0.15": HPFilterType.HP015,
    "0.5": HPFilterType.HP05,
}

_FLAG_NAMES = [
    "Muscle_Artifact",
    "Powerline_Interference",
    "Baseline_Drift",
    "Bad_Electrode_Contact",
    "Low_SNR",
]


def _to_native(obj):
    """Recursively coerce numpy scalars/arrays/tuples to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_to_native(v) for v in obj.tolist()]
    return obj


class InvalidEcgInput(Exception):
    """Raised when the input bytes don't decode into a 12-lead ECG."""


def _ensure_decoded(glove) -> None:
    raw = glove.raw_signals
    n = max((v.size for v in raw.values()), default=0) if raw else 0
    if n == 0 or "II" not in raw:
        raise InvalidEcgInput(
            f"no valid 12-lead ECG decoded ({n} samples). Input must be a raw "
            ".ret recording (or its base64) - files like .pdf/.png are not ECG "
            "packet streams."
        )


def _read_input_bytes(args) -> bytes:
    if args.base64 is not None:
        return base64.b64decode(args.base64)
    if args.base64_file is not None:
        with open(args.base64_file, "r", encoding="utf-8") as f:
            return base64.b64decode(f.read().strip())
    with open(args.path, "rb") as f:
        return f.read()


def build_glove(args) -> EcgGlove:
    filters = []
    if args.notch:
        filters = [int(x) for x in args.notch.split(",") if x.strip()]
    return EcgGlove(
        sampling_rate=args.sampling_rate,
        clean_method=args.clean_method,
        peak_method=args.peak_method,
        filters=filters,
        spike_removal=not args.no_spike_removal,
        hp_filter_type=_HP_MAP[args.hp],
        powerline_freq=args.powerline_freq,
        enable_baseline_correction=args.baseline_correction,
        enable_smoothing=args.smoothing,
        smoothing_window=args.smoothing_window,
    )


def analyze(data_bytes: bytes, args) -> dict:
    glove = build_glove(args)

    # The engine prints non-fatal diagnostics (e.g. "Error calculating power
    # spectrum" on short windows) to stdout; identical to the GUI console.
    # Suppress only these prints — the computation is untouched — for a clean CLI.
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        try:
            glove.decode_data(data_bytes)
        except ValueError as e:  # decoder raises when no ECG packets are found
            raise InvalidEcgInput(str(e)) from e
        _ensure_decoded(glove)
        quality = glove.compute_quality()  # full dict from analyze_all_leads
    lead_quality = quality.get("lead_quality", {})

    # GUI-side category scores.
    category_scores = {
        "noise": _category_score(
            lead_quality, ["Muscle_Artifact", "Powerline_Interference"]
        ),
        "movement": _category_score(lead_quality, ["Baseline_Drift"]),
        "contact": _category_score(lead_quality, ["Bad_Electrode_Contact"]),
    }

    # Measurements (mirror gui_ecg.py: quality already shown if this fails).
    measurements = None
    analysis_lead = None
    measurement_error = None
    try:
        with contextlib.redirect_stdout(sink):
            results = glove.process()
        analysis_lead = results.get("AnalysisLead")
        measurements = results.get("ecgData", {}).get("measurements")
    except Exception as err:  # noqa: BLE001 - mirror GUI behaviour
        measurement_error = str(err)

    return {
        "overall_quality": quality.get("overall_quality"),
        "classification": quality.get("classification"),
        "best_window_time_s": quality.get("best_window_time_s"),
        "best_window_quality": quality.get("best_window_quality"),
        "bad_leads": quality.get("bad_leads"),
        "unusable_leads": quality.get("unusable_leads"),
        "category_scores": category_scores,
        "lead_quality": lead_quality,
        "per_5s_window": quality.get("per_5s_window", []),
        "analysis_lead": analysis_lead,
        "measurements": measurements,
        "measurement_error": measurement_error,
    }


def _fmt(v, nd=3):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "N/A"


def print_report(out: dict) -> None:
    print("=" * 60)
    print("ECG 12-LEAD QUALITY")
    print("=" * 60)
    print(f"Overall quality : {_fmt(out['overall_quality'])}")
    print(f"Classification  : {out['classification']}")
    bwt = out["best_window_time_s"]
    if bwt:
        print(
            f"Best window     : {bwt[0]:.1f}-{bwt[1]:.1f}s "
            f"(q={_fmt(out['best_window_quality'])})"
        )
    else:
        print("Best window     : N/A")
    print(f"Bad / unusable  : {out['bad_leads']} / {out['unusable_leads']}")
    cs = out["category_scores"]
    print(
        f"Category scores : noise={_fmt(cs['noise'], 2)}  "
        f"movement={_fmt(cs['movement'], 2)}  contact={_fmt(cs['contact'], 2)}"
    )

    print("\nPer-lead:")
    print(f"  {'Lead':<5} {'record_q':>9} {'nk_qual':>8}  flags")
    for lead, lq in out["lead_quality"].items():
        rec = lq.get("whole_record_quality")
        nk = lq.get("nk_quality")
        active = [f for f in _FLAG_NAMES if lq.get(f)]
        print(
            f"  {lead:<5} {_fmt(rec):>9} {_fmt(nk, 2):>8}  "
            f"{', '.join(active) if active else 'OK'}"
        )

    windows = out["per_5s_window"]
    if windows:
        print("\nPer-5s sliding windows (weighted total):")
        for i, w in enumerate(windows):
            star = " *" if out["best_window_time_s"] == (w["start_s"], w["end_s"]) else ""
            print(
                f"  W{i+1:<2} {w['start_s']:.1f}-{w['end_s']:.1f}s  "
                f"total={_fmt(w['total'])}{star}"
            )

    print("\n" + "=" * 60)
    print("MEASUREMENTS")
    print("=" * 60)
    if out["measurement_error"]:
        print(f"Analysis failed: {out['measurement_error']}")
    elif out["measurements"]:
        m = out["measurements"]
        print(f"Analysis lead   : {out['analysis_lead']}")
        for label, key, unit, nd in [
            ("Heart rate", "HeartRate_BPM", "BPM", 1),
            ("RR interval", "RR_Interval_ms", "ms", 0),
            ("P duration", "P_Duration_ms", "ms", 0),
            ("PR interval", "PR_Interval_ms", "ms", 0),
            ("QRS duration", "QRS_Duration_ms", "ms", 0),
            ("QT interval", "QT_Interval_ms", "ms", 0),
            ("QTc interval", "QTc_Interval_ms", "ms", 0),
            ("P axis", "P_Axis", "deg", 0),
            ("QRS axis", "QRS_Axis", "deg", 0),
            ("T axis", "T_Axis", "deg", 0),
        ]:
            val = m.get(key)
            if isinstance(val, (int, float)):
                print(f"  {label:<14}: {val:.{nd}f} {unit}")
            else:
                print(f"  {label:<14}: N/A")
    else:
        print("No measurements available")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", nargs="?", help="Path to a .ret recording")
    p.add_argument("--base64", help="Raw ECG bytes as a base64 string")
    p.add_argument("--base64-file", help="File containing base64-encoded ECG bytes")
    p.add_argument("--json", action="store_true", help="Emit JSON instead of text")

    # Filter config (defaults match the GUI defaults).
    p.add_argument("--sampling-rate", type=int, default=500, dest="sampling_rate")
    p.add_argument("--clean-method", default="none", dest="clean_method")
    p.add_argument("--peak-method", default="neurokit", dest="peak_method")
    p.add_argument("--notch", default="60", help="Comma-separated notch freqs, e.g. 50,60")
    p.add_argument("--hp", default="0.15", choices=["0.05", "0.15", "0.5"],
                   help="High-pass cutoff in Hz")
    p.add_argument("--no-spike-removal", action="store_true", dest="no_spike_removal")
    p.add_argument("--baseline-correction", action="store_true", dest="baseline_correction")
    p.add_argument("--smoothing", action="store_true")
    p.add_argument("--smoothing-window", type=int, default=5, dest="smoothing_window")
    p.add_argument("--powerline-freq", type=int, default=60, dest="powerline_freq")

    args = p.parse_args()
    if not args.path and args.base64 is None and args.base64_file is None:
        p.error("provide a .ret path, --base64, or --base64-file")

    data_bytes = _read_input_bytes(args)
    try:
        out = analyze(data_bytes, args)
    except InvalidEcgInput as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    if args.json:
        print(json.dumps(_to_native(out), indent=2))
    else:
        print_report(out)


if __name__ == "__main__":
    main()
