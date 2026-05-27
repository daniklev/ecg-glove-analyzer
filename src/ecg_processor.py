import numpy as np
from scipy.signal import butter, filtfilt, welch, iirnotch, tf2sos, sosfiltfilt
import neurokit2 as nk
from typing import Dict, Any, Optional, TypedDict
import warnings
import pandas as pd
from numpy.typing import NDArray

warnings.filterwarnings("ignore")


class LeadData(TypedDict):
    """Type definition for lead data structure."""

    lead_signals: Dict[str, NDArray[np.float64]]
    cleaned_signals: Dict[str, NDArray[np.float64]]


class EcgQualityProcessor:
    # Weights for clinical use
    CLINICAL_LEAD_WEIGHTS = {
        "I": 0.07,
        "II": 0.12,
        "III": 0.06,
        "aVR": 0.04,
        "aVL": 0.06,
        "aVF": 0.09,
        "V1": 0.10,
        "V2": 0.10,
        "V3": 0.10,
        "V4": 0.08,
        "V5": 0.09,
        "V6": 0.09,
    }

    # Weights for ambulance use
    AMBULANCE_LEAD_WEIGHTS = {
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

    def __init__(self, sampling_rate: int = 500, use_ambulance_weights: bool = False):
        self.sampling_rate = sampling_rate
        self.lead_weights = (
            self.AMBULANCE_LEAD_WEIGHTS
            if use_ambulance_weights
            else self.CLINICAL_LEAD_WEIGHTS
        )

    def analyze_lead_quality(self, signal: np.ndarray, cleaned) -> Dict[str, Any]:
        """Analyze the quality of a single ECG lead."""
        results = {
            "Muscle_Artifact": False,
            "Bad_Electrode_Contact": False,
            "Powerline_Interference": False,
            "Baseline_Drift": False,
            "Low_SNR": False,
            "QRS_Amplitude": None,
            "SNR_dB": None,
            "nk_quality": None,
        }

        # Pre-flight: detect "no real ECG" — flat-line, ADC-rail saturation,
        # or transient saturation spikes.
        # Key insights:
        #   - Real ECG has sharp brief QRS spikes → dev_max ≫ dev_med
        #     (ratio of median-to-max ≈ 0.05–0.1).
        #   - A flat/railed segment has no QRS-like peak → ratio stays high.
        #   - A saturation spike (movement artefact railing the ADC for a few
        #     samples) leaves most of the window quiet but a small fraction
        #     pinned at ±ADC limits. Energy ratios (MA, LF) miss this because
        #     the spike has broadband but extreme spectrum — only the
        #     ABSOLUTE values near ±32768 betray it.
        if signal.size > 0:
            dev = np.abs(signal - np.median(signal))
            dev_max = float(dev.max())
            dev_med = float(np.median(dev))
            sig_std = float(signal.std())
            ratio = dev_med / dev_max if dev_max > 0 else 0.0

            # A: essentially no signal
            # B: no QRS-like peaks + low std (flat / low-amp railing)
            if sig_std < 1.0 or (ratio >= 0.35 and sig_std < 100.0):
                results["Bad_Electrode_Contact"] = True

            # C: ADC-rail saturation spike — more than 0.5 % of samples
            #    sitting within 100 raw units of the int16 limit (±32768).
            #    See 4.ret W5/W6: brief railing artefact gives ptp=65121 with
            #    only 0.5–1.5 s of real ECG; without this check the window
            #    scores 0.97 because MA/LF percentages stay small.
            rail_threshold = 30000.0
            n_railed = int(np.sum(np.abs(signal) > rail_threshold))
            if n_railed / signal.size > 0.005:
                results["Bad_Electrode_Contact"] = True

        # Internal 50/60 Hz notch produces an "analysis-ready" signal.
        # Used for MA / BD / BC-LF / SNR detection so that spectral leakage from
        # strong powerline interference does not inflate the 40-100 Hz band and
        # cause false Muscle_Artifact / Low_SNR flags.
        try:
            sos50 = tf2sos(*iirnotch(50.0, Q=30.0, fs=self.sampling_rate))
            sos60 = tf2sos(*iirnotch(60.0, Q=30.0, fs=self.sampling_rate))
            analysis_sig = sosfiltfilt(sos50, signal)
            analysis_sig = sosfiltfilt(sos60, analysis_sig)
        except Exception:
            analysis_sig = signal

        # Additionally HP-filtered version — used for MA / PL detection so the
        # ratio reflects HF energy relative to the IN-BAND ECG content, not
        # the (typically dominant) DC/drift content. See v3_muscule_artifact
        # V3: MA on raw-notched = 5.8% (below 15% threshold) but on HP-notched
        # = 21% (correctly above threshold) — the muscle noise is real, it
        # just looked small relative to the DC/drift on the raw signal.
        try:
            b_hp_an, a_hp_an = butter(
                2, 0.5, btype="highpass", fs=self.sampling_rate, output="ba"
            )
            analysis_sig_hp = filtfilt(b_hp_an, a_hp_an, analysis_sig)
        except Exception:
            analysis_sig_hp = analysis_sig

        # Calculate power spectrum
        try:
            # PL_raw: how much 50/60 Hz energy was in the ORIGINAL signal.
            freqs_raw, psd_raw = welch(signal, fs=self.sampling_rate)
            total_raw = float(np.sum(psd_raw)) if psd_raw.size > 0 else 0.0
            pl_raw_ratio = 0.0
            if total_raw > 0:
                pl_raw_ratio = float(
                    psd_raw[(freqs_raw > 49) & (freqs_raw < 51)].sum()
                    + psd_raw[(freqs_raw > 59) & (freqs_raw < 61)].sum()
                ) / total_raw

            # Detection runs on the internally-notched signal.
            # nperseg = 2.5·fs gives Δf ≈ 0.4 Hz so the 0.01–0.5 Hz band has at
            # least one bin — without this the Baseline_Drift / BEC-LF checks
            # silently never fired (default Welch nperseg=256 → Δf≈2 Hz, no
            # bins below 1 Hz at all).
            # detrend='linear' removes per-segment mean AND linear trend so
            # recording-start transients / slow monotonic settling don't leak
            # into the LF band as false drift. Real baseline drift (breathing
            # at 0.15–0.4 Hz) is oscillatory, not linear, so it survives.
            nperseg_an = min(int(2.5 * self.sampling_rate), analysis_sig.size)
            freqs, psd = welch(
                analysis_sig,
                fs=self.sampling_rate,
                nperseg=nperseg_an,
                detrend="linear",
            )
            total_power = np.sum(psd) if psd.size > 0 else 0

            # Same PSD on the HP-filtered version — used for MA and PL
            # detection (ratios against ECG-band content, not vs DC/drift).
            freqs_hp, psd_hp = welch(
                analysis_sig_hp,
                fs=self.sampling_rate,
                nperseg=nperseg_an,
                detrend="linear",
            )
            total_power_hp = np.sum(psd_hp) if psd_hp.size > 0 else 0

            if total_power > 0:
                # 1. Muscle artifact — measured on HP'd signal (40-150 Hz,
                #    excluding 49-51/59-61 notch bands AND their 2nd harmonics
                #    99-101/119-121). On a properly cleaned ECG, the energy
                #    above 40 Hz is <0.5%; muscle noise pushes it past 6%.
                hf_mask_hp = (freqs_hp > 40) & (freqs_hp < 150)
                notch_bands_hp = (
                    ((freqs_hp > 49) & (freqs_hp < 51))
                    | ((freqs_hp > 59) & (freqs_hp < 61))
                    | ((freqs_hp > 99) & (freqs_hp < 101))
                    | ((freqs_hp > 119) & (freqs_hp < 121))
                )
                hf_power_hp = np.sum(psd_hp[hf_mask_hp & ~notch_bands_hp])
                if total_power_hp > 0 and hf_power_hp / total_power_hp > 0.06:
                    results["Muscle_Artifact"] = True

                # LF energy (drift band) — needed for several flags below.
                low_freq_power = np.sum(psd[(freqs > 0.01) & (freqs < 0.5)])
                lf_ratio = low_freq_power / total_power
                # Absolute drift amplitude in raw units (RMS in LF band).
                # Using std × sqrt(ratio) is independent of Welch normalisation.
                sig_std_an = float(np.std(analysis_sig))
                drift_std = sig_std_an * float(np.sqrt(lf_ratio))

                # 2. Bad electrode contact via LF — heavy AND large-amplitude
                #    drift only. Pure ratio would falsely fire on low-amp leads.
                if lf_ratio > 0.4 and drift_std > 300.0:
                    results["Bad_Electrode_Contact"] = True

                # 3. Powerline: residual after notch on HP'd signal (same
                #    rationale as MA — ratio vs ECG content, not vs LF).
                pl_residual_hp = 0.0
                if total_power_hp > 0:
                    pl_residual_hp = float(
                        psd_hp[(freqs_hp > 49) & (freqs_hp < 51)].sum()
                        + psd_hp[(freqs_hp > 59) & (freqs_hp < 61)].sum()
                    ) / total_power_hp
                if pl_raw_ratio > 0.05 and pl_residual_hp > 0.05:
                    results["Powerline_Interference"] = True

                # 4. Baseline drift — gated by absolute amplitude to skip
                #    low-amp leads where ratio is high but drift is tiny
                #    in raw units (e.g. v3_v4_flat at drift_std=80 is just
                #    respiratory residue, not diagnostic drift).
                if lf_ratio > 0.25 and drift_std > 130.0:
                    results["Baseline_Drift"] = True

            # Compute quality index based on method
            quality_idx = nk.ecg_quality(
                cleaned, sampling_rate=self.sampling_rate
            )
            # For averageQRS and other numeric methods
            quality_idx = np.array(quality_idx, dtype=np.float64)
            quality_score = float(np.nanmean(quality_idx))
            # Ensure the score is between 0 and 1
            quality_score = max(0.0, min(1.0, quality_score))
            results["nk_quality"] = quality_score

        except Exception as e:
            print(f"Error calculating power spectrum: {str(e)}")
            freqs = np.array([])
            psd = np.array([])

        # 5. SNR — signal-power vs HF-only noise. The notched analysis signal
        #    can still carry massive LF drift (e.g. Ara/1.ret II has 90% energy
        #    below 1 Hz). That drift is recoverable by any HP filter and
        #    shouldn't count as "noise" for diagnostic-quality purposes — only
        #    truly un-cleanable broadband HF noise (>40 Hz, excluding the
        #    powerline notch band) should.
        signal_amplitude = float(np.max(analysis_sig) - np.min(analysis_sig))
        sig_centered = analysis_sig - float(np.median(analysis_sig))
        b_bp, a_bp = butter(2, [0.5, 40], btype="bandpass", fs=self.sampling_rate, output="ba")  # type: ignore
        b_hp40, a_hp40 = butter(4, 40, btype="highpass", fs=self.sampling_rate, output="ba")  # type: ignore
        clean_signal = filtfilt(b_bp, a_bp, sig_centered)
        hf_noise = filtfilt(b_hp40, a_hp40, sig_centered)
        signal_power = float(np.var(clean_signal)) + 1e-12
        noise_power = float(np.var(hf_noise)) + 1e-12
        snr = 10 * np.log10(signal_power / noise_power)
        # Keep an "extracted noise" object available for backward compatibility
        noise = sig_centered - clean_signal  # noqa: F841

        results["SNR_dB"] = float(snr)
        if snr < 10:
            results["Low_SNR"] = True

        # QRS amplitude check
        results["QRS_Amplitude"] = float(signal_amplitude)

        return results

    @staticmethod
    def _harmonic_mean(values) -> float:
        """Harmonic mean robust to zero/near-zero entries."""
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return 0.0
        if np.any(arr <= 1e-9):
            return float(np.min(arr))
        return float(arr.size / np.sum(1.0 / arr))

    def _final_quality_for_segment(
        self,
        raw_seg: np.ndarray,
        clean_seg: np.ndarray,
        nk_fallback: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Per-2.5s-window per-lead quality.

        Returns dict with:
            tech     — flag-based score in [0, 1]
            nk_eff   — NeuroKit2 quality, clipped to [0, 1]
                       (whole-signal NK of this lead if per-window failed,
                        0.5 if both unavailable)
            final    — tech * (0.80 + 0.20 * nk_eff)
            flags    — raw flag dict (for inspection)
        """
        qres = self.analyze_lead_quality(raw_seg, clean_seg)
        tech = float(self._calculate_lead_quality_score(qres))
        any_flag_active = any(
            qres.get(k) for k in
            ("Muscle_Artifact", "Bad_Electrode_Contact",
             "Powerline_Interference", "Baseline_Drift", "Low_SNR")
        )

        # NK invalid policy: per-window NK often fails on 2.5 s (too few
        # R-peaks for template). Fall back to the lead's whole-signal NK
        # (more reliable, computed on the full record). If THAT also failed,
        # final fallback = 0.5 (neutral).
        nk_raw = qres.get("nk_quality")
        per_window_valid = (
            nk_raw is not None and not (isinstance(nk_raw, float) and np.isnan(nk_raw))
        )
        if per_window_valid:
            nk_eff = float(max(0.0, min(1.0, float(nk_raw))))
            # Low-NK flag on valid per-window NK: morphological problems that
            # flag-based detectors may miss (sub-threshold muscle noise etc.).
            if nk_raw < 0.5:
                tech = max(0.0, tech - 0.2)
        elif (
            nk_fallback is not None
            and isinstance(nk_fallback, (int, float))
            and nk_fallback == nk_fallback  # not NaN
        ):
            nk_eff = float(max(0.0, min(1.0, float(nk_fallback))))
            # When per-window NK is unavailable, the fallback (whole-signal
            # NK) is less specific to this 2.5s. Apply tech penalty ONLY if
            # another flag is also firing — this catches "real" issues
            # (v3_muscule_artifact V3: MA + NK fallback low → penalty) while
            # protecting clean simulator data (1.ret III: NK low but no flags
            # → no penalty, score stays high).
            if nk_fallback < 0.5 and any_flag_active:
                tech = max(0.0, tech - 0.2)
        else:
            nk_eff = 0.5

        final = tech * (0.80 + 0.20 * nk_eff)
        return {
            "tech": tech,
            "nk_eff": nk_eff,
            "final": float(np.clip(final, 0.0, 1.0)),
            "flags": qres,
        }

    def analyze_all_leads(
        self,
        data: LeadData,
        basic_window_sec: float = 2.5,
    ) -> Dict[str, Any]:
        """Windowed quality pipeline.

        Pipeline (per user's spec):
          1. Slice each lead into non-overlapping basic_window_sec segments.
          2. Per segment: tech_score + nk_eff → final = tech * (0.80 + 0.20*nk_eff).
          3. Sliding 5-sec window = pair of two consecutive basic segments
             (step = basic_window_sec, overlap = basic_window_sec).
          4. Per-lead 5-sec quality = harmonic_mean(q1, q2).
          5. Per-5-sec-window total = weighted_mean across 12 leads
             (same weighted formula as final score — no metric mismatch).
          6. best_window = argmax of weighted total over sliding 5-sec windows.
          7. whole-record total = mean of per-5-sec-window totals over the record.
          8. Penalty by weight of bad / unusable leads (per-lead = mean across
             sliding 5-sec windows):
                bad_weight       — leads with 0.40 ≤ q < 0.65
                unusable_weight  — leads with        q < 0.40
                overall *= 1 - 0.15*bad_weight - 0.35*unusable_weight
          9. Classification cap: if unusable_leads ≥ 3 → "Not usable".
        """
        if not data.get("lead_signals") or not data.get("cleaned_signals"):
            raise ValueError("Both lead_signals and cleaned_signals must be provided")

        raw_signals = data["lead_signals"]
        clean_signals = data["cleaned_signals"]
        lead_names = [
            l for l in raw_signals
            if l in clean_signals
            and raw_signals[l].size > 0
            and clean_signals[l].size > 0
        ]

        empty_result: Dict[str, Any] = {
            "lead_quality": {},
            "overall_quality": 0.0,
            "best_window_quality": 0.0,
            "best_window_time_s": None,
            "per_5s_window": [],
            "bad_leads": 0,
            "unusable_leads": 0,
            "classification": "Not usable",
            "problem_summary": [],
        }
        if not lead_names:
            return empty_result

        fs = self.sampling_rate
        wlen = int(basic_window_sec * fs)
        if wlen <= 0:
            return empty_result

        n_total = min(raw_signals[l].size for l in lead_names)
        nwin = n_total // wlen
        n_5s = max(0, nwin - 1)  # sliding pairs with step = basic_window_sec

        # --- Step 1-2: per-lead per-2.5s final quality + legacy whole-signal pass ---
        per_window_final: Dict[str, list] = {}
        lead_legacy: Dict[str, Dict[str, Any]] = {}

        for lead in lead_names:
            raw = raw_signals[lead]
            clean = clean_signals[lead]
            try:
                lead_legacy[lead] = self.analyze_lead_quality(raw, clean)
            except Exception as e:
                print(f"Warning: whole-signal analysis failed for {lead}: {e}")
                lead_legacy[lead] = {}

            # Whole-signal NK is reliable (full record gives NeuroKit enough
            # R-peaks); use it as fallback for per-2.5s windows where NK
            # routinely fails.
            nk_whole = lead_legacy.get(lead, {}).get("nk_quality")

            per_window_final[lead] = []
            for i in range(nwin):
                seg_raw = raw[i * wlen : (i + 1) * wlen]
                seg_clean = clean[i * wlen : (i + 1) * wlen]
                try:
                    r = self._final_quality_for_segment(
                        seg_raw, seg_clean, nk_fallback=nk_whole
                    )
                    per_window_final[lead].append(float(r["final"]))
                except Exception:
                    per_window_final[lead].append(0.0)

        # --- Step 3-5: sliding 5-sec windows ---
        per_5s_window: list = []
        weights_sum = sum(self.lead_weights.get(l, 0.08) for l in lead_names) or 1.0

        for s in range(n_5s):
            leads_q: Dict[str, float] = {}
            for lead in lead_names:
                qs = per_window_final[lead]
                if s + 1 < len(qs):
                    leads_q[lead] = self._harmonic_mean([qs[s], qs[s + 1]])
                else:
                    leads_q[lead] = 0.0
            total_5s = sum(
                self.lead_weights.get(l, 0.08) * q for l, q in leads_q.items()
            ) / weights_sum
            per_5s_window.append(
                {
                    "start_s": s * basic_window_sec,
                    "end_s": (s + 2) * basic_window_sec,
                    "leads": leads_q,
                    "total": float(np.clip(total_5s, 0.0, 1.0)),
                }
            )

        # --- Step 6: best 5-sec window by weighted total ---
        best_window_total = 0.0
        best_window_time = None
        if per_5s_window:
            totals = [w["total"] for w in per_5s_window]
            best_idx = int(np.argmax(totals))
            best_window_total = totals[best_idx]
            best_window_time = (
                per_5s_window[best_idx]["start_s"],
                per_5s_window[best_idx]["end_s"],
            )

        # --- Step 7: whole-record total = mean of per-5s totals ---
        record_total = (
            float(np.mean([w["total"] for w in per_5s_window]))
            if per_5s_window
            else 0.0
        )

        # --- Step 8: per-lead aggregate + NK whole-signal penalty ---
        # Per-5s leads_q above is the RAW harmonic mean — it reflects local
        # 5-sec quality without considering whole-record morphology.
        # The NK penalty applies HERE (record-level) because nk_quality is a
        # property of the entire signal (template built from all R-peaks);
        # collapsing it into per-window cells would mask the per-window data.
        # Severe NK (<0.5) always penalised. Moderate NK (0.5–0.65) gated by
        # "real-signal" check (median std>150) — protects simulator data
        # where NK systematically scores 0.4–0.6 even on perfect QRS.
        median_std = float(np.median([
            float(np.std(raw_signals[l].astype(float))) for l in lead_names
        ]))
        big_signal = median_std > 150.0

        lead_record_q: Dict[str, float] = {}
        for lead in lead_names:
            vals = [w["leads"].get(lead, 0.0) for w in per_5s_window]
            base = float(np.mean(vals)) if vals else 0.0
            nk_whole = lead_legacy.get(lead, {}).get("nk_quality")
            if (
                isinstance(nk_whole, (int, float))
                and nk_whole == nk_whole  # not NaN
            ):
                if nk_whole < 0.35:
                    base *= 0.40
                elif nk_whole < 0.50:
                    base *= 0.60
                elif nk_whole < 0.65 and big_signal:
                    base *= 0.40
            lead_record_q[lead] = base

        bad_weight = 0.0
        unusable_weight = 0.0
        bad_count = 0
        unusable_count = 0
        for lead, q in lead_record_q.items():
            w = self.lead_weights.get(lead, 0.08)
            if q < 0.40:
                unusable_weight += w
                unusable_count += 1
            elif q < 0.65:
                bad_weight += w
                bad_count += 1

        penalty_mult = max(0.0, 1.0 - 0.15 * bad_weight - 0.35 * unusable_weight)
        overall_quality = float(np.clip(record_total * penalty_mult, 0.0, 1.0))

        # --- Step 9: classification with hard cap ---
        if unusable_count >= 3:
            classification = "Not usable"
        elif overall_quality >= 0.85:
            classification = "Good"
        elif overall_quality >= 0.65:
            classification = "Questionable"
        else:
            classification = "Not usable"

        # Build per-lead output, preserving legacy keys used by the GUI
        lead_quality_out: Dict[str, Any] = {}
        for lead in lead_names:
            legacy = lead_legacy.get(lead, {})
            lead_quality_out[lead] = {
                **legacy,
                "whole_record_quality": lead_record_q[lead],
                "quality_5s_sliding": [
                    w["leads"].get(lead, 0.0) for w in per_5s_window
                ],
            }

        return {
            "lead_quality": lead_quality_out,
            "overall_quality": overall_quality,
            "record_total_unpenalized": float(record_total),
            "penalty_mult": float(penalty_mult),
            "best_window_quality": float(best_window_total),
            "best_window_time_s": best_window_time,
            "per_5s_window": per_5s_window,
            "bad_leads": bad_count,
            "unusable_leads": unusable_count,
            "classification": classification,
            "problem_summary": [],
        }

    def _calculate_lead_quality_score(self, quality_results: Dict) -> float:
        """Calculate a quality score between 0 and 1 for a lead."""
        score = 1.0
        if quality_results["Muscle_Artifact"]:
            score -= 0.2
        if quality_results["Bad_Electrode_Contact"]:
            score -= 0.4
        if quality_results["Powerline_Interference"]:
            score -= 0.2
        if quality_results["Baseline_Drift"]:
            score -= 0.2
        if quality_results["Low_SNR"]:
            score -= 0.3
        return max(0.0, score)

    def _generate_problem_description(
        self, lead_name: str, quality_results: Dict[str, Any]
    ) -> str:
        """Generate a user-friendly description of lead quality issues."""
        problems = []

        if quality_results["Bad_Electrode_Contact"]:
            problems.append("poor electrode contact")
        if quality_results["Muscle_Artifact"]:
            problems.append("muscle movement interference")
        if quality_results["Powerline_Interference"]:
            problems.append("electrical interference")
        if quality_results["Baseline_Drift"]:
            problems.append("baseline wandering")
        if quality_results["Low_SNR"]:
            problems.append("low signal quality")

        if problems:
            return f"Lead {lead_name}: {', '.join(problems)}"
        return ""
