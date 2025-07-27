import os
import numpy as np
import streamlit as st
from decoder import ECGPacketDecoder
from filters import FilterConfig, apply_filters
from quality import analyze_ecg_all_leads
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64


# === Configuration ===
DATA_DIR = "/Users/danik/Developer/ecg-glove-analyzer/data"  # Directory with .ret files
SAMPLING_RATE = 500
BASE_CHANNELS = ["I", "III", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

st.title("ECG Quality Viewer")

# Initialize processed list
if "processed" not in st.session_state:
    st.session_state.processed = []  # each entry: {id, file, cfg, raw, filt}

# Sidebar: Add new ECG
st.sidebar.header("Load & Configure ECG")
files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".ret")]
selected = st.sidebar.selectbox("ECG File", files)
filter_type = st.sidebar.radio(
    "Primary Filter",
    ["Morphology (0)", "HP 0.15Hz (1)", "HP 0.05Hz (2)", "HP 0.5Hz (3)"],
    index=0,
)
filter_idx = ["Morphology (0)", "HP 0.15Hz (1)", "HP 0.05Hz (2)", "HP 0.5Hz (3)"].index(
    filter_type
)
notch_input = st.sidebar.text_input("Notch Frequencies (comma)", "50,100")
notch_freqs = [int(x) for x in notch_input.split(",") if x.strip().isdigit()]
spike = st.sidebar.checkbox("Spike Removal", value=True)
baseline = st.sidebar.checkbox("Baseline Correction", value=True)
human = st.sidebar.checkbox("Human Filter (40-100 Hz)", value=True)

if st.sidebar.button("Process ECG"):
    # Read binary data
    path = os.path.join(DATA_DIR, selected)
    with open(path, "rb") as f:
        raw_bytes = f.read()
    # Decode
    decoder = ECGPacketDecoder()
    leads = decoder.decode_bytes(raw_bytes)
    # Validate
    if not leads or all(len(arr) == 0 for arr in leads.values()):
        st.sidebar.error("Failed to decode ECG data.")
    else:
        # Filter config
        cfg = FilterConfig(
            sampling_rate=SAMPLING_RATE,
            notch_frequencies=notch_freqs,
            filter_type=filter_idx,
            spike_removal=spike,
            baseline_correction=baseline,
            human_filter=human,
        )
        # Raw and filtered data
        raw = {ch: leads[ch] for ch in BASE_CHANNELS}
        filt = {ch: apply_filters(raw[ch], cfg) for ch in BASE_CHANNELS}
        # Derive additional leads
        raw["II"] = raw["I"] + raw["III"]
        raw["aVR"] = -(raw["I"] + raw["II"]) / 2
        raw["aVL"] = raw["I"] - raw["II"] / 2
        raw["aVF"] = raw["II"] - raw["I"] / 2
        filt["II"] = filt["I"] + filt["III"]
        filt["aVR"] = -(filt["I"] + filt["II"]) / 2
        filt["aVL"] = filt["I"] - filt["II"] / 2
        filt["aVF"] = filt["II"] - filt["I"] / 2
        # Add to session
        new_id = len(st.session_state.processed)
        st.session_state.processed.append(
            {
                "id": new_id,
                "file": selected,
                "cfg": {
                    "filter_type": filter_idx,
                    "notch": notch_freqs,
                    "spike": spike,
                    "baseline": baseline,
                    "human": human,
                },
                "raw": raw,
                "filt": filt,
                "raw_bytes": raw_bytes,  # Store original binary data
            }
        )

# Tabbed view for processed ECGs
if st.session_state.processed:
    tab_labels = [
        f"{item['id']}: {item['file']}" for item in st.session_state.processed
    ]
    tabs = st.tabs(tab_labels)
    for tab, item in zip(tabs, st.session_state.processed):
        with tab:
            # Header with clipboard button and close button
            col1, col2, col3 = st.columns([4, 1, 1])
            with col1:
                st.subheader(f"File: {item['file']}")
            with col2:
                # Convert raw bytes to base64 (if available)
                if "raw_bytes" in item:
                    base64_data = base64.b64encode(item["raw_bytes"]).decode("utf-8")
                    st.button(
                        "📋", help="Copy raw base64 data", key=f"copy_{item['id']}"
                    )
                    if st.session_state.get(f"copy_{item['id']}", False):
                        st.code(base64_data, language=None)
                        st.success("Base64 data displayed above - copy manually")
                else:
                    st.button(
                        "📋",
                        help="Raw data not available (reprocess to enable)",
                        key=f"copy_{item['id']}",
                        disabled=True,
                    )
            # with col3:
            #     # Close button to remove from processed list
            #     if st.button("❌", key=f"close_{item['id']}"):
            #         st.session_state.processed.pop(item["id"])
            #         st.rerun()

            # Raw/Filtered switch
            view = st.radio("View:", ["Filtered", "Raw"], key=f"view_{item['id']}")
            data = item["raw"] if view == "Raw" else item["filt"]
            # Compute quality metrics on filtered data with 5-second result window
            quality = analyze_ecg_all_leads(item["filt"])

            # First table: Lead quality summary
            lead_quality_data = []
            for lead in LEAD_NAMES:
                lead_info = quality["lead_quality"][lead]
                quality_score = lead_info["QualityScore"]
                problems = lead_info.get("Problems", [])

                # # Flag problems based on quality metrics
                # problems = []
                # if quality_score < 0.7:
                #     problems.append("Poor Quality")
                # if lead_info.get("SNR", 0) < 10:
                #     problems.append("Low SNR")
                # if lead_info.get("PowerLineNoise", 0) > 0.1:
                #     problems.append("Power Line Noise")
                # if lead_info.get("BaselineWander", 0) > 0.2:
                #     problems.append("Baseline Wander")
                # if lead_info.get("SaturationPercentage", 0) > 5:
                #     problems.append("Saturation")

                problems_str = ", ".join(problems) if problems else "✓ Good"

                lead_quality_data.append(
                    {
                        "Lead": lead,
                        "Quality": f"{quality_score:.3f}",
                        # "QRS": f"{lead_info.get('QRS_Amplitude', 0):.2f}",
                        "Status": (
                            "🔴 Poor"
                            if quality_score < 0.65
                            else "🟡 Fair" if quality_score < 0.85 else "🟢 Good"
                        ),
                        "Problems": problems_str,
                    }
                )

            # Second table: Detailed window-by-window analysis
            window_sec = 2.5  # Same as in quality.py
            wlen = int(window_sec * SAMPLING_RATE)
            n = len(data["I"])
            nwin = n // wlen if wlen > 0 else 0

            # Create window analysis data structure
            window_analysis = {}

            # Initialize dataframe variables to prevent "possibly unbound" errors
            quality_df_data = []
            problems_df_data = []
            snr_df_data = []
            qrs_amp_df_data = []
            muscle_art_df_data = []
            bad_contact_df_data = []
            power_int_df_data = []
            baseline_drift_df_data = []

            for lead in LEAD_NAMES:
                window_analysis[lead] = {}
                sig = item["filt"][lead]  # Always use filtered data for analysis
                for i in range(nwin):
                    seg = sig[i * wlen : (i + 1) * wlen]
                    # Import analyze_lead_quality function
                    from quality import analyze_lead_quality, compute_quality_score

                    metrics = analyze_lead_quality(seg, SAMPLING_RATE)
                    flags = metrics["flags"]
                    quality_score = compute_quality_score(flags)

                    # Format flags as problem indicators
                    active_problems = [flag for flag, active in flags.items() if active]
                    problems_str = (
                        ", ".join(active_problems) if active_problems else "✓ Good"
                    )

                    window_key = f"W{i+1}\n({i*window_sec:.1f}-{(i+1)*window_sec:.1f}s)"
                    window_analysis[lead][window_key] = {
                        "quality": f"{quality_score:.3f}",
                        "problems": problems_str,
                        "qrs_amp": f"{metrics['values']['qrs_amp']:.2f}",
                        "snr": f"{metrics['values']['snr']:.2f}/{flags.get('Low_SNR', 0):.2f}",
                        "muscle_art": f"{metrics['values']['m_a']:.3f}/{flags.get('Muscle_Artifact', 0):.3f}",
                        "bad_contact": f"{metrics['values']['b_e_c']:.3f}/{flags.get('Bad_Electrode_Contact', 0):.3f}",
                        "power_int": f"{metrics['values']['p_i']:.3f}/{flags.get('Powerline_Interference', 0):.3f}",
                        "baseline_dr": f"{metrics['values']['b_d']:.3f}/{flags.get('Baseline_Drift', 0):.3f}",
                    }

            # Create dataframes for different metrics
            if nwin > 0:
                window_cols = [
                    f"W{i+1}\n({i*window_sec:.1f}-{(i+1)*window_sec:.1f}s)"
                    for i in range(nwin)
                ]

                # Quality Score DataFrame
                quality_df_data = []
                for lead in LEAD_NAMES:
                    row = {"Lead": lead}
                    for window_key in window_cols:
                        row[window_key] = (
                            window_analysis[lead]
                            .get(window_key, {})
                            .get("quality", "N/A")
                        )
                    quality_df_data.append(row)

                # Problems DataFrame
                problems_df_data = []
                for lead in LEAD_NAMES:
                    row = {"Lead": lead}
                    for window_key in window_cols:
                        row[window_key] = (
                            window_analysis[lead]
                            .get(window_key, {})
                            .get("problems", "N/A")
                        )
                    problems_df_data.append(row)

                # SNR DataFrame
                snr_df_data = []
                for lead in LEAD_NAMES:
                    row = {"Lead": lead}
                    for window_key in window_cols:
                        row[window_key] = (
                            window_analysis[lead].get(window_key, {}).get("snr", "N/A")
                        )
                    snr_df_data.append(row)

                # QRS Amplitude DataFrame
                qrs_amp_df_data = []
                for lead in LEAD_NAMES:
                    row = {"Lead": lead}
                    for window_key in window_cols:
                        row[window_key] = (
                            window_analysis[lead]
                            .get(window_key, {})
                            .get("qrs_amp", "N/A")
                        )
                    qrs_amp_df_data.append(row)

                # Muscle Artifact DataFrame
                muscle_art_df_data = []
                for lead in LEAD_NAMES:
                    row = {"Lead": lead}
                    for window_key in window_cols:
                        row[window_key] = (
                            window_analysis[lead]
                            .get(window_key, {})
                            .get("muscle_art", "N/A")
                        )
                    muscle_art_df_data.append(row)

                # Bad Contact DataFrame
                bad_contact_df_data = []
                for lead in LEAD_NAMES:
                    row = {"Lead": lead}
                    for window_key in window_cols:
                        row[window_key] = (
                            window_analysis[lead]
                            .get(window_key, {})
                            .get("bad_contact", "N/A")
                        )
                    bad_contact_df_data.append(row)

                # Power Interference DataFrame
                power_int_df_data = []
                for lead in LEAD_NAMES:
                    row = {"Lead": lead}
                    for window_key in window_cols:
                        row[window_key] = (
                            window_analysis[lead]
                            .get(window_key, {})
                            .get("power_int", "N/A")
                        )
                    power_int_df_data.append(row)

                # Baseline Drift DataFrame
                baseline_drift_df_data = []
                for lead in LEAD_NAMES:
                    row = {"Lead": lead}
                    for window_key in window_cols:
                        row[window_key] = (
                            window_analysis[lead]
                            .get(window_key, {})
                            .get("baseline_dr", "N/A")
                        )
                    baseline_drift_df_data.append(row)

            # Create Plotly figure
            t = np.arange(len(data["I"])) / SAMPLING_RATE
            fig = make_subplots(rows=3, cols=4, subplot_titles=LEAD_NAMES)

            # Calculate x-axis range based on best quality windows
            window_sec = 2.5  # Same as in quality.py
            if "best_windows_used" in quality and quality["best_windows_used"]:
                best_windows = quality["best_windows_used"]
                # Calculate time range for the best consecutive windows (5-second result window)
                result_start_time = min(best_windows) * window_sec
                result_end_time = max(best_windows) * window_sec + window_sec
                # Add small buffer around the best window
                buffer_time = 1  # 1 second buffer on each side
                x_range = [
                    max(0, result_start_time - buffer_time),
                    min(t[-1], result_end_time + buffer_time),
                ]
            else:
                # Fallback to default 3 seconds if no best windows available
                x_range = [0, min(3, t[-1])]

            # Calculate y-ranges centered around baseline
            y_ranges = {}
            for lead in LEAD_NAMES:
                # get min and max values for the lead
                lead_data = data[lead][
                    int(x_range[0] * SAMPLING_RATE) : int(x_range[1] * SAMPLING_RATE)
                ]
                lead_min = np.min(lead_data)
                lead_max = np.max(lead_data)
                # Calculate range centered around baseline (0)
                baseline = (lead_min + lead_max) / 2
                range_margin = max(abs(lead_min - baseline), abs(lead_max - baseline))
                y_ranges[lead] = [
                    lead_min,
                    lead_max,
                ]

            # Plot each lead
            for idx, lead in enumerate(LEAD_NAMES):
                r = idx // 4 + 1
                c = idx % 4 + 1
                fig.add_trace(
                    go.Scatter(x=t, y=data[lead], name=lead, mode="lines"), row=r, col=c
                )
                # Set y-axis range centered around baseline
                fig.update_yaxes(range=y_ranges[lead], row=r, col=c)
                # Set x-axis range for 3 seconds
                fig.update_xaxes(range=x_range, row=r, col=c)

            fig.update_xaxes(matches="x")
            fig.update_layout(
                autosize=True,
                height=600,
                showlegend=False,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"responsive": True},
                key=f"plot_{item['id']}",
            )

            # Display quality metrics
            st.write("**Total Quality:**", quality["total_quality"])
            st.write("**Classification:**", quality["classification"])

            # Display best windows used for quality calculation
            if "best_windows_used" in quality and quality["best_windows_used"]:
                window_sec = 2.5  # Same as in quality.py
                best_windows = quality["best_windows_used"]

                # Calculate time ranges for best consecutive windows (5-second result window)
                time_ranges = []
                for window_idx in best_windows:
                    start_time = window_idx * window_sec
                    end_time = (window_idx + 1) * window_sec
                    time_ranges.append(f"{start_time:.1f}-{end_time:.1f}s")

                # Calculate overall time frame for the 5-second result window
                result_start_time = min(best_windows) * window_sec
                result_end_time = max(best_windows) * window_sec + window_sec

                st.write(
                    "**Best Quality Result Window (5 seconds):**",
                    ", ".join(
                        [
                            f"W{idx+1} ({time_range})"
                            for idx, time_range in zip(best_windows, time_ranges)
                        ]
                    ),
                )
                st.write(
                    "**Result Window Time Frame:**",
                    f"{result_start_time:.1f}-{result_end_time:.1f}s (consecutive {result_end_time - result_start_time:.1f}s window)",
                )
            else:
                st.write(
                    "**Best Quality Result Window:**",
                    "No consecutive windows available for 5-second analysis",
                )

            # Display tables
            st.subheader("Lead Quality Summary")
            st.table(lead_quality_data)

            st.subheader("Detailed Window Analysis")
            if nwin > 0:
                st.write("**Quality Scores by Window:**")
                st.dataframe(quality_df_data, use_container_width=True)

                st.write("**Problems by Window:**")
                st.dataframe(problems_df_data, use_container_width=True)

                st.write("**QRS Amplitude by Window:**")
                st.dataframe(qrs_amp_df_data, use_container_width=True)

                st.write("**SNR (dB) by Window:** (snr db value/ flag % value)")
                st.dataframe(snr_df_data, use_container_width=True)

                st.write("**Muscle Artifact by Window:** (m_a value/ flag % value)")
                st.dataframe(muscle_art_df_data, use_container_width=True)

                st.write("**Bad Contact by Window:** (b_e_c value/ flag % value)")
                st.dataframe(bad_contact_df_data, use_container_width=True)

                st.write("**Power Interference by Window:** (p_i value/ flag % value)")
                st.dataframe(power_int_df_data, use_container_width=True)

                st.write("**Baseline Drift by Window:** (b_d value/ flag % value)")
                st.dataframe(baseline_drift_df_data, use_container_width=True)
            else:
                st.write("No windows available for analysis.")

            # Display filter config
            st.write("**Filter Config:**", item["cfg"])
