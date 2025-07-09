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
            # Compute quality metrics on filtered data
            quality = analyze_ecg_all_leads(item["filt"], sampling_rate=SAMPLING_RATE)

            # Detailed per-lead table with problems flagged
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
                        "SNR": f"{lead_info.get('SNR_dB', 0):.2f}",
                        "QRS": f"{lead_info.get('QRS_Amplitude', 0):.2f}",
                        "MA": f"{lead_info.get('m_a', 0):.2f}",
                        "BEC": f"{lead_info.get('b_e_c', 0):.2f}",
                        "PI": f"{lead_info.get('p_i', 0):.2f}",
                        "BD": f"{lead_info.get('b_d', 0):.2f}",
                        "Status": (
                            "🔴 Poor"
                            if quality_score < 0.5
                            else "🟡 Fair" if quality_score < 0.7 else "🟢 Good"
                        ),
                        "Problems": problems_str,
                    }
                )

            # Create Plotly figure
            t = np.arange(len(data["I"])) / SAMPLING_RATE
            fig = make_subplots(rows=3, cols=4, subplot_titles=LEAD_NAMES)

            # Set initial x-axis range (3 seconds)
            # Check if full screen is enabled
            x_range = [0, 3]  # Default to 3 seconds
            if "full_screen" in st.session_state and st.session_state["full_screen"]:
                # If full screen, extend to 5 seconds
                st.session_state["full_screen"] = True
                x_range = [0, 5]

            # Calculate y-ranges centered around baseline
            y_ranges = {}
            for lead in LEAD_NAMES:
                baseline = data[lead][0]  # Use first value as baseline
                y_ranges[lead] = (baseline - 500, baseline + 500)

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

            # Display quality metrics - st.write doesn't accept key parameter
            st.write("**Total Quality:**", quality["total_quality"])
            st.write("**Classification:**", quality["classification"])
            st.table(lead_quality_data)
            st.write("**Filter Config:**", item["cfg"])
