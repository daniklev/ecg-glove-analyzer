import os
import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QGroupBox,
    QSplitter,
    QMessageBox,
    QTabWidget,
    QComboBox,
)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.figure import Figure
import numpy as np
from ecg_glove import EcgGlove
from typing import Optional, Dict, Any

# Role to store full file path in QListWidgetItem data
USER_ROLE = 32  # Qt.UserRole value

# Default settings
DEFAULT_SIGNAL_COLOR = "#00ffff"  # cyan
DEFAULT_GRID_COLOR = "#404040"  # dark gray
VOLTAGE_SCALE = 0.5  # mV per division
APP_VERSION = "1.0.2"


class CollapsibleBox(QGroupBox):
    def __init__(self, title="", parent=None):
        super().__init__(title, parent)
        self.setCheckable(True)
        self.setChecked(True)
        self.toggled.connect(self._on_toggled)
        self.content = QWidget(self)
        self.content_layout = QVBoxLayout()
        self.content.setLayout(self.content_layout)
        layout = QVBoxLayout(self)
        layout.addWidget(self.content)

    def _on_toggled(self, checked):
        self.content.setVisible(checked)

    def addWidget(self, widget):
        self.content_layout.addWidget(widget)


class EcgTab(QWidget):
    def __init__(
        self, filepath: str, config: Optional[Dict[str, Any]] = None, parent=None
    ):
        super().__init__(parent)
        self.filepath = filepath
        self.config = config or {}
        self.ecg_glove: Optional[EcgGlove] = None
        self.axes = []
        self._syncing = False

        # Create layout
        layout = QVBoxLayout(self)

        # Create top info section with horizontal layout
        top_info = QWidget()
        top_info_layout = QHBoxLayout(top_info)

        # Settings display
        self.settings_label = QLabel()
        self.settings_label.setStyleSheet(
            "background-color: #3b3b3b; padding: 10px; border-radius: 5px; margin: 5px;"
        )
        self.settings_label.setMinimumWidth(300)
        self.update_settings_display()

        # Results display
        self.results_text = QLabel()
        self.results_text.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.results_text.setStyleSheet(
            "background-color: #3b3b3b; padding: 10px; border-radius: 5px; margin: 5px;"
        )
        self.results_text.setMinimumWidth(300)

        top_info_layout.addWidget(self.settings_label)
        top_info_layout.addWidget(self.results_text)

        # Add matplotlib figure with navigation toolbar
        self.figure = Figure(figsize=(12, 8), facecolor="#2b2b2b")
        # self.canvas = FigureCanvas(self.figure)
        # self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout.addWidget(top_info)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def _sync_xlim(self, ax):
        """Synchronize x-axis limits across all plots."""
        if self._syncing:
            return

        try:
            self._syncing = True
            xlim = ax.get_xlim()
            for other_ax in self.axes:
                if other_ax != ax:
                    other_ax.set_xlim(xlim)
            self.canvas.draw_idle()
        finally:
            self._syncing = False

    def _sync_ylim(self, ax):
        """Synchronize y-axis limits across all plots."""
        if self._syncing:
            return

        try:
            self._syncing = True
            ylim = ax.get_ylim()
            for other_ax in self.axes:
                if other_ax != ax:
                    other_ax.set_ylim(ylim)
            self.canvas.draw_idle()
        finally:
            self._syncing = False

    def update_settings_display(self):
        """Update the settings display label with current configuration"""
        if self.config:
            settings_text = "Analysis Configuration:\n"
            settings_text += (
                f"Cleaning Method: {self.config.get('clean_method', 'neurokit')}\n"
            )
            settings_text += (
                f"Peak Detection: {self.config.get('peak_method', 'neurokit')}\n"
            )
            settings_text += (
                f"Quality Assessment: {self.config.get('quality_method', 'averageQRS')}"
            )
            self.settings_label.setText(settings_text)
        else:
            self.settings_label.setText("No analysis performed yet")

    def get_configuration_name(self):
        """Generate a unique name for the current configuration"""
        if not self.config:
            return "Default"
        clean = self.config.get("clean_method", "neurokit")
        peak = self.config.get("peak_method", "neurokit")
        quality = self.config.get("quality_method", "averageQRS")
        return f"{clean}-{peak}-{quality}"


class EcgAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECG Analyzer")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize attributes
        self.current_file = None
        self.tabs = {}  # Store tabs by filepath
        self.lead_checks = {}
        self.filter_checks = {}

        # Set dark theme for the application
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #2b2b2b;
                color: #e0e0e0;
            }
            QLabel {
                color: #e0e0e0;
                background-color: transparent;
            }
            QPushButton {
                background-color: #3b3b3b;
                color: #e0e0e0;
                border: 1px solid #505050;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #454545;
            }
            QGroupBox {
                color: #e0e0e0;
                background-color: #2b2b2b;
                border: 1px solid #505050;
                border-radius: 5px;
                margin-top: 20px;
            }
            QGroupBox::title {
                color: #e0e0e0;
                background-color: transparent;
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 5px;
                margin-left: 8px;
            }
            QComboBox {
                background-color: #3b3b3b;
                color: #e0e0e0;
                border: 1px solid #505050;
                border-radius: 3px;
                padding: 5px;
                margin-top: 5px;
            }
            QListWidget {
                background-color: #2b2b2b;
                color: #e0e0e0;
                border: 1px solid #505050;
            }
            QCheckBox {
                color: #e0e0e0;
                background-color: transparent;
            }
            QScrollArea {
                background-color: #2b2b2b;
                border: 1px solid #505050;
            }
            QTabWidget::pane {
                background-color: #2b2b2b;
                border: 1px solid #505050;
            }
            QTabBar::tab {
                background-color: #3b3b3b;
                color: #e0e0e0;
                padding: 5px;
                border: 1px solid #505050;
            }
            QTabBar::tab:selected {
                background-color: #454545;
            }
            """
        )
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Create splitter for sidebar and main area
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Create and setup sidebar
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar.setMaximumWidth(300)

        # File selection group
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()

        self.select_folder_btn = QPushButton("Select Folder")
        self.select_folder_btn.clicked.connect(self.select_folder)

        self.file_list = QListWidget()
        self.file_list.itemSelectionChanged.connect(self.file_selected)

        file_layout.addWidget(self.select_folder_btn)
        file_layout.addWidget(self.file_list)
        file_group.setLayout(file_layout)

        # Neurokit settings section
        nk_settings = QGroupBox("Settings")
        nk_settings_layout = QVBoxLayout()

        # ECG Cleaning Method
        clean_group = QGroupBox("ECG Cleaning Method")
        clean_layout = QVBoxLayout()
        self.clean_method = QComboBox()
        self.clean_method.addItems(["neurokit", "biosppy", "pantompkins", "hamilton"])
        self.clean_method.setCurrentText("neurokit")
        clean_layout.addWidget(self.clean_method)
        clean_group.setLayout(clean_layout)

        # R-Peak Detection
        peak_group = QGroupBox("R-Peak Detection")
        peak_layout = QVBoxLayout()
        self.peak_method = QComboBox()
        self.peak_method.addItems(
            [
                "neurokit",
                "biosppy",
                "gamboa",
                "promac",
                "emrich2023",
                "rodrigues2021",
                "kalidas2017",
                "manikandan2012",
                "nabian2018",
                "engzeemod2012",
                "elgendi2010",
                "gamboa2008",
                "christov2004",
                "martinez2004",
                "zong2003",
                "hamilton2002",
                "pantompkins1985",
            ]
        )
        self.peak_method.setCurrentText("neurokit")  # Default to neurokit
        peak_layout.addWidget(self.peak_method)
        peak_group.setLayout(peak_layout)

        # Add all settings groups
        nk_settings_layout.addWidget(clean_group)
        nk_settings_layout.addWidget(peak_group)
        nk_settings.setLayout(nk_settings_layout)

        # Analysis button
        self.process_btn = QPushButton("Process")
        self.process_btn.clicked.connect(self.process_data)
        self.process_btn.setEnabled(False)

        # Add all elements to sidebar
        sidebar_layout.addWidget(file_group)
        sidebar_layout.addWidget(nk_settings)
        sidebar_layout.addWidget(self.process_btn)
        sidebar_layout.addStretch()

        # Create main display area with tabs
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)

        # Set dark theme for plots
        plt.style.use("dark_background")

        # Add widgets to splitter
        splitter.addWidget(sidebar)
        splitter.addWidget(self.tab_widget)

        # Set initial splitter sizes
        splitter.setSizes([300, 900])

        # Add version label
        version_label = QLabel(f"ECG Analyzer {APP_VERSION}")
        version_label.setStyleSheet("color: #808080; padding: 5px;")  # Gray color
        version_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        sidebar_layout.addWidget(version_label)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.file_list.clear()
            # list and sort by filename, display basename only
            ret_files = sorted([f for f in os.listdir(folder) if f.endswith('.ret')])
            for fname in ret_files:
                item = QListWidgetItem(fname)
                # store full path for later retrieval
                item.setData(USER_ROLE, os.path.join(folder, fname))
                self.file_list.addItem(item)

    def file_selected(self):
        items = self.file_list.selectedItems()
        if items:
            # retrieve full path stored in item data
            self.current_file = items[0].data(USER_ROLE)
            self.process_btn.setEnabled(True)
        else:
            self.current_file = None
            self.process_btn.setEnabled(False)

    def get_selected_leads(self):
        return [lead for lead, cb in self.lead_checks.items() if cb.isChecked()]

    def close_tab(self, index):
        tab = self.tab_widget.widget(index)
        # Find the key in self.tabs that corresponds to this tab
        key_to_remove = None
        for key, stored_tab in self.tabs.items():
            if stored_tab == tab:
                key_to_remove = key
                break
        if key_to_remove:
            del self.tabs[key_to_remove]
        self.tab_widget.removeTab(index)

    def process_data(self):
        if not self.current_file:
            return

        try:
            # Get current configuration
            config = {
                "clean_method": self.clean_method.currentText(),
                "peak_method": self.peak_method.currentText(),
            }

            # Create configuration key
            config_key = (
                f"{self.current_file}_{config['clean_method']}_{config['peak_method']}"
            )

            # Create new tab if this configuration doesn't exist
            if config_key not in self.tabs:
                tab = EcgTab(self.current_file, config=config)
                self.tabs[config_key] = tab
                tab_name = f"{os.path.basename(self.current_file)} ({tab.get_configuration_name()})"
                self.tab_widget.addTab(tab, tab_name)
            else:
                tab = self.tabs[config_key]

            # Load and decode data
            with open(self.current_file, "rb") as f:
                data_bytes = f.read()

            tab.ecg_glove = EcgGlove(sampling_rate=500)
            tab.ecg_glove.decode_data(data_bytes)

            # Get selected methods from dropdowns
            clean_method = self.clean_method.currentText()
            peak_method = self.peak_method.currentText()

            # Analyze quality first
            quality_results = tab.ecg_glove.compute_quality(clean_method=clean_method)

            # Store quality scores and measurement results
            tab.quality_scores = quality_results

            # Analyze ECG if quality is acceptable
            results = tab.ecg_glove.process(
                clean_method=clean_method, peak_method=peak_method
            )

            # Format results for display
            # result_text = "Analysis Results:\n\n"
            result_text = f"Analysis Lead: {results['AnalysisLead']}\n\n"

            # Add measurements
            if "measurements" in results:
                result_text += "ECG Measurements:\n"
                measurements = results["measurements"]
                if measurements.get("HeartRate_BPM"):
                    result_text += (
                        f"Heart Rate: {measurements['HeartRate_BPM']:.1f} BPM\n"
                    )
                if measurements.get("RR_Interval_ms"):
                    result_text += (
                        f"RR Interval: {measurements['RR_Interval_ms']:.0f} ms\n"
                    )
                if measurements.get("P_Duration_ms"):
                    result_text += (
                        f"P Duration: {measurements['P_Duration_ms']:.0f} ms\n"
                    )
                if measurements.get("PR_Interval_ms"):
                    result_text += (
                        f"PR Interval: {measurements['PR_Interval_ms']:.0f} ms\n"
                    )
                if measurements.get("QRS_Duration_ms"):
                    result_text += (
                        f"QRS Duration: {measurements['QRS_Duration_ms']:.0f} ms\n"
                    )
                if measurements.get("QT_Interval_ms"):
                    result_text += (
                        f"QT Interval: {measurements['QT_Interval_ms']:.0f} ms\n"
                    )
                if measurements.get("QTc_Interval_ms"):
                    result_text += (
                        f"QTc Interval: {measurements['QTc_Interval_ms']:.0f} ms\n"
                    )

            # Add overall quality results
            if "overall_quality" in quality_results:
                overall_quality = quality_results["overall_quality"]
                result_text += f"\nOverall Signal Quality: {overall_quality:.2f}\n"

            #     # Add quality warnings if any
            #     if quality_results.get("problem_summary"):
            #         result_text += "\nQuality Issues:\n"
            #         for problem in quality_results["problem_summary"]:
            #             result_text += f"- {problem}\n"

            tab.results_text.setText(result_text)

            # Update plots
            self.plot_ecg_data(tab)

            # Switch to the tab
            self.tab_widget.setCurrentWidget(tab)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")

    def plot_ecg_data(self, tab):
        if not tab.ecg_glove:
            return
        # Skip plotting if no signal data available
        # if not any(arr.size > 0 for arr in tab.ecg_glove.lead_signals.values()):
        if not any(arr.size > 0 for arr in tab.ecg_glove.cleaned_signals.values()):
            tab.figure.clear()
            tab.canvas.draw_idle()
            return

        tab.figure.clear()

        # Fixed lead order for 6x2 layout
        lead_order = [
            ("I", "V1"),
            ("II", "V2"),
            ("III", "V3"),
            ("aVR", "V4"),
            ("aVL", "V5"),
            ("aVF", "V6"),
        ]

        # Get quality scores
        quality_scores = (
            tab.quality_scores.get("lead_quality", {})
            if hasattr(tab, "quality_scores")
            else {}
        )

        # Create figure with 6 rows and 2 columns
        tab.axes = []
        y_min = float("inf")
        y_max = float("-inf")
        signals_data = {}

        # Pre-calculate signal data and find ranges
        for row_leads in lead_order:
            for lead in row_leads:
                # signal = tab.ecg_glove.lead_signals.get(lead, np.array([]))
                signal = tab.ecg_glove.cleaned_signals.get(lead, np.array([]))
                if signal.size > 0:
                    # Downsample for very large signals (> 10000 points)
                    if signal.size > 10000:
                        downsample_factor = signal.size // 10000 + 1
                        signal = signal[::downsample_factor]

                    times = np.arange(signal.size) / tab.ecg_glove.sampling_rate
                    y_min = min(y_min, np.min(signal))
                    y_max = max(y_max, np.max(signal))
                    signals_data[lead] = (times, signal)
                else:
                    signals_data[lead] = (np.array([]), np.array([]))

        # Add margins to y-axis limits
        y_range = y_max - y_min
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range

        # Set up the figure for maximum space utilization
        tab.figure.subplots_adjust(
            left=0.02, right=0.98, bottom=0.02, top=0.98, hspace=0.1, wspace=0.1
        )

        # Create subplots with minimal styling
        first_ax = None
        for row, (left_lead, right_lead) in enumerate(lead_order):
            # Left plot
            if first_ax is None:
                ax_left = tab.figure.add_subplot(6, 2, 2 * row + 1)
                first_ax = ax_left
            else:
                ax_left = tab.figure.add_subplot(
                    6, 2, 2 * row + 1, sharex=first_ax, sharey=first_ax
                )

            # Right plot
            ax_right = tab.figure.add_subplot(
                6, 2, 2 * row + 2, sharex=first_ax, sharey=first_ax
            )

            tab.axes.extend([ax_left, ax_right])

            # Configure axes for maximum signal visibility
            for ax, lead in [(ax_left, left_lead), (ax_right, right_lead)]:
                times, signal = signals_data[lead]
                if signal.size > 0:
                    ax.plot(
                        times,
                        signal,
                        color=DEFAULT_SIGNAL_COLOR,
                        linewidth=0.8,
                        antialiased=True,
                    )
                    # Add lead label with quality information
                    if lead in quality_scores:
                        problems = []
                        lead_quality = quality_scores[lead]
                        quality_text = (
                            f"{lead} ({lead_quality.get('nk_quality', 'N/A'):.2f})"
                        )

                        if lead_quality.get("Low_SNR"):
                            color = "#ff6b6b"  # Red for poor quality
                        elif lead_quality.get("Muscle_Artifact") or lead_quality.get(
                            "Powerline_Interference"
                        ):
                            color = "#ffd93d"  # Yellow for moderate issues
                        else:
                            color = "#6bff6b"  # Green for good quality

                        if lead_quality.get("Muscle_Artifact"):
                            problems.append("MA")
                        if lead_quality.get("Powerline_Interference"):
                            problems.append("PI")
                        if lead_quality.get("Baseline_Drift"):
                            problems.append("BD")
                        if lead_quality.get("Bad_Electrode_Contact"):
                            problems.append("EC")

                        if problems:
                            quality_text += f" [{', '.join(problems)}]"

                        ax.text(
                            0.02,
                            0.85,
                            quality_text,
                            transform=ax.transAxes,
                            fontsize=8,
                            color=color,
                            alpha=0.8,
                        )
                    else:
                        ax.text(
                            0.02,
                            0.85,
                            lead,
                            transform=ax.transAxes,
                            fontsize=8,
                            color="white",
                            alpha=0.8,
                        )

                # Remove all unnecessary elements
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(True, alpha=0.1, color=DEFAULT_GRID_COLOR)
                for spine in ax.spines.values():
                    spine.set_visible(False)

        # Set limits for all plots
        if first_ax:
            first_ax.set_xlim(
                0,
                max(times[-1] for times, _ in signals_data.values() if times.size > 0),
            )
            first_ax.set_ylim(y_min, y_max)

        # No need for tight_layout since we're using subplots_adjust
        tab.canvas.draw_idle()


def main():
    app = QApplication(sys.argv)
    window = EcgAnalyzerGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
