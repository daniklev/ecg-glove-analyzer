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
    QLabel,
    QGroupBox,
    QCheckBox,
    QScrollArea,
    QSplitter,
    QMessageBox,
    QTabWidget,
    QToolButton,
    QComboBox,
)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.figure import Figure
import numpy as np
from test_ecg import EcgGlove
import math
from typing import Optional, Dict, Any

# Default settings
DEFAULT_SIGNAL_COLOR = "#00ffff"  # cyan
DEFAULT_GRID_COLOR = "#404040"  # dark gray
VOLTAGE_SCALE = 0.5  # mV per division


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
            QMainWindow { background-color: #2b2b2b; }
            QLabel { color: #e0e0e0; }
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
                border: 1px solid #505050;
                border-radius: 5px;
                margin-top: 20px;
            }
            QGroupBox::title {
                color: #e0e0e0;
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
            }
            QScrollArea {
                background-color: #2b2b2b;
                border: 1px solid #505050;
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

        # Quality Assessment
        quality_group = QGroupBox("Quality Assessment")
        quality_layout = QVBoxLayout()
        self.quality_method = QComboBox()
        self.quality_method.addItems(["averageQRS", "zhao2018"])
        self.quality_method.setCurrentText("averageQRS")  # Default to averageQRS
        quality_layout.addWidget(self.quality_method)
        quality_group.setLayout(quality_layout)

        # Add all settings groups
        nk_settings_layout.addWidget(clean_group)
        nk_settings_layout.addWidget(peak_group)
        nk_settings_layout.addWidget(quality_group)
        nk_settings.setLayout(nk_settings_layout)

        # Analysis button
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self.analyze_data)
        self.analyze_btn.setEnabled(False)

        # Add all elements to sidebar
        sidebar_layout.addWidget(file_group)
        sidebar_layout.addWidget(nk_settings)
        sidebar_layout.addWidget(self.analyze_btn)
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

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.file_list.clear()
            for file in os.listdir(folder):
                if file.endswith(".ret"):
                    self.file_list.addItem(os.path.join(folder, file))

    def file_selected(self):
        items = self.file_list.selectedItems()
        if items:
            self.current_file = items[0].text()
            self.analyze_btn.setEnabled(True)
        else:
            self.current_file = None
            self.analyze_btn.setEnabled(False)

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

    def analyze_data(self):
        if not self.current_file:
            return

        try:
            # Get current configuration
            config = {
                "clean_method": self.clean_method.currentText(),
                "peak_method": self.peak_method.currentText(),
                "quality_method": self.quality_method.currentText(),
            }

            # Create configuration key
            config_key = f"{self.current_file}_{config['clean_method']}_{config['peak_method']}_{config['quality_method']}"

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
            quality_method = self.quality_method.currentText()

            # Get quality scores and analyze
            tab.ecg_glove.quality_scores = tab.ecg_glove.compute_quality(
                clean_method=clean_method, quality_method=quality_method
            )

            results = tab.ecg_glove.analyze(
                clean_method=clean_method, peak_method=peak_method
            )

            # Show results in top info panel
            analysis_lead = results.get("AnalysisLead", "N/A")
            hr = results.get("HeartRate_BPM")

            result_text = f"Analysis Results:\n\n"
            result_text += f"Lead Used: {analysis_lead}\n"
            if hr is not None:
                result_text += f"Heart Rate: {hr:.1f} BPM\n"
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
            tab.ecg_glove.quality_scores
            if hasattr(tab.ecg_glove, "quality_scores")
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
                signal = tab.ecg_glove.lead_signals.get(lead, np.array([]))
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
                    # Add minimal lead label with quality score
                    quality_text = f"{lead}"
                    if lead in quality_scores:
                        score = quality_scores[lead]
                        if isinstance(score, float):
                            quality_text += f" ({score:.2f})"
                        else:
                            quality_text += f" ({score})"
                    ax.text(
                        0.02,
                        0.85,
                        quality_text,
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
