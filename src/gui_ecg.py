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
    QCheckBox,
    QScrollArea,
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
APP_VERSION = "1.0.3"


class CollapsibleBox(QGroupBox):
    def __init__(self, title="", parent=None):
        super().__init__(title, parent)
        self.setCheckable(True)
        self.setChecked(True)
        self.toggled.connect(self._on_toggled)

        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 20, 5, 5)  # Top margin for title

        # Create content widget
        self.content = QWidget(self)
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(5, 5, 5, 5)

        main_layout.addWidget(self.content)

        # Set initial state
        self._on_toggled(self.isChecked())

    def _on_toggled(self, checked):
        self.content.setVisible(checked)
        if checked:
            self.setMaximumHeight(16777215)  # Reset max height
        else:
            # Collapse to title height only
            self.setMaximumHeight(30)

    def addWidget(self, widget):
        self.content_layout.addWidget(widget)

    def addLayout(self, layout):
        self.content_layout.addLayout(layout)


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
        self.signals_data = {}  # Store signal data for y-limit calculations

        # Create layout
        layout = QVBoxLayout(self)

        # Create header with aligned titles
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(5, 5, 5, 10)
        header_layout.setSpacing(10)

        # Signal type section
        signal_type_widget = QWidget()
        signal_type_layout = QVBoxLayout(signal_type_widget)
        signal_type_layout.setContentsMargins(0, 0, 0, 0)

        signal_type_title = QLabel("Signal Type")
        signal_type_title.setStyleSheet(
            "font-weight: bold; color: #e0e0e0; font-size: 12px; margin-bottom: 5px;"
        )
        signal_type_title.setFixedHeight(20)  # Fixed height for alignment

        self.signal_type_combo = QComboBox()
        self.signal_type_combo.addItems(["Raw", "Filtered", "Cleaned"])
        self.signal_type_combo.setCurrentText("Cleaned")
        self.signal_type_combo.currentTextChanged.connect(self.update_plot)
        self.signal_type_combo.setFixedHeight(30)
        self.signal_type_combo.setMinimumWidth(80)
        self.signal_type_combo.setStyleSheet(
            """
            QComboBox {
                background-color: #3b3b3b;
                color: #e0e0e0;
                border: 1px solid #505050;
                border-radius: 3px;
                padding: 2px 5px;
                font-size: 11px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid #e0e0e0;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #3b3b3b;
                color: #e0e0e0;
                border: 1px solid #505050;
                selection-background-color: #4a9eff;
            }
        """
        )

        signal_type_layout.addWidget(signal_type_title)
        signal_type_layout.addWidget(self.signal_type_combo)
        signal_type_layout.addStretch()
        signal_type_widget.setMaximumWidth(
            120
        )  # Increased from 150 to accommodate text

        # Settings section
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        settings_layout.setContentsMargins(0, 0, 0, 0)

        settings_title = QLabel("Analysis Configuration")
        settings_title.setStyleSheet(
            "font-weight: bold; color: #e0e0e0; font-size: 12px; margin-bottom: 5px;"
        )
        settings_title.setFixedHeight(20)  # Fixed height for alignment

        # Create scrollable area for settings
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        settings_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        settings_scroll.setMaximumHeight(130)
        settings_scroll.setStyleSheet("QScrollArea { border: none; }")

        self.settings_label = QLabel()
        self.settings_label.setStyleSheet(
            "background-color: #3b3b3b; padding: 10px; border-radius: 5px; "
            "font-size: 11px; line-height: 1.2;"
        )
        self.settings_label.setWordWrap(True)
        self.settings_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.update_settings_display()

        settings_scroll.setWidget(self.settings_label)

        settings_layout.addWidget(settings_title)
        settings_layout.addWidget(settings_scroll)

        # Results section
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(0, 0, 0, 0)

        results_title = QLabel("Analysis Results")
        results_title.setStyleSheet(
            "font-weight: bold; color: #e0e0e0; font-size: 12px; margin-bottom: 5px;"
        )
        results_title.setFixedHeight(20)  # Fixed height for alignment

        # Create scrollable area for results
        results_scroll = QScrollArea()
        results_scroll.setWidgetResizable(True)
        results_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        results_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        results_scroll.setMaximumHeight(130)
        results_scroll.setStyleSheet("QScrollArea { border: none; }")

        self.results_text = QLabel()
        self.results_text.setStyleSheet(
            "background-color: #3b3b3b; padding: 10px; border-radius: 5px; "
            "font-size: 11px; line-height: 1.2;"
        )
        self.results_text.setWordWrap(True)
        self.results_text.setAlignment(Qt.AlignmentFlag.AlignTop)

        results_scroll.setWidget(self.results_text)

        results_layout.addWidget(results_title)
        results_layout.addWidget(results_scroll)

        # Add all sections to header
        header_layout.addWidget(signal_type_widget)
        header_layout.addWidget(settings_widget, 1)  # Give equal space
        header_layout.addWidget(results_widget, 1)  # Give equal space

        # Set minimum height for header but allow expansion
        header_widget.setMinimumHeight(150)
        # Remove maximum height constraint to allow splitter control

        # Add matplotlib figure with navigation toolbar
        self.figure = Figure(figsize=(12, 8), facecolor="#2b2b2b")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        # Create plot area widget containing toolbar and canvas
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(0)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        # Create vertical splitter between header and plots
        vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        vertical_splitter.addWidget(header_widget)
        vertical_splitter.addWidget(plot_widget)

        # Set initial sizes - smaller header, larger plot area
        vertical_splitter.setSizes([180, 600])

        # Allow header to be collapsed but not completely hidden
        vertical_splitter.setCollapsible(
            0, False
        )  # Header cannot be completely collapsed
        vertical_splitter.setCollapsible(
            1, False
        )  # Plot area cannot be completely collapsed

        # Add splitter to main layout
        layout.addWidget(vertical_splitter)

    def _sync_xlim(self, ax):
        """Synchronize x-axis limits across all plots and update y-limits individually."""
        if self._syncing:
            return

        try:
            self._syncing = True
            xlim = ax.get_xlim()

            # Update x-limits for all axes
            for other_ax in self.axes:
                if other_ax != ax:
                    other_ax.set_xlim(xlim)

            # Update y-limits individually for each lead based on visible x-range
            self._update_individual_ylimits(xlim)

            self.canvas.draw_idle()
        finally:
            self._syncing = False

    def _update_individual_ylimits(self, xlim):
        """Update y-limits for each lead individually based on visible x-range."""
        x_min, x_max = xlim

        # Fixed lead order for mapping axes to leads
        lead_order = [
            ("I", "V1"),
            ("II", "V2"),
            ("III", "V3"),
            ("aVR", "V4"),
            ("aVL", "V5"),
            ("aVF", "V6"),
        ]

        ax_index = 0
        for row_leads in lead_order:
            for lead in row_leads:
                if ax_index < len(self.axes):
                    ax = self.axes[ax_index]
                    times, signal = self.signals_data.get(
                        lead, (np.array([]), np.array([]))
                    )

                    if signal.size > 0 and times.size > 0:
                        # Find indices within visible x-range
                        visible_mask = (times >= x_min) & (times <= x_max)
                        if np.any(visible_mask):
                            visible_signal = signal[visible_mask]
                            y_min = np.min(visible_signal)
                            y_max = np.max(visible_signal)

                            # Add margins to y-axis limits
                            y_range = y_max - y_min
                            if y_range > 0:
                                y_min -= 0.1 * y_range
                                y_max += 0.1 * y_range
                            else:
                                # Handle case where signal is flat
                                y_min -= 0.1
                                y_max += 0.1

                            ax.set_ylim(y_min, y_max)

                ax_index += 1

    def _sync_ylim(self):
        """Removed - no longer synchronizing y-axis limits."""
        pass

    def update_settings_display(self):
        """Update the settings display label with current configuration"""
        if self.config:
            settings_text = ""
            settings_text += (
                f"<b>Cleaning:</b> {self.config.get('clean_method', 'none')}<br>"
            )
            settings_text += f"<b>Peak Detection:</b> {self.config.get('peak_method', 'neurokit')}<br>"

            # Display filter settings
            filters = self.config.get("filters", [])
            if filters:
                settings_text += (
                    f"<b>Notch Filters:</b> {', '.join(map(str, filters))} Hz<br>"
                )
            else:
                settings_text += "<b>Notch Filters:</b> None<br>"

            settings_text += (
                f"<b>High-pass:</b> {self.config.get('hp_filter_type', '0.15 Hz')}<br>"
            )

            # Additional processing options
            processing_options = []
            if self.config.get("spike_removal", True):
                processing_options.append("Spike Removal")
            if self.config.get("baseline_correction", False):
                processing_options.append("Baseline Correction")
            if self.config.get("signal_smoothing", False):
                window = self.config.get("smoothing_window", 5)
                processing_options.append(f"Smoothing (w={window})")

            if processing_options:
                settings_text += f"<b>Processing:</b> {', '.join(processing_options)}"
            else:
                settings_text += "<b>Processing:</b> None"

            self.settings_label.setText(settings_text)
        else:
            self.settings_label.setText("<i>No analysis configuration</i>")

    def get_configuration_name(self):
        """Generate a descriptive name for the current configuration"""
        if not self.config:
            return "Default"

        clean = self.config.get("clean_method", "none")
        peak = self.config.get("peak_method", "neurokit")

        # Add filter information
        filters = self.config.get("filters", [])
        filter_part = f"_{'-'.join(map(str, filters))}Hz" if filters else "_NoNotch"

        hp_type = (
            self.config.get("hp_filter_type", "0.15 Hz")
            .replace(" Hz", "")
            .replace(".", "")
        )

        # Add processing options
        options = []
        if self.config.get("spike_removal", True):
            options.append("SR")
        if self.config.get("baseline_correction", False):
            options.append("BC")
        if self.config.get("signal_smoothing", False):
            options.append("SM")
        if self.config.get("smoothing_window", 5) is not None:
            options.append(str(self.config["smoothing_window"]))

        option_part = f"_{''.join(options)}" if options else ""

        return f"{clean}-{peak}{filter_part}_HP{hp_type}{option_part}"

    def update_plot(self):
        """Update the plot based on the selected signal type"""
        if hasattr(self, "ecg_glove") and self.ecg_glove:
            self.plot_ecg_data()

    def plot_ecg_data(self):
        if not self.ecg_glove:
            return

        signal_type = self.signal_type_combo.currentText().lower()
        self.figure.clear()

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
        quality_scores = getattr(self, "quality_scores", {}).get("lead_quality", {})

        # Create figure with 6 rows and 2 columns
        self.axes = []
        self.signals_data = {}

        # Pre-calculate signal data
        max_time = 0
        for row_leads in lead_order:
            for lead in row_leads:
                # Select signal based on type
                if signal_type == "raw":
                    signal = self.ecg_glove.raw_signals.get(lead, np.array([]))
                elif signal_type == "filtered":
                    signal = self.ecg_glove.lead_signals.get(lead, np.array([]))
                else:  # cleaned
                    signal = self.ecg_glove.cleaned_signals.get(lead, np.array([]))

                if signal.size > 0:
                    # Downsample for very large signals (> 10000 points)
                    if signal.size > 10000:
                        downsample_factor = signal.size // 10000 + 1
                        signal = signal[::downsample_factor]

                    times = np.arange(signal.size) / self.ecg_glove.sampling_rate
                    max_time = max(max_time, times[-1] if times.size > 0 else 0)
                    self.signals_data[lead] = (times, signal)
                else:
                    self.signals_data[lead] = (np.array([]), np.array([]))

        # Set up the figure for maximum space utilization
        self.figure.subplots_adjust(
            left=0.02, right=0.98, bottom=0.02, top=0.98, hspace=0.1, wspace=0.1
        )

        # Create subplots with minimal styling - only share x-axis, not y-axis
        first_ax = None
        for row, (left_lead, right_lead) in enumerate(lead_order):
            # Left plot
            if first_ax is None:
                ax_left = self.figure.add_subplot(6, 2, 2 * row + 1)
                first_ax = ax_left
            else:
                ax_left = self.figure.add_subplot(6, 2, 2 * row + 1, sharex=first_ax)

            # Right plot - only share x-axis with first_ax
            ax_right = self.figure.add_subplot(6, 2, 2 * row + 2, sharex=first_ax)

            self.axes.extend([ax_left, ax_right])

            # Configure axes for maximum signal visibility
            for ax, lead in [(ax_left, left_lead), (ax_right, right_lead)]:
                times, signal = self.signals_data[lead]
                if signal.size > 0:
                    ax.plot(
                        times,
                        signal,
                        color=DEFAULT_SIGNAL_COLOR,
                        linewidth=0.8,
                        antialiased=True,
                    )

                    # Set individual y-limits for this lead
                    y_min = np.min(signal)
                    y_max = np.max(signal)
                    y_range = y_max - y_min
                    if y_range > 0:
                        y_min -= 0.1 * y_range
                        y_max += 0.1 * y_range
                    else:
                        y_min -= 0.1
                        y_max += 0.1
                    ax.set_ylim(y_min, y_max)

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

        # Set x-limits for all plots and connect zoom/pan events
        if first_ax:
            first_ax.set_xlim(0, max(max_time, 1))

            # Connect the xlim_changed event to sync function
            first_ax.callbacks.connect("xlim_changed", self._sync_xlim)

        # No need for tight_layout since we're using subplots_adjust
        self.canvas.draw_idle()


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
                margin-top: 15px;
                font-weight: bold;
            }
            QGroupBox::title {
                color: #e0e0e0;
                background-color: transparent;
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 5px 10px;
                margin-left: 8px;
                font-weight: bold;
            }
            QGroupBox::indicator {
                width: 13px;
                height: 13px;
                margin-left: 5px;
            }
            QGroupBox::indicator:unchecked {
                border: 1px solid #505050;
                background-color: #2b2b2b;
            }
            QGroupBox::indicator:checked {
                border: 1px solid #4a9eff;
                background-color: #4a9eff;
            }
            QComboBox {
                background-color: #3b3b3b;
                color: #e0e0e0;
                border: 1px solid #505050;
                border-radius: 3px;
                padding: 5px;
                margin-top: 5px;
                font-size: 11px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid #e0e0e0;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #3b3b3b;
                color: #e0e0e0;
                border: 1px solid #505050;
                selection-background-color: #4a9eff;
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
            QScrollArea QWidget {
                background-color: #2b2b2b;
            }
            QScrollArea QScrollBar:vertical {
                background-color: #404040;
                width: 12px;
                border-radius: 6px;
            }
            QScrollArea QScrollBar::handle:vertical {
                background-color: #606060;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollArea QScrollBar::handle:vertical:hover {
                background-color: #707070;
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
            QSplitter::handle {
                background-color: #505050;
            }
            QSplitter::handle:horizontal {
                width: 3px;
                background-color: #505050;
            }
            QSplitter::handle:vertical {
                height: 3px;
                background-color: #505050;
            }
            QSplitter::handle:hover {
                background-color: #606060;
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

        # Create and setup sidebar with scrolling
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        sidebar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        sidebar_scroll.setMaximumWidth(320)

        sidebar_content = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_content)
        sidebar_layout.setContentsMargins(5, 5, 5, 5)
        sidebar_layout.setSpacing(10)

        # File selection group - using CollapsibleBox
        file_group = CollapsibleBox("File Selection")
        file_group.setChecked(True)  # Keep expanded by default

        self.select_folder_btn = QPushButton("Select Folder")
        self.select_folder_btn.clicked.connect(self.select_folder)

        self.file_list = QListWidget()
        self.file_list.itemSelectionChanged.connect(self.file_selected)
        self.file_list.setMaximumHeight(150)  # Limit height to save space

        file_group.addWidget(self.select_folder_btn)
        file_group.addWidget(self.file_list)

        # Analysis Methods group - using CollapsibleBox
        methods_group = CollapsibleBox("Analysis Methods")
        methods_group.setChecked(True)  # Keep expanded by default

        # ECG Cleaning Method
        clean_widget = QWidget()
        clean_layout = QVBoxLayout(clean_widget)
        clean_layout.setContentsMargins(0, 0, 0, 0)
        clean_layout.addWidget(QLabel("ECG Cleaning Method:"))
        self.clean_method = QComboBox()
        self.clean_method.addItems(
            [
                "none",
                "neurokit",
                "biosppy",
                "vg",
                "engzeemod2012",
                "elgendi2010",
                "hamilton2002",
                "pantompkins1985",
            ]
        )
        self.clean_method.setCurrentText("none")
        clean_layout.addWidget(self.clean_method)

        # R-Peak Detection
        peak_widget = QWidget()
        peak_layout = QVBoxLayout(peak_widget)
        peak_layout.setContentsMargins(0, 0, 0, 0)
        peak_layout.addWidget(QLabel("R-Peak Detection:"))
        self.peak_method = QComboBox()
        self.peak_method.addItems(
            [
                "neurokit",
                "promac",
                "emrich2023",
                "rodrigues2021",
                "nabian2018",
                "kalidas2017",
                "manikandan2012",
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
        self.peak_method.setCurrentText("neurokit")
        peak_layout.addWidget(self.peak_method)

        methods_group.addWidget(clean_widget)
        methods_group.addWidget(peak_widget)

        # Filter Configuration group - using CollapsibleBox
        filter_group = CollapsibleBox("Filter Configuration")
        filter_group.setChecked(False)  # Collapsed by default to save space

        # Notch Filter Selection
        notch_widget = QWidget()
        notch_layout = QVBoxLayout(notch_widget)
        notch_layout.setContentsMargins(0, 0, 0, 0)
        notch_layout.addWidget(QLabel("Notch Filters (Power Line):"))
        self.filter_50hz = QCheckBox("50 Hz (Europe)")
        self.filter_60hz = QCheckBox("60 Hz (North America)")
        self.filter_60hz.setChecked(True)  # Default for US
        self.filter_100hz = QCheckBox("100 Hz (2nd Harmonic)")
        self.filter_120hz = QCheckBox("120 Hz (2nd Harmonic)")
        notch_layout.addWidget(self.filter_50hz)
        notch_layout.addWidget(self.filter_60hz)
        notch_layout.addWidget(self.filter_100hz)
        notch_layout.addWidget(self.filter_120hz)

        # High-pass Filter Selection
        hp_widget = QWidget()
        hp_layout = QVBoxLayout(hp_widget)
        hp_layout.setContentsMargins(0, 0, 0, 0)
        hp_layout.addWidget(QLabel("High-pass Filter:"))
        self.hp_filter_type = QComboBox()
        self.hp_filter_type.addItems(["0.05 Hz", "0.15 Hz", "0.5 Hz"])
        self.hp_filter_type.setCurrentText("0.15 Hz")  # Default
        hp_layout.addWidget(self.hp_filter_type)

        filter_group.addWidget(notch_widget)
        filter_group.addWidget(hp_widget)

        # Additional Processing group - using CollapsibleBox
        processing_group = CollapsibleBox("Additional Processing")
        processing_group.setChecked(False)  # Collapsed by default to save space

        self.spike_removal = QCheckBox("Spike Removal (Morphology Filter)")
        self.spike_removal.setChecked(True)  # Default enabled
        self.baseline_correction = QCheckBox("Baseline Drift Correction")
        self.signal_smoothing = QCheckBox("Signal Smoothing")

        # Smoothing window size
        smoothing_widget = QWidget()
        smoothing_layout = QHBoxLayout(smoothing_widget)
        smoothing_layout.setContentsMargins(0, 0, 0, 0)
        smoothing_layout.addWidget(QLabel("Window:"))
        self.smoothing_window = QComboBox()
        self.smoothing_window.addItems(["3", "5", "7", "9"])
        self.smoothing_window.setCurrentText("5")
        self.smoothing_window.setEnabled(False)  # Disabled by default
        smoothing_layout.addWidget(self.smoothing_window)

        # Connect smoothing checkbox to enable/disable window selection
        self.signal_smoothing.toggled.connect(self.smoothing_window.setEnabled)

        processing_group.addWidget(self.spike_removal)
        processing_group.addWidget(self.baseline_correction)
        processing_group.addWidget(self.signal_smoothing)
        processing_group.addWidget(smoothing_widget)

        # Analysis button
        self.process_btn = QPushButton("Process")
        self.process_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #4a9eff;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
                margin: 10px 0;
            }
            QPushButton:hover {
                background-color: #5ba8ff;
            }
            QPushButton:disabled {
                background-color: #666666;
                color: #999999;
            }
        """
        )
        self.process_btn.clicked.connect(self.process_data)
        self.process_btn.setEnabled(False)

        # Add all elements to sidebar layout
        sidebar_layout.addWidget(file_group)
        sidebar_layout.addWidget(methods_group)
        sidebar_layout.addWidget(filter_group)
        sidebar_layout.addWidget(processing_group)
        sidebar_layout.addWidget(self.process_btn)
        sidebar_layout.addStretch()

        # Add version label
        version_label = QLabel(f"ECG Analyzer {APP_VERSION}")
        version_label.setStyleSheet("color: #808080; font-size: 10px; padding: 5px;")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(version_label)

        # Set the content widget to the scroll area
        sidebar_scroll.setWidget(sidebar_content)

        # Create main display area with tabs
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)

        # Set dark theme for plots
        plt.style.use("dark_background")

        # Add widgets to splitter
        splitter.addWidget(sidebar_scroll)
        splitter.addWidget(self.tab_widget)

        # Set initial splitter sizes (smaller sidebar, more space for plots)
        splitter.setSizes([320, 1000])

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.file_list.clear()
            # list and sort by filename, display basename only
            ret_files = sorted([f for f in os.listdir(folder) if f.endswith(".ret")])
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
            # Get selected methods from dropdowns
            clean_method = self.clean_method.currentText()
            peak_method = self.peak_method.currentText()

            # Get enhanced filter selections
            filters = []
            if self.filter_50hz.isChecked():
                filters.append(50)
            if self.filter_60hz.isChecked():
                filters.append(60)
            if self.filter_100hz.isChecked():
                filters.append(100)
            if self.filter_120hz.isChecked():
                filters.append(120)

            # Get high-pass filter type
            hp_text = self.hp_filter_type.currentText()
            from ecg_filters import HPFilterType

            hp_type_mapping = {
                "0.05 Hz": HPFilterType.HP005,
                "0.15 Hz": HPFilterType.HP015,
                "0.5 Hz": HPFilterType.HP05,
            }
            hp_filter_type = hp_type_mapping.get(hp_text, HPFilterType.HP015)

            # Get additional processing options
            spike_removal = self.spike_removal.isChecked()
            baseline_correction = self.baseline_correction.isChecked()
            signal_smoothing = self.signal_smoothing.isChecked()
            smoothing_window = int(self.smoothing_window.currentText())

            # Get current configuration including filter settings
            config = {
                "clean_method": clean_method,
                "peak_method": peak_method,
                "filters": filters,
                "hp_filter_type": hp_text,
                "spike_removal": spike_removal,
                "baseline_correction": baseline_correction,
                "signal_smoothing": signal_smoothing,
                "smoothing_window": smoothing_window if signal_smoothing else None,
            }

            # Create configuration key that includes filter settings
            filter_key = "_".join(map(str, sorted(filters))) if filters else "none"
            config_key = (
                f"{self.current_file}_{config['clean_method']}_{config['peak_method']}_"
                f"notch{filter_key}_hp{hp_text.replace(' ', '').replace('.', '')}_"
                f"spike{spike_removal}_base{baseline_correction}_smooth{signal_smoothing}"
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

            # Get selected methods from dropdowns
            clean_method = self.clean_method.currentText()
            peak_method = self.peak_method.currentText()

            # Get enhanced filter selections
            filters = []
            if self.filter_50hz.isChecked():
                filters.append(50)
            if self.filter_60hz.isChecked():
                filters.append(60)
            if self.filter_100hz.isChecked():
                filters.append(100)
            if self.filter_120hz.isChecked():
                filters.append(120)

            # Get high-pass filter type
            hp_text = self.hp_filter_type.currentText()
            from ecg_filters import HPFilterType

            hp_type_mapping = {
                "0.05 Hz": HPFilterType.HP005,
                "0.15 Hz": HPFilterType.HP015,
                "0.5 Hz": HPFilterType.HP05,
            }
            hp_filter_type = hp_type_mapping.get(hp_text, HPFilterType.HP015)

            # Get additional processing options
            spike_removal = self.spike_removal.isChecked()
            baseline_correction = self.baseline_correction.isChecked()
            signal_smoothing = self.signal_smoothing.isChecked()
            smoothing_window = int(self.smoothing_window.currentText())

            tab.ecg_glove = EcgGlove(
                sampling_rate=500,
                clean_method=clean_method,
                peak_method=peak_method,
                filters=filters,
                spike_removal=spike_removal,
                hp_filter_type=hp_filter_type,
                powerline_freq=60,  # Default, will be overridden by filters list
                enable_baseline_correction=baseline_correction,
                enable_smoothing=signal_smoothing,
                smoothing_window=smoothing_window,
            )
            tab.ecg_glove.decode_data(data_bytes)

            # Analyze quality first
            quality_results = tab.ecg_glove.compute_quality()

            # Store quality scores and measurement results
            tab.quality_scores = quality_results

            # Analyze ECG if quality is acceptable
            results = tab.ecg_glove.process()

            # Format results for display with HTML formatting in two columns
            result_text = """
            <table style='width: 100%; border-collapse: collapse;'>
                <tr>
                    <td style='width: 50%; vertical-align: top; padding-right: 10px;'>
                        <b>Analysis Lead:</b> {analysis_lead}<br>
                        <br>
                        <b>Measurements:</b><br>
                        {measurements}
                    </td>
                    <td style='width: 50%; vertical-align: top; padding-left: 10px;'>
                        <b>Signal Quality:</b> {quality}<br>
                        <br>
                        <b>Electrical Axes:</b><br>
                        {axes}
                    </td>
                </tr>
            </table>
            """

            # Prepare data for formatting
            analysis_lead = results["AnalysisLead"]

            # Prepare measurements
            measurements_list = []
            if "ecgData" in results and "measurements" in results["ecgData"]:
                measurements = results["ecgData"]["measurements"]

                if measurements.get("HeartRate_BPM"):
                    measurements_list.append(
                        f"<b>HR:</b> {measurements['HeartRate_BPM']:.1f} BPM"
                    )
                if measurements.get("RR_Interval_ms"):
                    measurements_list.append(
                        f"<b>RR:</b> {measurements['RR_Interval_ms']:.0f} ms"
                    )
                if measurements.get("P_Duration_ms"):
                    measurements_list.append(
                        f"<b>P:</b> {measurements['P_Duration_ms']:.0f} ms"
                    )
                if measurements.get("PR_Interval_ms"):
                    measurements_list.append(
                        f"<b>PR:</b> {measurements['PR_Interval_ms']:.0f} ms"
                    )
                if measurements.get("QRS_Duration_ms"):
                    measurements_list.append(
                        f"<b>QRS:</b> {measurements['QRS_Duration_ms']:.0f} ms"
                    )
                if measurements.get("QT_Interval_ms"):
                    measurements_list.append(
                        f"<b>QT:</b> {measurements['QT_Interval_ms']:.0f} ms"
                    )
                if measurements.get("QTc_Interval_ms"):
                    measurements_list.append(
                        f"<b>QTc:</b> {measurements['QTc_Interval_ms']:.0f} ms"
                    )

            measurements_text = (
                "<br>".join(measurements_list)
                if measurements_list
                else "No measurements available"
            )

            # Prepare axes
            axes_list = []
            if "ecgData" in results and "measurements" in results["ecgData"]:
                measurements = results["ecgData"]["measurements"]

                if measurements.get("P_Axis") is not None:
                    axes_list.append(f"<b>P:</b> {measurements['P_Axis']:.0f}°")
                if measurements.get("QRS_Axis") is not None:
                    axes_list.append(f"<b>QRS:</b> {measurements['QRS_Axis']:.0f}°")
                if measurements.get("T_Axis") is not None:
                    axes_list.append(f"<b>T:</b> {measurements['T_Axis']:.0f}°")

            axes_text = (
                "<br>".join(axes_list) if axes_list else "No axis data available"
            )

            # Prepare quality
            quality_text = ""
            if "overall_quality" in quality_results:
                overall_quality = quality_results["overall_quality"]
                if isinstance(overall_quality, (int, float)):
                    quality_color = (
                        "#6bff6b"
                        if overall_quality > 0.7
                        else "#ffd93d" if overall_quality > 0.4 else "#ff6b6b"
                    )
                    quality_text = f"<span style='color: {quality_color}'>{overall_quality:.2f}</span>"
                else:
                    quality_text = str(overall_quality)
            else:
                quality_text = "Not available"

            # Format the final result text
            result_text = result_text.format(
                analysis_lead=analysis_lead,
                measurements=measurements_text,
                quality=quality_text,
                axes=axes_text,
            )

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
        tab.signals_data = {}

        # Pre-calculate signal data
        max_time = 0
        for row_leads in lead_order:
            for lead in row_leads:
                signal = tab.ecg_glove.cleaned_signals.get(lead, np.array([]))
                if signal.size > 0:
                    # Downsample for very large signals (> 10000 points)
                    if signal.size > 10000:
                        downsample_factor = signal.size // 10000 + 1
                        signal = signal[::downsample_factor]

                    times = np.arange(signal.size) / tab.ecg_glove.sampling_rate
                    max_time = max(max_time, times[-1] if times.size > 0 else 0)
                    tab.signals_data[lead] = (times, signal)
                else:
                    tab.signals_data[lead] = (np.array([]), np.array([]))

        # Set up the figure for maximum space utilization
        tab.figure.subplots_adjust(
            left=0.02, right=0.98, bottom=0.02, top=0.98, hspace=0.1, wspace=0.1
        )

        # Create subplots with minimal styling - only share x-axis, not y-axis
        first_ax = None
        for row, (left_lead, right_lead) in enumerate(lead_order):
            # Left plot
            if first_ax is None:
                ax_left = tab.figure.add_subplot(6, 2, 2 * row + 1)
                first_ax = ax_left
            else:
                ax_left = tab.figure.add_subplot(6, 2, 2 * row + 1, sharex=first_ax)

            # Right plot - only share x-axis with first_ax
            ax_right = tab.figure.add_subplot(6, 2, 2 * row + 2, sharex=first_ax)

            tab.axes.extend([ax_left, ax_right])

            # Configure axes for maximum signal visibility
            for ax, lead in [(ax_left, left_lead), (ax_right, right_lead)]:
                times, signal = tab.signals_data[lead]
                if signal.size > 0:
                    ax.plot(
                        times,
                        signal,
                        color=DEFAULT_SIGNAL_COLOR,
                        linewidth=0.8,
                        antialiased=True,
                    )

                    # Set individual y-limits for this lead
                    y_min = np.min(signal)
                    y_max = np.max(signal)
                    y_range = y_max - y_min
                    if y_range > 0:
                        y_min -= 0.1 * y_range
                        y_max += 0.1 * y_range
                    else:
                        y_min -= 0.1
                        y_max += 0.1
                    ax.set_ylim(y_min, y_max)

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

        # Set x-limits for all plots and connect zoom/pan events
        if first_ax:
            first_ax.set_xlim(0, max(max_time, 1))

            # Connect the xlim_changed event to sync function
            first_ax.callbacks.connect("xlim_changed", tab._sync_xlim)

        # No need for tight_layout since we're using subplots_adjust
        tab.canvas.draw_idle()


def main():
    app = QApplication(sys.argv)
    window = EcgAnalyzerGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
