# src/gui/widgets/results_panel.py

import logging
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QFrame
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from .video_player import VideoPlayerWidget
from .plot_widget import PlotWidget
from src.gui.gui_utils import get_first_available_series

logger = logging.getLogger(__name__)

class ResultsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        self.video_player = VideoPlayerWidget(self)
        main_layout.addWidget(self.video_player, 2)

        right_column_widget = QWidget()
        right_column_layout = QVBoxLayout(right_column_widget)
        
        top_layout = QHBoxLayout()
        self.rep_counter = QLabel("0")
        font = self.rep_counter.font(); font.setPointSize(48); font.setBold(True)
        self.rep_counter.setFont(font)
        self.rep_counter.setAlignment(Qt.AlignCenter)
        rep_box = self._create_box("Repeticiones", self.rep_counter)
        top_layout.addWidget(rep_box)
        
        self.status_label = QLabel("Listo")
        font = self.status_label.font(); font.setPointSize(12)
        self.status_label.setFont(font)
        self.status_label.setAlignment(Qt.AlignCenter)
        status_box = self._create_box("Estado", self.status_label)
        top_layout.addWidget(status_box)
        right_column_layout.addLayout(top_layout)

        self.plot_widget = PlotWidget(self)
        right_column_layout.addWidget(self.plot_widget)
        
        faults_label = QLabel("Fallos Detectados:")
        font = faults_label.font(); font.setBold(True)
        faults_label.setFont(font)
        right_column_layout.addWidget(faults_label)
        self.fault_list = QListWidget(self)
        right_column_layout.addWidget(self.fault_list)

        main_layout.addWidget(right_column_widget, 1)

    def _create_box(self, title, widget):
        box = QFrame(); box.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(box)
        title_label = QLabel(title)
        font = title_label.font(); font.setBold(True)
        title_label.setFont(font); title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        layout.addWidget(widget)
        return box

    def update_results(self, results):
        self.rep_counter.setText(str(results.get("repeticiones_contadas", "0")))
        
        video_path = results.get("debug_video_path")
        if video_path:
            self.video_player.load_video(video_path)
        
        df = results.get("dataframe_metricas")
        if df is None or df.empty:
            self.status_label.setText("Estado: No se generaron métricas.")
            self.plot_widget.clear_plots()
            self.fault_list.clear(); self.fault_list.addItem("Análisis no completado.")
            return

        self.status_label.setText("Estado: Análisis completado.")
        self.plot_widget.plot_data(df)

        self.fault_list.clear()
        faults = results.get("fallos_detectados", [])
        if not faults:
            self.fault_list.addItem("¡No se detectaron fallos!")
        else:
            for fault in faults:
                self.fault_list.addItem(f"Rep {fault['rep']}: {fault['type']} ({fault['value']})")

    def clear_results(self):
        self.rep_counter.setText("0")
        self.status_label.setText("Listo para analizar")
        self.plot_widget.clear_plots()
        self.fault_list.clear()
        self.video_player.clear_media()