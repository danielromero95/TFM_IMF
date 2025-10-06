# src/gui/widgets/plot_widget.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import pandas as pd
import logging

from src.gui.gui_utils import get_first_available_series

logger = logging.getLogger(__name__)

class PlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot_item = pg.PlotWidget()
        self.plot_item.setBackground('w')
        self.plot_item.showGrid(x=True, y=True)
        self.plot_item.setLabel('left', 'Ángulo', units='°')
        self.plot_item.setLabel('bottom', 'Frame')
        
        layout = QVBoxLayout()
        layout.addWidget(self.plot_item)
        self.setLayout(layout)

    def plot_data(self, df_metrics: pd.DataFrame):
        """
        Dibuja las métricas disponibles y las líneas de umbral en el DataFrame.
        """
        self.clear_plots()

        from src import config # Importamos config para acceder a los umbrales

        x_series = get_first_available_series(df_metrics, 'frame_index')
        y_series = get_first_available_series(df_metrics, 'knee_angle')

        if x_series is not None and y_series is not None:
            valid_data = pd.concat([x_series, y_series], axis=1).dropna()
            
            # Dibuja la curva de ángulo principal
            pen = pg.mkPen(color=(0, 120, 215), width=2)
            self.plot_item.plot(valid_data[x_series.name], valid_data[y_series.name], pen=pen)
            self.plot_item.setTitle("Ángulo de la Rodilla", color="k", size="12pt")
            
            # --- CAMBIO CLAVE: Dibujar las líneas de umbral ---
            high_thresh = config.SQUAT_HIGH_THRESH
            low_thresh = config.SQUAT_LOW_THRESH

            # Línea para el umbral superior (verde)
            pen_high = pg.mkPen(color=(0, 180, 0), style=pg.QtCore.Qt.DashLine)
            high_line = pg.InfiniteLine(pos=high_thresh, angle=0, pen=pen_high, label=f'Up Thresh: {high_thresh}°')
            self.plot_item.addItem(high_line)

            # Línea para el umbral inferior (rojo)
            pen_low = pg.mkPen(color=(215, 60, 60), style=pg.QtCore.Qt.DashLine)
            low_line = pg.InfiniteLine(pos=low_thresh, angle=0, pen=pen_low, label=f'Down Thresh: {low_thresh}°')
            self.plot_item.addItem(low_line)

            logger.info(f"Gráfico actualizado con la columna '{y_series.name}' y umbrales.")
        else:
            self.plot_item.setTitle("Datos de ángulo no disponibles", color="r", size="12pt")
            logger.warning("No se encontraron columnas de ángulo o de frame para dibujar.")

    def clear_plots(self):
        self.plot_item.clear()