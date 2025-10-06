# src/gui/gui_utils.py

import pandas as pd
from typing import Optional, List

# Mapa que asocia un concepto lógico (ej. 'ángulo de rodilla') con los
# posibles nombres de columna que puede tener en el DataFrame.
# La función buscará en orden de prioridad (de izquierda a derecha).
METRICS_MAP = {
    'knee_angle':  ['rodilla_izq', 'knee_angle_3d'],
    'hip_angle':   ['cadera_izq', 'hip_angle_3d'],
    'frame_index': ['frame_idx', 'frame'],
    'knee_symmetry': ['sim_rodilla']
}

def get_first_available_series(df: pd.DataFrame, logical_name: str) -> Optional[pd.Series]:
    """
    Busca en el METRICS_MAP el nombre lógico y devuelve la primera columna
    candidata que exista en el DataFrame.

    Args:
        df: El DataFrame de métricas.
        logical_name: El nombre lógico de la métrica (ej. 'knee_angle').

    Returns:
        La serie de pandas si se encuentra, o None si no se encuentra ninguna candidata.
    """
    candidates = METRICS_MAP.get(logical_name, [])
    for column_name in candidates:
        if column_name in df.columns:
            return df[column_name]
    return None