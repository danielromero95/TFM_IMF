# src/D_modeling/analysis_3d.py

import pandas as pd
from typing import List, Tuple
import logging

# Importamos las clases y funciones que hemos creado
from src.B_pose_estimation.estimators import EstimationResult
from src.D_modeling.math_utils import calculate_angle_3d

# Importamos el enumerado de landmarks de MediaPipe para tener una referencia clara
try:
    import mediapipe as mp
    PoseLandmark = mp.solutions.pose.PoseLandmark
except ImportError:
    PoseLandmark = None # Fallback si mediapipe no está

logger = logging.getLogger(__name__)


def calculate_3d_metrics(estimation_results: List[EstimationResult], fps: int) -> pd.DataFrame:
    """
    Calcula las métricas 3D clave a partir de una lista de resultados de estimación.
    """
    if not PoseLandmark:
        logger.error("MediaPipe no está disponible para calcular métricas 3D.")
        return pd.DataFrame()

    metrics_list = []

    for frame_idx, result in enumerate(estimation_results):
        if not result.world_landmarks:
            continue
            
        landmarks = {lm_name.name: result.world_landmarks[lm_name.value] for lm_name in PoseLandmark}

        left_shoulder = landmarks.get('LEFT_SHOULDER')
        left_hip = landmarks.get('LEFT_HIP')
        left_knee = landmarks.get('LEFT_KNEE')
        left_ankle = landmarks.get('LEFT_ANKLE')

        if all([left_shoulder, left_hip, left_knee, left_ankle]):
            knee_angle = calculate_angle_3d(left_hip, left_knee, left_ankle)
            hip_angle = calculate_angle_3d(left_shoulder, left_hip, left_knee)
            hip_height = left_hip.y
        else:
            knee_angle, hip_angle, hip_height = None, None, None

        # --- Llenamos la lista de métricas con los datos del fotograma actual
        metrics_list.append({
            'frame_idx': frame_idx,
            'time_s': frame_idx / fps,
            'rodilla_izq': knee_angle,
            'cadera_izq': hip_angle,
            'altura_cadera': hip_height
        })

    if not metrics_list:
        logger.warning("No se pudieron calcular métricas 3D para ningún fotograma.")
        return pd.DataFrame()

    return pd.DataFrame(metrics_list)


def count_reps_3d(df_metrics: pd.DataFrame, 
                  up_thresh: float, 
                  down_thresh: float,
                  depth_fail_thresh: float = 90.0) -> Tuple[int, List[dict]]:
    """
    Cuenta repeticiones y detecta fallos básicos usando métricas 3D.
    """
    if df_metrics.empty or 'rodilla_izq' not in df_metrics.columns:
        return 0, []

    df_metrics = df_metrics.dropna(subset=['rodilla_izq']).copy()
    if df_metrics.empty:
        return 0, []

    reps = 0
    faults = []
    state = 'up'
    min_angle_in_rep = 180.0

    for row in df_metrics.itertuples():
        angle = row.rodilla_izq
        
        if state == 'up' and angle < down_thresh:
            state = 'down'
            min_angle_in_rep = angle
        
        elif state == 'down':
            if angle < min_angle_in_rep:
                min_angle_in_rep = angle
                
            if angle > up_thresh:
                reps += 1
                state = 'up'
                
                if min_angle_in_rep > depth_fail_thresh:
                    fault_info = {
                        "rep": reps,
                        "type": "Poca Profundidad",
                        "value": f"Ángulo mínimo: {min_angle_in_rep:.1f}° (no bajó de {depth_fail_thresh}°)"
                    }
                    faults.append(fault_info)
                    logger.warning(f"Fallo detectado en repetición {reps}: {fault_info['type']}")

    return reps, faults