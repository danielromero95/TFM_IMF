# src/pipeline.py

import logging
import os

import pandas as pd
import cv2

from src import config
from src.A_preprocessing.frame_extraction import extract_and_preprocess_frames
from src.F_visualization.video_renderer import render_landmarks_on_video_hq

from src.B_pose_estimation.estimators import (
    BaseEstimator,
    CroppedPoseEstimator,
    BlazePose3DEstimator,
    EstimationResult
)
from src.B_pose_estimation.processing import (
    filter_and_interpolate_landmarks,
    calculate_metrics_from_sequence
)
from src.D_modeling.count_reps import count_repetitions_from_df

# Funciones de análisis 3D (implementar en src/D_modeling/analysis_3d.py)
try:
    from src.D_modeling.analysis_3d import calculate_3d_metrics, count_reps_3d
except ImportError:
    calculate_3d_metrics = None
    count_reps_3d = None

logger = logging.getLogger(__name__)


def build_estimator() -> BaseEstimator:
    """
    Fábrica de estimadores: devuelve BlazePose3DEstimator si USE_3D_ANALYSIS=True,
    o CroppedPoseEstimator en caso contrario.
    """
    if config.USE_3D_ANALYSIS:
        return BlazePose3DEstimator()
    else:
        return CroppedPoseEstimator()


def run_full_pipeline_in_memory(video_path: str, settings: dict, progress_callback=None):
    """
    Ejecuta el pipeline completo de análisis en memoria,
    eligiendo estimación 2D o 3D según config.USE_3D_ANALYSIS.
    """
    def notify(progress: int, message: str):
        logger.info(message)
        if progress_callback:
            progress_callback(progress)

    # Preparación de paths
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = settings.get('output_dir', '.')
    session_dir = os.path.join(output_dir, base_name)
    os.makedirs(session_dir, exist_ok=True)

    estimator = build_estimator()
    try:
        mode = '3D' if config.USE_3D_ANALYSIS else '2D'
        notify(0, f"Inicializando pipeline en modo {mode}...")

        # FASE 1: extracción y rotación de fotogramas
        notify(5, "FASE 1: Extrayendo y rotando fotogramas...")
        original_frames, fps = extract_and_preprocess_frames(
            video_path=video_path,
            rotate=settings.get('rotate'),
            sample_rate=settings.get('sample_rate', 1),
        )
        if not original_frames:
            raise ValueError("No se pudieron extraer fotogramas del vídeo.")

        # FASE 2: estimación de pose
        notify(15, "FASE 2: Estimando pose en los fotogramas...")
        estimation_results: list[EstimationResult] = []
        total_frames = len(original_frames)
        for idx, frame in enumerate(original_frames):
            # Reportar cada ~10% de los fotogramas
            if total_frames > 0 and idx % max(1, total_frames // 10) == 0:
                prog = 15 + int(60 * idx / total_frames)
                notify(prog, f"Procesando frame {idx+1}/{total_frames}...")
            result = estimator.estimate(frame)
            estimation_results.append(result)

        if config.USE_3D_ANALYSIS:
            # --- LÓGICA PARA EL MODO 3D ---
            notify(75, "FASE 3/4/5 (3D): Analizando métricas 3D y contando repeticiones...")

            # --- CAMBIO CLAVE: Pasamos los umbrales desde settings/config ---
            df_metrics = calculate_3d_metrics(estimation_results, fps)
            n_reps, faults_detected = count_reps_3d(
                df_metrics,
                up_thresh=settings.get('high_thresh', config.SQUAT_HIGH_THRESH),
                down_thresh=settings.get('low_thresh', config.SQUAT_LOW_THRESH),
                depth_fail_thresh=settings.get('depth_fail_thresh', 90.0) # Umbral de fallo de profundidad
            )

        else:
            # Lógica 2D actual
            notify(75, "FASE 3 (2D): Filtrando e interpolando landmarks...")
            # Reconstruir raw DataFrame 2D
            rows = []
            for frame_idx, res in enumerate(estimation_results):
                if res.landmarks:
                    row = {'frame': frame_idx}
                    for i, lm in enumerate(res.landmarks):
                        row.update({
                            f"x{i}": lm['x'],
                            f"y{i}": lm['y'],
                            f"z{i}": lm['z'],
                            f"vis{i}": lm['visibility'],
                        })
                    rows.append(row)
            df_raw_landmarks = pd.DataFrame(rows)
            filtered_sequence, crop_boxes = filter_and_interpolate_landmarks(df_raw_landmarks)

            notify(85, "FASE 4 (2D): Calculando métricas biomecánicas...")
            df_metrics = calculate_metrics_from_sequence(filtered_sequence, fps)

            notify(95, "FASE 5 (2D): Contando repeticiones...")
            n_reps = count_repetitions_from_df(
                df_metrics,
                low_thresh=settings.get('low_thresh', config.SQUAT_LOW_THRESH)
            )
            faults_detected = []

        # FASE EXTRA: vídeo de depuración
        debug_video_path = None
        if settings.get('generate_debug_video', False):
            notify(100, "FASE EXTRA: Renderizando vídeo de depuración...")
            annotated_frames = [res.annotated_image for res in estimation_results]
            height, width = annotated_frames[0].shape[:2]
            output_path = os.path.join(session_dir, f"{base_name}_debug.mp4")
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height)
            )
            for fr in annotated_frames:
                writer.write(fr)
            writer.release()
            debug_video_path = output_path

        # Guardado de métricas si está en modo depuración
        if settings.get('debug_mode', False):
            metric_file = os.path.join(session_dir, f"{base_name}_metrics.csv")
            df_metrics.to_csv(metric_file, index=False)
            logger.info(f"Métricas guardadas en: {metric_file}")

        notify(100, "PIPELINE COMPLETADO")
        return {
            "repeticiones_contadas": n_reps,
            "dataframe_metricas": df_metrics,
            "debug_video_path": debug_video_path,
            "fallos_detectados": faults_detected,
        }

    finally:
        estimator.close()
        logger.info("Estimator cerrado correctamente.")
