# src/B_pose_estimation/estimators.py

import cv2
import numpy as np
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict

# Principio 3: Importaciones centralizadas y limpias
try:
    import mediapipe as mp
    from mediapipe.python.solutions.pose import Pose
    from mediapipe.python.solutions.drawing_utils import draw_landmarks
except ImportError:
    logging.error("MediaPipe no está instalado. Por favor, ejecuta 'pip install mediapipe'.")
    raise

from src import config

logger = logging.getLogger(__name__)


# Principio 2: Objeto de resultado único y explícito
@dataclass
class EstimationResult:
    """Contenedor de datos para el resultado de una estimación de pose."""
    landmarks: Optional[List[Dict[str, float]]] = None
    world_landmarks: Optional[List[Dict[str, float]]] = None
    annotated_image: Optional[np.ndarray] = None
    crop_box: Optional[List[int]] = None
    raw_mediapipe_results: Optional[object] = None # Para depuración avanzada


# Principio 1: Interfaz común para todos los estimadores
class BaseEstimator(ABC):
    """Clase base abstracta para todos los estimadores de pose."""
    @abstractmethod
    def estimate(self, image: np.ndarray) -> EstimationResult:
        """Estima la pose en un único fotograma."""
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """Libera los recursos del modelo."""
        raise NotImplementedError


class PoseEstimator(BaseEstimator):
    """Estimador 2D básico que procesa la imagen completa."""
    def __init__(self):
        self.pose = Pose(
            static_image_mode=True,
            model_complexity=config.MODEL_COMPLEXITY,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE
        )

    def estimate(self, image: np.ndarray) -> EstimationResult:
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.pose_landmarks:
            return EstimationResult(annotated_image=image)

        landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility} for lm in results.pose_landmarks.landmark]
        
        annotated_image = image.copy()
        draw_landmarks(annotated_image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        
        return EstimationResult(
            landmarks=landmarks,
            annotated_image=annotated_image,
            raw_mediapipe_results=results
        )

    def close(self):
        self.pose.close()


class CroppedPoseEstimator(BaseEstimator):
    """Estimador 2D de dos fases: detecta en la imagen completa y analiza en un recorte."""
    def __init__(self, crop_margin=0.15, target_size=(256, 256)):
        self.crop_margin = crop_margin
        self.target_size = target_size
        
        # Un modelo para la detección inicial y otro para el análisis del recorte
        self.pose_full = Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)
        self.pose_crop = Pose(static_image_mode=True, model_complexity=config.MODEL_COMPLEXITY, min_detection_confidence=config.MIN_DETECTION_CONFIDENCE)

    def estimate(self, image: np.ndarray) -> EstimationResult:
        h0, w0 = image.shape[:2]
        results_full = self.pose_full.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results_full.pose_landmarks:
            return EstimationResult(annotated_image=image)

        # Calcula el bounding box del cuerpo
        xy_full = np.array([[lm.x * w0, lm.y * h0] for lm in results_full.pose_landmarks.landmark])
        x_min, y_min = np.min(xy_full, axis=0)
        x_max, y_max = np.max(xy_full, axis=0)
        
        # Añade margen
        dx, dy = (x_max - x_min) * self.crop_margin, (y_max - y_min) * self.crop_margin
        x1, y1 = max(int(x_min - dx), 0), max(int(y_min - dy), 0)
        x2, y2 = min(int(x_max + dx), w0), min(int(y_max + dy), h0)
        
        crop_box = [x1, y1, x2, y2]
        crop = image[y1:y2, x1:x2]
        
        if crop.size == 0:
            return EstimationResult(annotated_image=image, crop_box=crop_box)
            
        # Analiza el recorte
        crop_resized = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_LINEAR)
        results_crop = self.pose_crop.process(cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB))
        
        annotated_crop = crop_resized.copy()
        landmarks_crop = None
        if results_crop.pose_landmarks:
            landmarks_crop = [{'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility} for lm in results_crop.pose_landmarks.landmark]
            draw_landmarks(annotated_crop, results_crop.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        # NOTA: Devolvemos la imagen recortada y anotada, no la original. El pipeline decidirá cómo usarla.
        return EstimationResult(
            landmarks=landmarks_crop,
            annotated_image=annotated_crop,
            crop_box=crop_box,
            raw_mediapipe_results=results_crop
        )

    def close(self):
        self.pose_full.close()
        self.pose_crop.close()


class BlazePose3DEstimator(BaseEstimator):
    """Estimador que utiliza MediaPipe Pose para extraer landmarks 3D del mundo real."""
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )

    def estimate(self, image: np.ndarray) -> EstimationResult:
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            return EstimationResult(annotated_image=image)
            
        # --- CAMBIO CLAVE AQUÍ ---
        # Antes convertíamos los landmarks a diccionarios. Ahora los pasamos como
        # los objetos originales de MediaPipe, que es lo que el resto del código espera.
        landmarks_2d = results.pose_landmarks.landmark
        world_landmarks_3d = results.pose_world_landmarks.landmark
        
        annotated_image = image.copy()
        draw_landmarks(annotated_image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        
        return EstimationResult(
            landmarks=landmarks_2d,
            world_landmarks=world_landmarks_3d,
            annotated_image=annotated_image,
            raw_mediapipe_results=results
        )

    def close(self):
        self.pose.close()