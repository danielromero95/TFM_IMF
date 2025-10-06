# src/D_modeling/math_utils.py

import numpy as np

def calculate_angle_3d(p1, p2, p3) -> float:
    """
    Calcula el ángulo ∡p1–p2–p3 en espacio 3D.
    Asume que p1, p2, p3 son objetos con atributos .x, .y, .z.
    """
    # Vectores desde el punto central (p2) a los otros dos puntos
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])

    # Fórmula del producto escalar: a · b = |a| |b| cos(theta)
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    # Añadimos epsilon para evitar división por cero si una norma es 0
    norm_product += 1e-8 

    # Calculamos el coseno y lo acotamos para evitar errores de punto flotante
    cos_angle = np.clip(dot_product / norm_product, -1.0, 1.0)
    
    angle_rad = np.arccos(cos_angle)
    
    return np.degrees(angle_rad)