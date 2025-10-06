import cv2
import mediapipe as mp
import numpy as np

# --- Constantes de Visualización ---
# Ahora fijamos una altura máxima para la ventana, mucho mejor para vídeos verticales
DISPLAY_HEIGHT = 800

# --- Funciones de Cálculo ---
def calculate_angle_3d(p1, p2, p3):
    """Calcula el ángulo entre 3 puntos en el espacio 3D."""
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 180.0

    angle_rad = np.arccos(dot_product / (magnitude_v1 * magnitude_v2))
    return np.degrees(angle_rad)

# --- Configuración de MediaPipe ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    smooth_landmarks=True
)
mp_drawing = mp.solutions.drawing_utils

# --- Procesamiento del Vídeo ---
video_path = 'test_squat.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: No se pudo abrir el vídeo en la ruta: {video_path}")
    exit()

frame_count = 0
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    frame_count += 1
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    display_image = image.copy()

    if results.pose_world_landmarks:
        landmarks_3d = results.pose_world_landmarks.landmark
        left_hip = landmarks_3d[mp_pose.PoseLandmark.LEFT_HIP.value]
        left_knee = landmarks_3d[mp_pose.PoseLandmark.LEFT_KNEE.value]
        left_ankle = landmarks_3d[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        left_shoulder = landmarks_3d[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        
        knee_angle = calculate_angle_3d(left_hip, left_knee, left_ankle)
        torso_angle = calculate_angle_3d(left_shoulder, left_hip, left_knee)
        
        # Escribimos los ángulos en la imagen
        cv2.putText(display_image, f"Rodilla: {knee_angle:.1f}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(display_image, f"Torso: {torso_angle:.1f}", (20, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            display_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # <<< LÓGICA DE REDIMENSIONADO CORREGIDA >>>
    h, w, _ = display_image.shape
    # Calculamos la proporción ancho/alto
    aspect_ratio = w / h
    # Calculamos el nuevo ancho basado en la altura fija
    display_width = int(DISPLAY_HEIGHT * aspect_ratio)
    
    # Redimensionamos la imagen final
    display_image_resized = cv2.resize(display_image, (display_width, DISPLAY_HEIGHT))

    cv2.imshow('BlazePose 3D - Banco de Pruebas', display_image_resized)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
pose.close()