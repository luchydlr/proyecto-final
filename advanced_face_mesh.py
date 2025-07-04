import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configurar Face Mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2,  # Detectar hasta 2 caras
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Inicializar cámara
cap = cv2.VideoCapture(0)

# Función para obtener coordenadas de landmarks específicos
def get_landmark_coordinates(landmarks, width, height):
    """Convierte landmarks normalizados a coordenadas de píxeles"""
    coords = []
    for landmark in landmarks.landmark:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        coords.append([x, y])
    return coords

print("Controles:")
print("- Presiona 'q' para salir")
print("- Presiona 'c' para alternar entre diferentes estilos de dibujo")

draw_style = 0  # Variable para alternar estilos

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Preparar imagen
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    
    # Procesar con MediaPipe
    results = face_mesh.process(image_rgb)
    
    # Convertir de vuelta a BGR
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # Dibujar resultados
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            if draw_style == 0:
                # Estilo 1: Contornos básicos
                mp_drawing.draw_landmarks(
                    image=image_bgr,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                    
            elif draw_style == 1:
                # Estilo 2: Mesh completo
                mp_drawing.draw_landmarks(
                    image=image_bgr,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=1))
                        
            elif draw_style == 2:
                # Estilo 3: Solo puntos importantes
                coords = get_landmark_coordinates(face_landmarks, width, height)
                
                # Dibujar algunos puntos importantes
                important_points = [1, 9, 10, 151, 175, 263, 33, 133, 362, 263]
                for point_idx in important_points:
                    if point_idx < len(coords):
                        cv2.circle(image_bgr, tuple(coords[point_idx]), 3, (0, 255, 255), -1)
            
            # Mostrar información
            coords = get_landmark_coordinates(face_landmarks, width, height)
            cv2.putText(image_bgr, f'Landmarks detectados: {len(coords)}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Mostrar estilo actual
    styles = ['Contornos', 'Mesh completo', 'Puntos clave']
    cv2.putText(image_bgr, f'Estilo: {styles[draw_style]} (presiona c para cambiar)', 
               (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Mostrar imagen
    cv2.imshow('Face Mesh Avanzado', image_bgr)
    
    # Controles de teclado
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        draw_style = (draw_style + 1) % 3

# Limpiar
cap.release()
cv2.destroyAllWindows()