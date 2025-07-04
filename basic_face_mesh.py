import cv2
import mediapipe as mp
import numpy as np
import sys
import os

class FaceMeshDetector:
    def __init__(self, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Inicializar el detector de Face Mesh
        
        Args:
            max_num_faces: Número máximo de caras a detectar
            min_detection_confidence: Confianza mínima para detección
            min_tracking_confidence: Confianza mínima para seguimiento
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def detect_face_mesh(self, image):
        """
        Detectar face mesh en una imagen
        
        Args:
            image: Imagen en formato BGR (OpenCV)
            
        Returns:
            results: Resultados de MediaPipe
            image_rgb: Imagen en formato RGB
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        return results, image_rgb
    
    def draw_landmarks(self, image, results, style='contours'):
        """
        Dibujar landmarks en la imagen
        
        Args:
            image: Imagen donde dibujar
            results: Resultados de MediaPipe
            style: Estilo de dibujo ('contours', 'tesselation', 'points')
        """
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if style == 'contours':
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                
                elif style == 'tesselation':
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=(0, 255, 0), thickness=1))
                
                elif style == 'points':
                    # Dibujar solo puntos importantes
                    height, width = image.shape[:2]
                    important_points = [1, 9, 10, 151, 175, 263, 33, 133, 362, 263]
                    
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        if idx in important_points:
                            x = int(landmark.x * width)
                            y = int(landmark.y * height)
                            cv2.circle(image, (x, y), 3, (0, 255, 255), -1)
        
        return image

def main():
    """Función principal"""
    print("=== Face Mesh Detector ===")
    print("Autor: LucasRomero26")
    print("Fecha: 2025-06-25")
    print("\nControles:")
    print("- Presiona 'q' para salir")
    print("- Presiona 'c' para cambiar estilo")
    print("- Presiona 's' para guardar captura")
    print("\nIniciando...")
    
    # Verificar que la cámara esté disponible
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara")
        return
    
    # Inicializar detector
    detector = FaceMeshDetector(max_num_faces=2)
    
    # Estilos disponibles
    styles = ['contours', 'tesselation', 'points']
    current_style = 0
    
    # Contador para capturas
    capture_count = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            print("No se pudo leer el frame de la cámara")
            break
        
        # Voltear horizontalmente para efecto espejo
        frame = cv2.flip(frame, 1)
        
        # Detectar face mesh
        results, _ = detector.detect_face_mesh(frame)
        
        # Dibujar landmarks
        frame = detector.draw_landmarks(frame, results, styles[current_style])
        
        # Agregar información en pantalla
        height, width = frame.shape[:2]
        cv2.putText(frame, f'Estilo: {styles[current_style]}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if results.multi_face_landmarks:
            cv2.putText(frame, f'Caras detectadas: {len(results.multi_face_landmarks)}', 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, 'q: salir | c: cambiar estilo | s: captura', 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Mostrar frame
        cv2.imshow('Face Mesh - LucasRomero26', frame)
        
        # Manejar teclas
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            current_style = (current_style + 1) % len(styles)
            print(f"Cambiado a estilo: {styles[current_style]}")
        elif key == ord('s'):
            # Guardar captura
            filename = f'face_mesh_capture_{capture_count:03d}.jpg'
            cv2.imwrite(filename, frame)
            capture_count += 1
            print(f"Captura guardada: {filename}")
    
    # Limpiar recursos
    cap.release()
    cv2.destroyAllWindows()
    print("¡Programa terminado!")

if __name__ == "__main__":
    main()
    