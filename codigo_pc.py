# SOLO PERCLOS (ojos) y MAR (bostezo) + CLASIFICACIÓN por tablas
# Requiere MediaPipe Tasks face_landmarker.task

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision as mp_vision
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque
import time

# =========================
# Parámetros ajustables
# =========================
MODEL_PATH = 'facemesh/face_landmarker.task'
CAM_INDEX = 0

# Ventana para PERCLOS (en frames) y umbral YAWN
FRAMES_INTERVAL = 30                 # ~1 s si ~30 FPS
YAWN_MIN_CONSEC_FRAMES = 10          # frames seguidos boca abierta ≈ 0.33 s a 30 FPS

# Detección básica (normalizado [0..1])
EAR_THRESH = 0.009                    # ojo "cerrado" (aprox)
MAR_OPEN_THRESH = 0.10                # boca "abierta" (aprox)

# Ventana para tasa de bostezos
YAWN_WINDOW_SEC = 60                  # tasa de bostezos por minuto

CSV_FILENAME = 'perclos_log.csv'

# Índices FaceMesh
RIGHT_EYE_UP, RIGHT_EYE_DOWN = 145, 159
LEFT_EYE_UP,  LEFT_EYE_DOWN  = 374, 386
UPPER_LIP_CENTER, LOWER_LIP_CENTER = 13, 14

def initialize_detector():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")
    print("Inicializando detector...")
    base_options = mp_tasks_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        running_mode=mp_vision.RunningMode.VIDEO
    )
    return mp_vision.FaceLandmarker.create_from_options(options)

def are_eyes_closed(marks):
    diff_r = marks[RIGHT_EYE_UP].y - marks[RIGHT_EYE_DOWN].y
    diff_l = marks[LEFT_EYE_UP].y - marks[LEFT_EYE_DOWN].y
    return (diff_r < EAR_THRESH) and (diff_l < EAR_THRESH)

def mar_value(marks):
    return marks[LOWER_LIP_CENTER].y - marks[UPPER_LIP_CENTER].y

def convert_bgr_to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def create_image_object(image_rgb):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

def display_state_text(frame, text, position=(30, 30)):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

def draw_landmarks_on_frame(frame, marks):
    h, w = frame.shape[:2]
    for lm in marks:
        x = int(lm.x * w); y = int(lm.y * h)
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

def classify_state(perclos_ratio, yawns_per_min):
    """
    Reglas basadas en tus tablas:
    - PERCLOS manda. Dentro del rango, uso bostezos/min para afinar.
    """
    if perclos_ratio > 0.8:
        return "MICROSUEÑO"
    elif 0.4 <= perclos_ratio <= 0.8:
        # Somnolencia; si >4 bostezos/min refuerza la etiqueta
        return "SOMNOLENCIA"
    elif 0.3 <= perclos_ratio < 0.4:
        # Fatiga base; si >4 bostezos/min, elevar a somnolencia
        return "SOMNOLENCIA" if yawns_per_min > 4 else "FATIGA"
    else:  # perclos < 0.3
        # Normalidad; si 1–4 bostezos/min, puede ser fatiga leve
        if yawns_per_min > 4:
            return "SOMNOLENCIA"
        elif 1 <= yawns_per_min <= 4:
            return "FATIGA"
        else:
            return "NORMALIDAD"

def process_video(detector):
    print("Abriendo cámara...")
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara.")
        return

    frames_cnt = 0
    closed_frames = 0
    state_text = ''
    yawn_count_total = 0
    mouth_open_frame_count = 0

    # Para tasa de bostezos/min (ventana deslizante)
    yawn_timestamps = deque()  # segundos (time.time())

    start_time = time.time()

    with open(CSV_FILENAME, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'PERCLOS (%)', 'Estado', 'BostezosTot', 'Bostezos/min'])

        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 5))
        max_points = 60
        perclos_vals = deque([0]*max_points, maxlen=max_points)
        yawns_min_vals = deque([0]*max_points, maxlen=max_points)
        timestamps = deque(['']*max_points, maxlen=max_points)

        line_perclos, = ax.plot(list(perclos_vals), label="PERCLOS (%)")
        line_yawn_min, = ax.plot(list(yawns_min_vals), label="Bostezos/min")
        ax.set_ylim(0, 100)
        ax.set_title("Fatiga/Somnolencia/Microsueño (PERCLOS + MAR)")
        ax.set_ylabel("PERCLOS / Bostezos/min")
        ax.set_xlabel("Tiempo")
        ax.legend()
        fig.tight_layout()

        print("🟢 Cámara funcionando. Presiona 'q' para salir.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ No se pudo leer frame de la cámara.")
                break

            ts_ms = int((time.time() - start_time) * 1000)

            rgb_image = convert_bgr_to_rgb(frame)
            image_obj = create_image_object(np.ascontiguousarray(rgb_image))
            detection_result = detector.detect_for_video(image_obj, ts_ms)

            if detection_result and detection_result.face_landmarks:
                marks = detection_result.face_landmarks[0]

                draw_landmarks_on_frame(frame, marks)

                # --- OJOS / PERCLOS ---
                if are_eyes_closed(marks):
                    closed_frames += 1

                # --- BOCA / MAR (bostezo sostenido) ---
                mouth_opening = mar_value(marks)
                is_yawning = mouth_opening > MAR_OPEN_THRESH

                # Dibujo guía labios
                h, w = frame.shape[:2]
                x13, y13 = int(marks[UPPER_LIP_CENTER].x * w), int(marks[UPPER_LIP_CENTER].y * h)
                x14, y14 = int(marks[LOWER_LIP_CENTER].x * w), int(marks[LOWER_LIP_CENTER].y * h)
                color = (0, 0, 255) if is_yawning else (0, 255, 0)
                cv2.circle(frame, (x13, y13), 4, (0, 255, 0), -1)
                cv2.circle(frame, (x14, y14), 4, (0, 255, 0), -1)
                cv2.line(frame, (x13, y13), (x14, y14), color, 2)

                # Conteo de bostezo
                if is_yawning:
                    mouth_open_frame_count += 1
                    if mouth_open_frame_count == YAWN_MIN_CONSEC_FRAMES:
                        yawn_count_total += 1
                        yawn_timestamps.append(time.time())
                else:
                    mouth_open_frame_count = 0

            # ---- Ventana deslizante para bostezos/min (últimos 60 s)
            now = time.time()
            while yawn_timestamps and (now - yawn_timestamps[0] > YAWN_WINDOW_SEC):
                yawn_timestamps.popleft()
            yawns_per_min = len(yawn_timestamps) * (60.0 / YAWN_WINDOW_SEC)

            # ---- Cada FRAMES_INTERVAL, calcular PERCLOS y clasificar
            if frames_cnt >= FRAMES_INTERVAL:
                perclos_ratio = closed_frames / float(FRAMES_INTERVAL)     # 0..1
                perclos_percent = perclos_ratio * 100.0

                estado = classify_state(perclos_ratio, yawns_per_min)
                state_text = f"{estado}"

                timestamp = datetime.now().strftime("%H:%M:%S")
                writer.writerow([timestamp, round(perclos_percent, 2), estado, yawn_count_total, round(yawns_per_min, 2)])
                print(f"[{timestamp}] PERCLOS: {perclos_percent:.2f}%  Bz/min: {yawns_per_min:.2f} → {estado}")

                # Gráfica en vivo
                perclos_vals.append(perclos_percent)
                yawns_min_vals.append(yawns_per_min)
                timestamps.append(timestamp)

                line_perclos.set_ydata(list(perclos_vals))
                line_perclos.set_xdata(range(len(perclos_vals)))
                line_yawn_min.set_ydata(list(yawns_min_vals))
                line_yawn_min.set_xdata(range(len(yawns_min_vals)))

                ax.set_xticks(range(len(timestamps)))
                ax.set_xticklabels(list(timestamps), rotation=45, fontsize=8)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()

                # Reset ventana PERCLOS
                closed_frames = 0
                frames_cnt = 0

            display_state_text(frame, f"Estado: {state_text}", position=(30, 30))
            display_state_text(frame, f"PERCLOS: {min(100, int(perclos_percent)) if frames_cnt else 0}%", position=(30, 70))
            display_state_text(frame, f"Bostezos/min: {yawns_per_min:.1f}", position=(30, 110))
            display_state_text(frame, f"Bostezos tot: {yawn_count_total}", position=(30, 150))

            frames_cnt += 1
            cv2.imshow("FaceMesh + PERCLOS + MAR (Clasificación)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()

# === MAIN ===
if __name__ == "__main__":
    detector = initialize_detector()
    process_video(detector)
