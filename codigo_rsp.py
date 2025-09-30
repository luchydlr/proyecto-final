import cv2
import numpy as np
from picamera2 import Picamera2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import csv
from datetime import datetime
import math

# =========================
# 🔹 Inicializar detector
# =========================
def initialize_detector():
    base_options = mp_python.BaseOptions(
        model_asset_path="facemesh/face_landmarker.task"
    )
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)

# =========================
# 🔹 Funciones auxiliares
# =========================
def euclidean(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

def eye_aspect_ratio(landmarks, right=True):
    if right:
        top, bottom = landmarks[145], landmarks[159]
    else:
        top, bottom = landmarks[374], landmarks[386]
    return euclidean(top, bottom)

def mouth_aspect_ratio(landmarks):
    vertical_pairs = [(13, 14), (82, 87), (312, 317)]
    horizontals = (78, 308)
    v_dist = np.mean([euclidean(landmarks[a], landmarks[b]) for a, b in vertical_pairs])
    h_dist = euclidean(landmarks[horizontals[0]], landmarks[horizontals[1]])
    return v_dist / h_dist if h_dist > 0 else 0

def draw_landmarks_on_frame(frame, marks):
    for landmark in marks[0]:
        x = int(landmark.x * frame.shape[1])
        y = int(landmark.y * frame.shape[0])
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

def display_state_text(frame, text, position=(30, 30)):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 255), 2, cv2.LINE_AA)

# =========================
# 🔹 Proceso principal
# =========================
def process_video(detector, frames_interval=10, perclos_threshold=60, csv_filename='perclos_log.csv'):
    print("Inicializando cámara OV5647 con Picamera2...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
    picam2.configure(config)
    picam2.start()

    frames_cnt, closed_frames = 0, 0
    yawn_count, blink_count = 0, 0
    eyes_closed_prev = False
    yawn_frames = 0

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'PERCLOS (%)', 'State', 'Yawns', 'Blinks'])

        print("🟢 Cámara lista. Presiona 'q' para salir.")

        while True:
            frame = picam2.capture_array()
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Mediapipe espera RGB

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            result = detector.detect(mp_image)

            state_text = "AWAKE"

            if result.face_landmarks:
                marks = result.face_landmarks
                draw_landmarks_on_frame(frame, marks)

                # === OJOS ===
                ear_r = eye_aspect_ratio(marks[0], right=True)
                ear_l = eye_aspect_ratio(marks[0], right=False)
                eyes_closed = (ear_r < 0.009 and ear_l < 0.009)

                if eyes_closed:
                    closed_frames += 1

                if not eyes_closed_prev and eyes_closed:
                    eyes_closed_prev = True
                elif eyes_closed_prev and not eyes_closed:
                    blink_count += 1
                    eyes_closed_prev = False

                # === BOCA ===
                mar = mouth_aspect_ratio(marks[0])
                if mar > 0.65:
                    yawn_frames += 1
                else:
                    if yawn_frames > 5:  # mantenida abierta
                        yawn_count += 1
                    yawn_frames = 0

            # === PERCLOS cada N frames ===
            if frames_cnt >= frames_interval:
                perclos = 100 * closed_frames / frames_interval
                state_text = "SLEEPY" if perclos > perclos_threshold else "AWAKE"
                timestamp = datetime.now().strftime("%H:%M:%S")
                writer.writerow([timestamp, round(perclos, 2), state_text, yawn_count, blink_count])
                print(f"[{timestamp}] PERCLOS: {perclos:.2f}% | {state_text} | Yawns: {yawn_count} | Blinks: {blink_count}")
                closed_frames = 0
                frames_cnt = 0

            display_state_text(frame, state_text)
            cv2.imshow("Fatigue Monitor - OV5647", frame)

            frames_cnt += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    picam2.stop()
    cv2.destroyAllWindows()

# =========================
# 🔹 MAIN
# =========================
if __name__ == "__main__":
    detector = initialize_detector()
    process_video(detector)
