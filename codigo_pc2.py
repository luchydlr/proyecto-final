import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque

def initialize_detector():
    print("Inicializando detector...")
    base_options = mp_python.BaseOptions(model_asset_path='/Users/lucianadelarosa/Desktop/Facemesh_PF/facemesh/face_landmarker.task')
    options = mp_python.vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=2
    )
    return mp_python.vision.FaceLandmarker.create_from_options(options)

def detect_landmarks(detector, image):
    detection_result = detector.detect(image)
    return detection_result, detection_result.face_blendshapes, detection_result.face_landmarks

def are_eyes_closed(marks):
    diff_r = marks[0][145].y - marks[0][159].y
    diff_l = marks[0][374].y - marks[0][386].y
    return diff_r < 0.009 and diff_l < 0.009

def is_mouth_open(marks):
    top_lip = marks[0][0].y
    bottom_lip = marks[0][17].y
    diff = bottom_lip - top_lip
    return diff > 0.10

def convert_bgr_to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def create_image_object(image):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

def display_state_text(frame, text, position=(30, 30)):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

def draw_landmarks_on_frame(frame, marks):
    for landmark in marks[0]:
        x = int(landmark.x * frame.shape[1])
        y = int(landmark.y * frame.shape[0])
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

def process_video(detector, frames_interval=50, perclos_threshold=60, csv_filename='perclos_log.csv'):
    print("Abriendo cámara...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara.")
        return

    frames_cnt, closed_frames = 0, 0
    state_text = ''
    yawn_count = 0
    yawn_timer = 0
    mouth_open_frame_count = 0

    blink_count = 0
    eyes_closed_prev = False  # Para detectar transición abierto → cerrado

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'PERCLOS (%)', 'State', 'Yawns', 'Blinks'])

        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 5))
        max_points = 30
        perclos_vals = deque([0] * max_points, maxlen=max_points)
        yawn_vals = deque([0] * max_points, maxlen=max_points)
        blink_vals = deque([0] * max_points, maxlen=max_points)
        timestamps = deque([''] * max_points, maxlen=max_points)

        line_perclos, = ax.plot(list(perclos_vals), label="PERCLOS (%)", color='blue')
        line_yawn, = ax.plot(list(yawn_vals), label="YAWNS acumulados", color='red')
        line_blinks, = ax.plot(list(blink_vals), label="BLINKS acumulados", color='green')

        ax.set_ylim(0, 100)
        ax.set_title("Monitoreo de Fatiga en Tiempo Real")
        ax.set_ylabel("PERCLOS / YAWNS / BLINKS")
        ax.set_xlabel("Tiempo")
        ax.legend()
        fig.tight_layout()

        print("🟢 Cámara funcionando. Presiona 'q' para salir.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_image = convert_bgr_to_rgb(frame)
            image_np = np.ascontiguousarray(rgb_image)
            image_obj = create_image_object(image_np)

            result, blendshapes, marks = detect_landmarks(detector, image_obj)

            if marks:
                draw_landmarks_on_frame(frame, marks)

                if are_eyes_closed(marks):
                    closed_frames += 1

                # --- Lógica de parpadeos ---
                if not eyes_closed_prev and are_eyes_closed(marks):
                    eyes_closed_prev = True
                elif eyes_closed_prev and not are_eyes_closed(marks):
                    blink_count += 1
                    eyes_closed_prev = False

                # --- Boca / Bostezo ---
                x_top = int(marks[0][0].x * frame.shape[1])
                y_top = int(marks[0][0].y * frame.shape[0])
                x_bot = int(marks[0][17].x * frame.shape[1])
                y_bot = int(marks[0][17].y * frame.shape[0])
                mouth_opening = marks[0][17].y - marks[0][0].y
                is_yawning = mouth_opening > 0.10

                cv2.circle(frame, (x_top, y_top), 4, (0, 255, 0), -1)
                cv2.circle(frame, (x_bot, y_bot), 4, (0, 255, 0), -1)
                line_color = (0, 0, 255) if is_yawning else (0, 255, 0)
                cv2.line(frame, (x_top, y_top), (x_bot, y_bot), line_color, 2)

                if is_yawning:
                    mouth_open_frame_count += 1
                    if mouth_open_frame_count == 10:
                        yawn_count += 1
                        yawn_timer = 30
                else:
                    mouth_open_frame_count = 0

            if frames_cnt >= frames_interval:
                perclos = 100 * closed_frames / frames_interval

                if yawn_timer > 0:
                    state_text = "YAWNING"
                    yawn_timer -= 1
                else:
                    state_text = "SLEEPY" if perclos > perclos_threshold else "AWAKE"

                timestamp = datetime.now().strftime("%H:%M:%S")
                writer.writerow([timestamp, round(perclos, 2), state_text, yawn_count, blink_count])
                print(f"[{timestamp}] PERCLOS: {perclos:.2f}% → {state_text} | Yawns: {yawn_count} | Blinks: {blink_count}")

                perclos_vals.append(perclos)
                yawn_vals.append(yawn_count)
                blink_vals.append(blink_count)
                timestamps.append(timestamp)

                line_perclos.set_ydata(list(perclos_vals))
                line_perclos.set_xdata(range(len(perclos_vals)))
                line_yawn.set_ydata(list(yawn_vals))
                line_yawn.set_xdata(range(len(yawn_vals)))
                line_blinks.set_ydata(list(blink_vals))
                line_blinks.set_xdata(range(len(blink_vals)))

                ax.set_xticks(range(len(timestamps)))
                ax.set_xticklabels(list(timestamps), rotation=45, fontsize=8)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()

                closed_frames = 0
                frames_cnt = 0

            display_state_text(frame, state_text, position=(30, 30))
            display_state_text(frame, f"YAWNS: {yawn_count}", position=(30, 70))
            display_state_text(frame, f"BLINKS: {blink_count}", position=(30, 110))
            frames_cnt += 1

            cv2.imshow("FaceMesh + PERCLOS", frame)
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
