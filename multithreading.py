import cv2
import numpy as np 
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import math
import threading
from picamera2 import picam2
import datetime
import time
import queue
import csv

# Initialize MediaPipe Face Detection
def initialize_detector():
    base_options = mp_python.BaseOptions(
        model_asset_path="face_landmarker_v2_with_blendshapes.task"
    )
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=2
    )
    return vision.FaceLandmarker.create_from_options(options)

#aux functions
def distance(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

#eye aspect ratio
def ear(landmarks, right=True):
    if right:
        top, bottom = landmarks[145], landmarks[159]
    else:
        top, bottom = landmarks[374], landmarks[386]
    return distance(top, bottom)

#mouth aspect ratio
def mar(landmarks):
    vertical_pairs = [(13, 14), (82, 87), (312, 317)]
    horizontals = (78, 308)
    v_dist = np.mean([distance(landmarks[a], landmarks[b]) for a, b in vertical_pairs])
    h_dist = distance(landmarks[horizontals[0]], landmarks[horizontals[1]])
    return v_dist / h_dist if h_dist > 0 else 0

def draw_landmarks_on_frame(frame, marks):
    for landmark in marks[0]:
        x = int(landmark.x * frame.shape[1])
        y = int(landmark.y * frame.shape[0])
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
def display_state_text(frame, text, position=(30, 30)):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 255), 2, cv2.LINE_AA)
    
#thread 1: Capture frames from the camera
def capture_thread(picam2, frame_queue):
    while True:
        frame = picam2.capture_array()
        if frame is not None:
            frame_queue.put(frame)

def processing_thread(detector, frame_queue, result_queue, frames_interval=10, perclos_threshold=60):
    frames_cnt, closed_frames = 0, 0
    yawn_count, blink_count = 0, 0
    eyes_closed_prev = False
    yawn_frames = 0
    
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            result = detector.detect(mp_image)

            state_text = "AWAKE"
            if result.face_landmarks:
                marks = result.face_landmarks
                draw_landmarks_on_frame(frame, marks)

                # === OJOS ===
                ear_r = ear(marks[0], right=True)
                ear_l = ear(marks[0], right=False)
                eyes_closed = (ear_r < 0.009 and ear_l < 0.009)

                if eyes_closed:
                    closed_frames += 1

                if not eyes_closed_prev and eyes_closed:
                    eyes_closed_prev = True
                elif eyes_closed_prev and not eyes_closed:
                    blink_count += 1
                    eyes_closed_prev = False
                
                # === BOCA ===
                mar = mar(marks[0])
                if mar > 0.65:
                    yawn_frames += 1
                else:
                    if yawn_frames > 5:
                        yawn_count += 1
                    yawn_frames = 0

            # === PERCLOS ===
            if frames_cnt >= frames_interval:
                perclos = 100 * closed_frames / frames_interval
                state_text = "SLEEPY" if perclos > perclos_threshold else "AWAKE"
                timestamp = datetime.now().strftime("%H:%M:%S")
                result_queue.put((frame, state_text, perclos, yawn_count, blink_count, timestamp))
                closed_frames = 0
                frames_cnt = 0

            frames_cnt += 1
            result_queue.put((frame, state_text, None, yawn_count, blink_count, None))
            
def logger_thread(result_queue, csv_filename='perclos_log.csv'):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'PERCLOS (%)', 'State', 'Yawns', 'Blinks'])
        while True:
            if not result_queue.empty():
                frame, state_text, perclos, yawns, blinks, timestamp = result_queue.get()
                if timestamp:
                    writer.writerow([timestamp, round(perclos, 2), state_text, yawns, blinks])
                    print(f"[{timestamp}] PERCLOS: {perclos:.2f}% | {state_text} | Yawns: {yawns} | Blinks: {blinks}")

def display_thread(result_queue):
    while True:
        if not result_queue.empty():
            frame, state_text, _, _, _, _ = result_queue.get()
            display_state_text(frame, state_text)
            cv2.imshow("Fatigue Monitor - OV5647", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

#main
if __name__ == "__main__":
    print("Inicializando cámara OV5647 con Picamera2...")
    picam2 = picam2()
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    detector = initialize_detector()
    frame_queue = queue.Queue()
    result_queue = queue.Queue()

    # Crear hilos
    t1 = threading.Thread(target=capture_thread, args=(picam2, frame_queue))
    t2 = threading.Thread(target=processing_thread, args=(detector, frame_queue, result_queue))
    t3 = threading.Thread(target=logger_thread, args=(result_queue,))
    t4 = threading.Thread(target=display_thread, args=(result_queue,))

    # Iniciar hilos
    for t in [t1, t2, t3, t4]:
        t.daemon = True
        t.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Finalizando programa...")
        picam2.stop()
        cv2.destroyAllWindows()