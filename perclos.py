import mediapipe as mp
import cv2
import numpy as np
from mediapipe.tasks import python as mp_python
import os

def initialize_detector():
    base_options = mp_python.BaseOptions(model_asset_path='face_landmarker.task')
    options = mp_python.vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )
    return mp_python.vision.FaceLandmarker.create_from_options(options)

def detect_landmarks(detector, image):
    detection_result = detector.detect(image)
    return detection_result, detection_result.face_blendshapes, detection_result.face_landmarks

def are_eyes_closed(marks):
    diff_r = marks[0][145].y - marks[0][159].y
    diff_l = marks[0][374].y - marks[0][386].y
    return diff_r < 0.009 and diff_l < 0.009

def convert_bgr_to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def create_image_object(image):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

def display_state_text(frame, text, position=(50, 50)):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)

def process_webcam(detector, frames_interval=30, perclos_percentage=60):
    capture = cv2.VideoCapture(0)
    fps = 30
    frames_cnt, closed_frames = 0, 0
    state_text = ''

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        #frame = cv2.resize(frame, (640, 480))
        image = convert_bgr_to_rgb(frame)
        image = np.ascontiguousarray(image)
        image_0 = create_image_object(image)
        results, blends, marks = detect_landmarks(detector, image_0)

        if marks and are_eyes_closed(marks):
            closed_frames += 1

        if frames_cnt > frames_interval:
            frames_cnt = 0
            perclos = 100 * closed_frames / frames_interval
            closed_frames = 0
            state_text = 'SLEEPY' if perclos > perclos_percentage else 'AWAKE'

        display_state_text(frame, state_text)
        frames_cnt += 1
        cv2.imshow('Webcam Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

# Ejecutar
detector = initialize_detector()
process_webcam(detector)
