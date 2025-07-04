import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
import os

def initialize_detector():
    base_options = mp_python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
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

def display_state_text(frame, text, position=(30, 30)):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

def draw_landmarks_on_frame(frame, marks):
    for landmark in marks[0]:
        x = int(landmark.x * frame.shape[1])
        y = int(landmark.y * frame.shape[0])
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

def process_video(detector, frames_interval=30, perclos_threshold=60):
    cap = cv2.VideoCapture(0)
    frames_cnt, closed_frames = 0, 0
    state_text = ''

    while cap.isOpened():
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

        if frames_cnt >= frames_interval:
            perclos = 100 * closed_frames / frames_interval
            state_text = "SLEEPY" if perclos > perclos_threshold else "AWAKE"
            closed_frames = 0
            frames_cnt = 0

        display_state_text(frame, state_text)
        frames_cnt += 1

        cv2.imshow("FaceMesh + PERCLOS", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# MAIN
if __name__ == "__main__":
    detector = initialize_detector()
    process_video(detector)