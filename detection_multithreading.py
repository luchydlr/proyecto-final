import cv2
import numpy as np
import math
import csv
import time
import queue
import threading
import json
from datetime import datetime

from picamera2 import Picamera2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import paho.mqtt.client as mqtt
import ssl

# ================== CONFIGURACIÓN GENERAL ==================
# Cámara
CAM_SIZE = (640, 480)
FPS_TARGET = 30

# PERCLOS (por bloques de N frames, como tu lógica original)
FRAMES_INTERVAL = 10           # cada cuántos frames calculas PERCLOS
PERCLOS_THRESHOLD = 60.0       # % para marcar "SLEEPY"

# Umbrales de evento (mantengo tu criterio)
EYE_CLOSED_THRESHOLD = 0.009   # EAR proxy (145-159) y (374-386)
YAWN_MAR_THRESHOLD = 0.65
YAWN_HOLD_FRAMES = 5           # mar sostenido para contar bostezo

# CSV
CSV_FILENAME = "perclos_log.csv"

# MQTT (mismo hilo de inferencia)
MQTT_ENABLE = True
MQTT_BROKER = "a1omfl67425kjv-ats.iot.us-east-2.amazonaws.com"  # Cambia por tu broker
MQTT_PORT = 1883                                   # 1883 sin TLS, 8883 con TLS
MQTT_TOPIC_METRICS = "fatiga/metricas"
MQTT_TOPIC_ALERTS  = "fatiga/alerts"

# TLS (opcional para AWS IoT Core)
MQTT_USE_TLS = False           # Pon True para TLS (puerto 8883)
CA_CERTS = "/home/lucianadelarosa/Desktop/proyecto-final/AmazonRootCA1.pem"
CLIENT_CERT = "/home/lucianadelarosa/Desktop/proyecto-final/certificate.pem.crt"
CLIENT_KEY  = "/home/lucianadelarosa/Desktop/proyecto-final/private.pem.key"

# Anti-spam de alertas
ALERT_COOLDOWN_S = 5.0

# Colas/hilos
frame_queue = queue.Queue(maxsize=1)   # captura -> inferencia (drop-old)
annot_queue = queue.Queue(maxsize=1)   # inferencia -> UI (drop-old)
stop_event = threading.Event()

# ================== MEDIA PIPE (CREACIÓN DE DETECTOR) ==================
def initialize_detector():
    base_options = mp_python.BaseOptions(
        model_asset_path="face_landmarker_v2_with_blendshapes.task"
    )
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)

# ================== AUXILIARES DE MÉTRICAS ==================
def euclidean(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

def eye_aspect_ratio(landmarks, right=True):
    # Tus índices originales como proxy de EAR
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
    return (v_dist / h_dist) if h_dist > 0 else 0.0

def draw_landmarks_on_frame_rgb(rgb_frame, marks):
    h, w = rgb_frame.shape[:2]
    for landmark in marks[0]:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(rgb_frame, (x, y), 1, (0, 255, 0), -1)

def put_text_rgb(rgb_frame, text, pos=(12, 28), color=(0, 255, 255)):
    cv2.putText(rgb_frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

# ================== HILO DE CAPTURA ==================
class CaptureWorker(threading.Thread):
    def __init__(self, stop_evt):
        super().__init__(daemon=True)
        self.stop_evt = stop_evt
        self.picam2 = None

    def run(self):
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"format": "RGB888", "size": CAM_SIZE}
            )
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(0.2)
            print("🟢 Cámara lista (Picamera2).")
        except Exception as e:
            print(f"❌ Error iniciando cámara: {e}")
            self.stop_evt.set()
            return

        while not self.stop_evt.is_set():
            try:
                rgb_frame = self.picam2.capture_array()  # RGB
            except Exception:
                time.sleep(0.005)
                continue

            # drop-old
            if frame_queue.full():
                try: frame_queue.get_nowait()
                except queue.Empty: pass
            frame_queue.put(rgb_frame)

        try:
            self.picam2.stop()
        except Exception:
            pass

# ================== HILO DE INFERENCIA (CSV + MQTT AQUÍ) ==================
class InferenceWorker(threading.Thread):
    def __init__(self, stop_evt):
        super().__init__(daemon=True)
        self.stop_evt = stop_evt
        self.detector = initialize_detector()

        # Estado/métricas
        self.frames_cnt = 0
        self.closed_frames = 0
        self.yawn_count = 0
        self.blink_count = 0
        self.eyes_closed_prev = False
        self.yawn_frames = 0

        # CSV
        self.csv_file = open(CSV_FILENAME, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Timestamp', 'PERCLOS (%)', 'State', 'Yawns', 'Blinks', 'EAR_L', 'EAR_R', 'MAR'])

        # MQTT
        self.mqtt_client = None
        self.last_alert_ts = {"sleepy": 0.0, "microsleep": 0.0, "yawn": 0.0}
        if MQTT_ENABLE:
            self._init_mqtt()

    # ---------- MQTT helpers ----------
    def _init_mqtt(self):
        try:
            self.mqtt_client = mqtt.Client()
            if MQTT_USE_TLS:
                self.mqtt_client.tls_set(
                    ca_certs=CA_CERTS,
                    certfile=CLIENT_CERT,
                    keyfile=CLIENT_KEY,
                    cert_reqs=ssl.CERT_REQUIRED,
                    tls_version=ssl.PROTOCOL_TLSv1_2
                )
                self.mqtt_client.tls_insecure_set(False)
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            # Nota: como estamos en un hilo dedicado, podemos usar loop() manual en cada publish
            print("🔗 MQTT conectado.")
        except Exception as e:
            print(f"⚠️ No se pudo conectar a MQTT: {e}")
            self.mqtt_client = None

    def _mqtt_publish(self, topic, payload_dict):
        if not (MQTT_ENABLE and self.mqtt_client):
            return
        try:
            self.mqtt_client.publish(topic, json.dumps(payload_dict), qos=0, retain=False)
            self.mqtt_client.loop(timeout=0.05)
        except Exception as e:
            # Intento de reconexión simple
            try:
                self.mqtt_client.reconnect()
            except Exception:
                pass

    def _maybe_alert(self, kind, payload):
        # kind: 'sleepy' | 'microsleep' | 'yawn'
        now = time.time()
        last = self.last_alert_ts.get(kind, 0.0)
        if (now - last) >= ALERT_COOLDOWN_S:
            self._mqtt_publish(MQTT_TOPIC_ALERTS, payload)
            self.last_alert_ts[kind] = now

    # ---------- Métricas ----------
    def _process_metrics(self, marks):
        landmarks = marks[0]

        # EAR proxy (tu definición)
        ear_r = eye_aspect_ratio(landmarks, right=True)
        ear_l = eye_aspect_ratio(landmarks, right=False)
        eyes_closed = (ear_r < EYE_CLOSED_THRESHOLD and ear_l < EYE_CLOSED_THRESHOLD)

        # Blink por flanco
        if eyes_closed:
            self.closed_frames += 1
        if not self.eyes_closed_prev and eyes_closed:
            self.eyes_closed_prev = True
        elif self.eyes_closed_prev and not eyes_closed:
            self.blink_count += 1
            self.eyes_closed_prev = False

        # MAR (bostezo)
        mar = mouth_aspect_ratio(landmarks)
        yawn_event = False
        if mar > YAWN_MAR_THRESHOLD:
            self.yawn_frames += 1
        else:
            if self.yawn_frames > YAWN_HOLD_FRAMES:
                self.yawn_count += 1
                yawn_event = True
            self.yawn_frames = 0

        return eyes_closed, mar, ear_l, ear_r, yawn_event

    def _perclos_step_and_log(self, state_text, ear_l, ear_r, mar):
        """
        Cada FRAMES_INTERVAL calcula PERCLOS en % (frames cerrados / N) y:
        - Escribe fila en CSV
        - Publica métricas por MQTT (topic métricas)
        - Emite alerta 'sleepy' si PERCLOS > umbral
        """
        if self.frames_cnt >= FRAMES_INTERVAL:
            perclos = 100.0 * self.closed_frames / max(FRAMES_INTERVAL, 1)
            sleepy = perclos > PERCLOS_THRESHOLD
            state_at_window = "SLEEPY" if sleepy else state_text
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # CSV
            self.csv_writer.writerow([ts, round(perclos, 2), state_at_window,
                                      self.yawn_count, self.blink_count,
                                      round(ear_l, 5), round(ear_r, 5), round(mar, 3)])

            # MQTT (métricas periódicas)
            metrics_payload = {
                "ts": ts,
                "perclos_pct": round(perclos, 2),
                "state": state_at_window,
                "yawns": int(self.yawn_count),
                "blinks": int(self.blink_count),
                "ear_left": float(ear_l),
                "ear_right": float(ear_r),
                "mar": float(mar),
                "window_frames": int(FRAMES_INTERVAL)
            }
            self._mqtt_publish(MQTT_TOPIC_METRICS, metrics_payload)

            # Alerta sleepy por PERCLOS
            if sleepy:
                self._maybe_alert("sleepy", {
                    "ts": ts,
                    "estado": "sleepy",
                    "perclos_pct": round(perclos, 2)
                })

            # Reset de ventana
            self.closed_frames = 0
            self.frames_cnt = 0

            # Opcional: imprime para consola
            print(f"[{ts}] PERCLOS: {perclos:.2f}% | {state_at_window} | Yawns: {self.yawn_count} | Blinks: {self.blink_count}")

        return state_text

    def run(self):
        try:
            while not self.stop_evt.is_set():
                try:
                    rgb_frame = frame_queue.get(timeout=0.2)  # RGB
                except queue.Empty:
                    continue

                # Detección
                state_text = "AWAKE"
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = self.detector.detect(mp_image)

                ear_l = ear_r = mar = 0.0
                if result.face_landmarks:
                    draw_landmarks_on_frame_rgb(rgb_frame, result.face_landmarks)
                    eyes_closed, mar, ear_l, ear_r, yawn_event = self._process_metrics(result.face_landmarks)

                    if eyes_closed:
                        state_text = "EYES CLOSED"
                    if mar > YAWN_MAR_THRESHOLD:
                        state_text = "YAWNING"

                    # Alerta por bostezo puntual
                    if yawn_event:
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self._maybe_alert("yawn", {"ts": ts, "estado": "yawn", "mar": float(mar)})

                # Microsueño simple: ojos cerrados muchos frames seguidos (derivaría de closed_frames alto dentro de la ventana);
                # si quieres una condición distinta, la podemos añadir con otro contador.

                # Paso de PERCLOS por bloques + logging + MQTT métricas
                self.frames_cnt += 1
                state_text = self._perclos_step_and_log(state_text, ear_l, ear_r, mar)

                # UI
                put_text_rgb(rgb_frame, state_text, (12, 28))
                bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

                if annot_queue.full():
                    try: annot_queue.get_nowait()
                    except queue.Empty: pass
                annot_queue.put(bgr)

                frame_queue.task_done()
        finally:
            try:
                self.csv_file.flush()
                self.csv_file.close()
            except Exception:
                pass
            if MQTT_ENABLE and self.mqtt_client:
                try:
                    self.mqtt_client.disconnect()
                except Exception:
                    pass

# ================== MAIN / UI ==================
def main():
    print("Iniciando hilos (captura / inferencia)…")
    cap_t = CaptureWorker(stop_event)
    inf_t = InferenceWorker(stop_event)

    cap_t.start()
    inf_t.start()

    print("🟢 Presiona 'q' para salir.")
    try:
        while not stop_event.is_set():
            try:
                frame = annot_queue.get(timeout=0.3)
                cv2.imshow("Fatigue Monitor - OV5647 (MT CSV+MQTT)", frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    stop_event.set()
            except queue.Empty:
                pass
    finally:
        stop_event.set()
        cap_t.join(timeout=2.0)
        inf_t.join(timeout=2.0)
        cv2.destroyAllWindows()
        print("Finalizado.")

if __name__ == "__main__":
    main()
