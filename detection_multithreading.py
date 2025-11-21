# ================== IMPORTS ==================
import os
import cv2
import math
import csv
import time
import json
import ssl
import queue
import threading
from collections import deque
from datetime import datetime

import numpy as np
from picamera2 import Picamera2

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from paho.mqtt import client as mqtt

# ================== CONFIGURACIÓN ==================
CAM_SIZE = (640, 480)

FRAMES_INTERVAL      = 60          # nº frames por ventana de salida
FATIGA_PERCLOS       = 30.0        # %
SOMNO_MIN_PERCLOS    = 40.0        # %
MICROSLEEP_FRAMES    = 15          # racha de frames cerrados -> microsueño

PERCLOS_WINDOW_S     = 60.0        # segundos de ventana deslizante
MIN_CLOSE_MS         = 200         # mínima duración de cierre (filtro anti micro-parpadeo)

# ---- UMBRALES GEOMÉTRICOS ----
EYE_CLOSED_THRESHOLD = 0.009       # tu métrica vertical top-bottom con FaceLandmarker
YAWN_MAR_THRESHOLD   = 0.65
YAWN_HOLD_FRAMES     = 5

# ---- BANDAS DE EAR (CLASIFICACIÓN DIRECTA) ----
EAR_BANDS = {
    "NORMAL":      (0.25,  1.00),
    "FATIGA":      (0.21,  0.25),
    "SOMNOLENCIA": (0.15,  0.21),
    "MICROSUEÑO":  (0.00,  0.15),
}

# Salidas
CSV_FILENAME         = "session_log.csv"            # por ventana
FRAME_CSV_FILENAME   = "session_log_frames.csv"     # por frame con estado por EAR y perclos
WRITE_FRAME_CSV      = True
HEADLESS             = os.getenv("HEADLESS", "0") == "1"

# ========== AWS IoT Core ==========
ENDPOINT   = "a1omfl67425kjv-ats.iot.us-east-2.amazonaws.com"
PORT       = 8883
CLIENT_ID  = "rsp5"
TOPIC      = "alertas"

CA_PATH   = "aws/AmazonRootCA1.pem"
CERT_PATH = "aws/certificate.pem.crt"
KEY_PATH  = "aws/private.pem.key"

MODEL_PATH = "facemesh/face_landmarker.task"

frame_queue = queue.Queue(maxsize=1)
annot_queue = queue.Queue(maxsize=1)
stop_event  = threading.Event()

# UMBRALES FRECUENCIAS (tabla)
BLINK_MICRO_MAX   = 6.0
BLINK_SOMNO_MAX   = 12.0
BLINK_FATIGA_MAX  = 20.0
BLINK_NORM_MIN    = 17.0
BLINK_NORM_MAX    = 25.0
YAWN_FATIGA_MIN   = 1.0
YAWN_FATIGA_MAX   = 4.0
YAWN_SOMNO_MIN    = 4.0  # >4 -> somnolencia

# ================== AUXILIARES DE MÉTRICAS ==================
def euclidean(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

def eye_aspect_ratio_like(landmarks, right=True):
    # tu métrica: distancia vertical entre dos puntos (no normalizada)
    top, bottom = (landmarks[145], landmarks[159]) if right else (landmarks[374], landmarks[386])
    return euclidean(top, bottom)

def mouth_aspect_ratio(landmarks):
    vertical_pairs = [(13, 14), (82, 87), (312, 317)]
    horizontals = (78, 308)
    v_dist = np.mean([euclidean(landmarks[a], landmarks[b]) for a, b in vertical_pairs])
    h_dist = euclidean(landmarks[horizontals[0]], landmarks[horizontals[1]])
    return (v_dist / h_dist) if h_dist > 0 else 0.0

def put_text_rgb(rgb_frame, text, pos=(12, 28), color=(0, 255, 255)):
    cv2.putText(rgb_frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

def draw_landmarks_on_frame_rgb(rgb_frame, marks):
    h, w = rgb_frame.shape[:2]
    for lm in marks[0]:
        x = int(lm.x * w); y = int(lm.y * h)
        cv2.circle(rgb_frame, (x, y), 1, (0, 255, 0), -1)

def initialize_detector():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo no encontrado: {MODEL_PATH}.")
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)

def estado_por_ear(ear_val):
    """Clasifica estado a partir del valor EAR usando bandas fijas."""
    
    if ear_val is None: 
        return None
    ear_val=5*ear_val  # debug
    print(f"EAR mínimo: {ear_val}")
    for estado, (mn, mx) in EAR_BANDS.items():
        if mn <= ear_val < mx:
            return estado
    return "NORMAL"

# ================== Hilo de Captura ==================
class CaptureWorker(threading.Thread):
    def __init__(self, stop_evt):
        super().__init__(daemon=True)
        self.stop_evt = stop_evt
        self.picam2 = None

    def run(self):
        try:
            self.picam2 = Picamera2()
            cfg = self.picam2.create_preview_configuration(main={"format":"RGB888","size":CAM_SIZE})
            self.picam2.configure(cfg)
            self.picam2.start()
            time.sleep(0.2)
            print("🟢 Cámara lista (Picamera2).")
        except Exception as e:
            import traceback
            print(f"❌ Error iniciando cámara: {e}")
            traceback.print_exc()
            self.stop_evt.set()
            return

        try:
            while not self.stop_evt.is_set():
                try:
                    rgb = self.picam2.capture_array()
                except Exception:
                    time.sleep(0.005)
                    continue

                if frame_queue.full():
                    try: frame_queue.get_nowait()
                    except queue.Empty: pass
                frame_queue.put(rgb)
        except Exception as e:
            import traceback
            print(f"❌ Excepción en CaptureWorker: {e}")
            traceback.print_exc()
            self.stop_evt.set()
        finally:
            try: self.picam2.stop()
            except Exception: pass

# ================== Hilo de Inferencia (CSV + MQTT) ==================
class InferenceWorker(threading.Thread):
    def __init__(self, stop_evt):
        super().__init__(daemon=True)
        self.stop_evt = stop_evt

        self.detector = initialize_detector()

        # Estado por ventana (igual que tu versión)
        self.frames_cnt = 0
        self.closed_frames = 0
        self.closed_streak = 0
        self.microsleep_flag = False

        self.blink_count_win = 0
        self.yawn_count_win  = 0

        self.eyes_closed_prev = False
        self.yawn_frames = 0
        self.window_start_ts = time.time()

        # PERCLOS deslizante
        self.perclos_win = deque()
        self.perclos_win_times = deque()
        self.min_close_fr = None
        self.run_closed = 0
        self.fps_est = 30.0
        self._fps_init_time = time.time()
        self._fps_init_frames = 0

        # CSVs
        self.csv_file = open(CSV_FILENAME, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Timestamp','Estado_final','PERCLOS (%)',
                                  'BlinkRate (/min)','YawnRate (/min)','Blinks','Yawns'])

        self.frame_csv_file = open(FRAME_CSV_FILENAME, mode='w', newline='') if WRITE_FRAME_CSV else None
        if self.frame_csv_file:
            self.frame_csv_writer = csv.writer(self.frame_csv_file)
            self.frame_csv_writer.writerow([
                'FrameID','ts_s','EAR_min(2ojos)','MAR',
                'EyeClosed','BlinkEvt','YawnEvt',
                'Blinks_acc','Yawns_acc',
                'ClosedRun_ms','PerFrame_ms',
                'PERCLOS_win(%)','Estado_EAR','Estado_final'
            ])

        # MQTT
        self.mqtt_client = mqtt.Client(client_id=CLIENT_ID)
        self.mqtt_client.tls_set(
            ca_certs=CA_PATH,
            certfile=CERT_PATH,
            keyfile=KEY_PATH,
            tls_version=ssl.PROTOCOL_TLSv1_2
        )
        self.mqtt_client.on_connect = self._on_connect
        try:
            self.mqtt_client.connect(ENDPOINT, PORT, keepalive=60)
            self.mqtt_client.loop_start()
            print("🔗 MQTT (AWS IoT) inicializado.")
        except Exception as e:
            import traceback
            print(f"⚠️ No se pudo conectar a MQTT: {e}")
            traceback.print_exc()

        # contadores por frame (para CSV por frame)
        self.frame_idx = 0
        self.closed_run_ms = 0.0
        self.per_frame_ms = 0.0
        self.last_ts = time.time()

    def _on_connect(self, client, userdata, flags, rc):
        print("✅ Conectado a AWS IoT Core" if rc == 0 else f"❌ Error de conexión MQTT, código: {rc}")

    def _publish(self, payload, qos=1):
        if not self.mqtt_client:
            return
        try:
            self.mqtt_client.publish(TOPIC, json.dumps(payload), qos=qos)
        except Exception:
            try:
                self.mqtt_client.reconnect()
            except Exception:
                pass

    def _process_metrics(self, marks):
        lms = marks[0]
        ear_r_like = eye_aspect_ratio_like(lms, right=True)
        ear_l_like = eye_aspect_ratio_like(lms, right=False)
        ear_min    = min(ear_r_like, ear_l_like)
        eyes_closed = (ear_r_like < EYE_CLOSED_THRESHOLD and ear_l_like < EYE_CLOSED_THRESHOLD)

        # Parpadeo (flanco)
        blink_evt = 0
        if not self.eyes_closed_prev and eyes_closed:
            self.eyes_closed_prev = True
        elif self.eyes_closed_prev and not eyes_closed:
            self.blink_count_win += 1
            blink_evt = 1
            self.eyes_closed_prev = False

        mar = mouth_aspect_ratio(lms)
        yawn_evt = 0
        if mar > YAWN_MAR_THRESHOLD:
            self.yawn_frames += 1
        else:
            if self.yawn_frames > YAWN_HOLD_FRAMES:
                self.yawn_count_win += 1
                yawn_evt = 1
            self.yawn_frames = 0

        return ear_min, mar, eyes_closed, blink_evt, yawn_evt

    def _estado_por_perclos(self, perclos):
        if perclos >= SOMNO_MIN_PERCLOS:
            return "SOMNOLENCIA"
        if perclos >= FATIGA_PERCLOS:
            return "FATIGA"
        return "NORMAL"

    def _estado_por_blink_rate(self, rate):
        if rate < BLINK_MICRO_MAX:      return "MICROSUEÑO"
        if rate < BLINK_SOMNO_MAX:      return "SOMNOLENCIA"
        if rate <= BLINK_FATIGA_MAX:    return "FATIGA"
        if BLINK_NORM_MIN <= rate <= BLINK_NORM_MAX: return "NORMAL"
        return "NORMAL"

    def _estado_por_yawn_rate(self, rate):
        if rate > YAWN_SOMNO_MIN:                       return "SOMNOLENCIA"
        if YAWN_FATIGA_MIN <= rate <= YAWN_FATIGA_MAX:  return "FATIGA"
        return "NORMAL"

    def _fusion_prioridad(self, estado_ear, perclos_pct, blink_rate, yawn_rate, microsleep_flag):
        if microsleep_flag:
            return "MICROSUEÑO"
        # candidato por tasas
        candidatos = [
            #self._estado_por_blink_rate(blink_rate),
            self._estado_por_yawn_rate(yawn_rate),
            self._estado_por_perclos(perclos_pct)
        ]
        print(f"Candidatos tasas: {candidatos}") 
        rank = {"NORMAL":0, "FATIGA":1, "SOMNOLENCIA":2, "MICROSUEÑO":3}
        estado_tasas = max(candidatos, key=lambda x: rank[x])
        print(f"Estado EAR: {estado_ear}, Estado tasas: {estado_tasas}")
        # prioriza el más grave entre EAR y tasas
        if estado_ear is None:
            return estado_tasas
        return estado_ear if rank[estado_ear] > rank[estado_tasas] else estado_tasas

    def _update_fps_minclose(self):
        if self.min_close_fr is None:
            self._fps_init_frames += 1
            dt = max(time.time() - self._fps_init_time, 1e-3)
            self.fps_est = max(5.0, min(60.0, self._fps_init_frames / dt))
            self.min_close_fr = max(1, int(round((MIN_CLOSE_MS/1000.0) * self.fps_est)))

    def _update_perclos_sliding(self, valid, eyes_closed_raw, t_s):
        while self.perclos_win_times and (t_s - self.perclos_win_times[0] > PERCLOS_WINDOW_S):
            self.perclos_win_times.popleft()
            self.perclos_win.popleft()
        if not valid:
            return
        if eyes_closed_raw:
            self.run_closed += 1
            self.perclos_win.append(0); self.perclos_win_times.append(t_s)
            if self.run_closed >= self.min_close_fr:
                self.perclos_win[-1] = 1
                if self.run_closed == self.min_close_fr:
                    n_back = min(self.min_close_fr - 1, len(self.perclos_win) - 1)
                    for i in range(1, n_back + 1):
                        self.perclos_win[-1 - i] = 1
        else:
            self.run_closed = 0
            self.perclos_win.append(0); self.perclos_win_times.append(t_s)

    def _perclos_current(self):
        denom = len(self.perclos_win)
        if denom == 0: return 0.0, 0.0
        frac = sum(self.perclos_win) / float(denom)
        return frac, (frac * 100.0)

    def run(self):
        try:
            while not self.stop_evt.is_set():
                try:
                    rgb = frame_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                start = time.time()
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self.detector.detect(mp_image)
                t_now = time.time()
                dt_ms = (t_now - self.last_ts) * 1000.0
                self.last_ts = t_now

                valid = False
                ear_min = None
                mar = None
                eyes_closed_raw = 0
                blink_evt = 0
                yawn_evt = 0

                overlay = "Sin rostro"

                if result.face_landmarks:
                    draw_landmarks_on_frame_rgb(rgb, result.face_landmarks)
                    ear_min, mar, eyes_closed, blink_evt, yawn_evt = self._process_metrics(result.face_landmarks)
                    valid = True
                    eyes_closed_raw = 1 if eyes_closed else 0

                    # contadores de ventana
                    if eyes_closed:
                        self.closed_frames += 1
                        self.closed_streak += 1
                        self.closed_run_ms += dt_ms
                    else:
                        self.closed_streak = 0
                        self.closed_run_ms = 0.0

                    self.frames_cnt += 1
                    if self.closed_streak >= MICROSLEEP_FRAMES:
                        self.microsleep_flag = True

                    # cierre de ventana FRAMES_INTERVAL
                    if self.frames_cnt >= FRAMES_INTERVAL:
                        elapsed = max(time.time() - self.window_start_ts, 1e-3)
                        perclos_pct_win = 100.0 * self.closed_frames / float(max(1, FRAMES_INTERVAL))
                        blink_rate  = (self.blink_count_win * 60.0) / elapsed
                        yawn_rate   = (self.yawn_count_win  * 60.0) / elapsed

                        # PERCLOS deslizante (60s)
                        perclos_frac, perclos_pct_sliding = self._perclos_current()

                        # Estado por EAR (instantáneo, usamos el ear_min del último frame disponible)
                        estado_ear = estado_por_ear(ear_min)

                        estado_final = self._fusion_prioridad(
                            estado_ear=estado_ear,
                            perclos_pct=max(perclos_pct_win, perclos_pct_sliding),
                            blink_rate=blink_rate,
                            yawn_rate=yawn_rate,
                            microsleep_flag=self.microsleep_flag
                        )
                        print(f"Estado final: {estado_final} | PERCLOS win: {perclos_pct_win:.2f}%, sliding: {perclos_pct_sliding:.2f}% | ")

                        # CSV por ventana
                        self.csv_writer.writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            estado_final,
                            round(max(perclos_pct_win, perclos_pct_sliding), 2),
                            round(blink_rate, 2),
                            round(yawn_rate, 2),
                            int(self.blink_count_win),
                            int(self.yawn_count_win)
                        ])

                        # MQTT si alerta
                        if estado_final != "NORMAL":
                            self._publish({
                                "device_id": CLIENT_ID,
                                "estado": estado_final,
                                "perclos": round(max(perclos_pct_win, perclos_pct_sliding), 2),
                                "blink_rate": round(blink_rate, 2),
                                "yawn_rate": round(yawn_rate, 2),
                                "blinks": int(self.blink_count_win),
                                "yawns": int(self.yawn_count_win),
                                "ts": int(time.time())
                            }, qos=1)

                        # reset ventana
                        self.frames_cnt = 0
                        self.closed_frames = 0
                        self.closed_streak = 0
                        self.microsleep_flag = False
                        self.blink_count_win = 0
                        self.yawn_count_win  = 0
                        self.window_start_ts = time.time()

                    # overlay rápido por EAR
                    estado_ear = estado_por_ear(ear_min)
                    overlay = estado_ear if estado_ear else "NORMAL"
                else:
                    put_text_rgb(rgb, "Sin rostro", (12, 56), (0, 255, 255))

                # PERCLOS deslizante 60s
                self._update_fps_minclose()
                self._update_perclos_sliding(valid, eyes_closed_raw, t_now)
                perclos_frac, perclos_pct = self._perclos_current()

                # CSV por frame
                if self.frame_csv_file:
                    estado_ear = estado_por_ear(ear_min)
                    # Estado final instantáneo: si EAR dice > grave, úsalo; si no, NORMAL (el de ventana ya se guarda arriba)
                    estado_instant = estado_ear if estado_ear else "NORMAL"
                    self.frame_csv_writer.writerow([
                        self.frame_idx, f"{t_now:.3f}",
                        f"{ear_min:.4f}" if ear_min is not None else "",
                        f"{mar:.4f}" if mar is not None else "",
                        int(eyes_closed_raw), int(blink_evt), int(yawn_evt),
                        int(self.blink_count_win), int(self.yawn_count_win),
                        f"{self.closed_run_ms:.2f}", f"{dt_ms:.2f}",
                        f"{perclos_pct:.2f}", estado_instant, estado_instant
                    ])
                    self.frame_idx += 1

                # UI
                put_text_rgb(rgb, overlay, (12, 28))
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if annot_queue.full():
                    try: annot_queue.get_nowait()
                    except queue.Empty: pass
                annot_queue.put(bgr)

                frame_queue.task_done()
        except Exception as e:
            import traceback
            print(f"❌ Excepción en InferenceWorker: {e}")
            traceback.print_exc()
            self.stop_evt.set()
        finally:
            try: self.csv_file.flush(); self.csv_file.close()
            except Exception: pass
            try:
                if self.frame_csv_file:
                    self.frame_csv_file.flush(); self.frame_csv_file.close()
            except Exception: pass
            try:
                if self.mqtt_client:
                    self.mqtt_client.loop_stop(); self.mqtt_client.disconnect()
            except Exception: pass

# ================== MAIN / UI ==================
def main():
    print("Iniciando hilos (captura / inferencia)…")
    cap_t = CaptureWorker(stop_event)
    inf_t = InferenceWorker(stop_event)

    cap_t.start()
    inf_t.start()

    print(f"🟢 Presiona 'q' para salir. HEADLESS={HEADLESS}")
    try:
        while not stop_event.is_set():
            try:
                frame = annot_queue.get(timeout=0.3)
                if not HEADLESS:
                    cv2.imshow("Fatigue Monitor - OV5647 (PERCLOS+EAR bands)", frame)
                    if (cv2.waitKey(1) & 0xFF) == ord('q'):
                        stop_event.set()
            except queue.Empty:
                pass
    finally:
        stop_event.set()
        cap_t.join(timeout=2.0)
        inf_t.join(timeout=2.0)
        if not HEADLESS:
            cv2.destroyAllWindows()
        print("Finalizado.")

if __name__ == "__main__":
    main()
