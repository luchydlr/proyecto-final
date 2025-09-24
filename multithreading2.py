import cv2, numpy as np, math, time, queue, threading, os
from datetime import datetime, timezone
from picamera2 import Picamera2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

# ================== CONFIG ==================
load_dotenv()
DEVICE_ID            = os.getenv("DEVICE_ID", "rsp5")

# Cámara
CAM_SIZE             = (640, 480)

# Umbrales y ventana (ajústalos en pruebas)
FRAMES_INTERVAL      = 30          # frames por ventana (p. ej. ~1 s si ~30FPS)
SOMNO_MIN_PERCLOS    = 40.0        # 40–59.99% => SOMNOLENCIA
FATIGA_PERCLOS       = 60.0        # >=60% => FATIGA
YAWNS_FOR_FATIGA_WIN = 2           # PERCLOS>=50% y ≥2 bostezos -> FATIGA
YAWNS_FOR_SOMNO_WIN  = 1           # ≥1 bostezo -> SOMNOLENCIA
MICROSLEEP_FRAMES    = 15          # ojos cerrados consecutivos -> MICROSUEÑO

# EAR/MAR (tus índices/heurísticas)
EYE_CLOSED_THRESHOLD = 0.009
YAWN_MAR_THRESHOLD   = 0.65
YAWN_HOLD_FRAMES     = 5

# MySQL (RDS)
MYSQL_HOST     = os.getenv("MYSQL_HOST")
MYSQL_PORT     = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_DB       = os.getenv("MYSQL_DB")
MYSQL_USER     = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")

# Colas & stop
frame_queue = queue.Queue(maxsize=1)   # capture -> inference
annot_queue = queue.Queue(maxsize=1)   # inference -> UI
db_queue    = queue.Queue(maxsize=500) # inference -> DB
stop_event  = threading.Event()

# ================== Mediapipe detector ==================
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

# ================== Métricas ==================
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
    return (v_dist / h_dist) if h_dist > 0 else 0.0

def draw_landmarks_on_frame_rgb(rgb_frame, marks):
    h, w = rgb_frame.shape[:2]
    for lm in marks[0]:
        x = int(lm.x * w); y = int(lm.y * h)
        cv2.circle(rgb_frame, (x, y), 1, (0, 255, 0), -1)

def put_text_rgb(rgb_frame, text, pos=(12, 28), color=(0, 255, 255)):
    cv2.putText(rgb_frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

# ================== Capture Thread ==================
class CaptureWorker(threading.Thread):
    def __init__(self, stop_evt):
        super().__init__(daemon=True); self.stop_evt = stop_evt; self.picam2 = None
    def run(self):
        try:
            self.picam2 = Picamera2()
            cfg = self.picam2.create_preview_configuration(main={"format":"RGB888","size":CAM_SIZE})
            self.picam2.configure(cfg); self.picam2.start(); time.sleep(0.2)
            print("🟢 Cámara lista (Picamera2).")
        except Exception as e:
            print(f"❌ Error cámara: {e}"); self.stop_evt.set(); return
        while not self.stop_evt.is_set():
            try: rgb = self.picam2.capture_array()
            except Exception: time.sleep(0.005); continue
            if frame_queue.full():
                try: frame_queue.get_nowait()
                except queue.Empty: pass
            frame_queue.put(rgb)
        try: self.picam2.stop()
        except Exception: pass

# ================== DB Worker (MySQL) ==================
class DBWorker(threading.Thread):
    """ Inserta filas en alerts """
    def __init__(self, stop_evt, batch_size=20, flush_sec=2.0):
        super().__init__(daemon=True)
        self.stop_evt=stop_evt; self.batch_size=batch_size; self.flush_sec=flush_sec
        self.buf=[]; self.conn=None; self.last_flush=time.time()
    def _connect(self):
        if self.conn and self.conn.is_connected(): return
        self.conn = mysql.connector.connect(
            host=MYSQL_HOST, port=MYSQL_PORT, database=MYSQL_DB,
            user=MYSQL_USER, password=MYSQL_PASSWORD,
            autocommit=True, charset="utf8mb4", collation="utf8mb4_general_ci"
        )
    def _flush(self):
        if not self.buf: return
        try:
            self._connect()
            rows=[(x["device_id"],x["estado"],x["perclos"],x["blinks"],x["yawns"],x["ts"]) for x in self.buf]
            sql="""INSERT INTO alerts (device_id, estado, perclos, blinks, yawns, ts)
                   VALUES (%s,%s,%s,%s,%s,%s)"""
            with self.conn.cursor() as cur: cur.executemany(sql, rows)
            self.buf.clear(); self.last_flush=time.time()
        except Error as e:
            print(f"❌ Error DB: {e}"); time.sleep(1)
    def run(self):
        while not self.stop_evt.is_set():
            try: item = db_queue.get(timeout=0.3); self.buf.append(item); db_queue.task_done()
            except queue.Empty: pass
            if len(self.buf)>=self.batch_size or (time.time()-self.last_flush)>=self.flush_sec: self._flush()
        self._flush()
        if self.conn:
            try: self.conn.close()
            except Exception: pass

# ================== Inference Thread ==================
class InferenceWorker(threading.Thread):
    def __init__(self, stop_evt):
        super().__init__(daemon=True); self.stop_evt=stop_evt
        self.detector = initialize_detector()
        # contadores globales
        self.frames_cnt=0; self.closed_frames=0; self.closed_streak=0
        self.blink_count=0; self.yawn_count=0; self.eyes_closed_prev=False; self.yawn_frames=0
        # contadores por ventana
        self.prev_yawns=0; self.prev_blinks=0

    def _process_metrics(self, marks):
        lms = marks[0]
        ear_r = eye_aspect_ratio(lms, right=True)
        ear_l = eye_aspect_ratio(lms, right=False)
        eyes_closed = (ear_r < EYE_CLOSED_THRESHOLD and ear_l < EYE_CLOSED_THRESHOLD)

        # consecutivos (microsueño) + perclos por ventana
        if eyes_closed:
            self.closed_streak += 1
            self.closed_frames += 1
        else:
            self.closed_streak = 0

        # blinks por flanco
        if not self.eyes_closed_prev and eyes_closed:
            self.eyes_closed_prev = True
        elif self.eyes_closed_prev and not eyes_closed:
            self.blink_count += 1
            self.eyes_closed_prev = False

        # bostezos
        mar = mouth_aspect_ratio(lms)
        if mar > YAWN_MAR_THRESHOLD:
            self.yawn_frames += 1
        else:
            if self.yawn_frames > YAWN_HOLD_FRAMES:
                self.yawn_count += 1
            self.yawn_frames = 0

        return eyes_closed, mar, ear_l, ear_r

    def _decide_state_window(self, perclos, yawns_win):
        # Prioridad: FATIGA > SOMNOLENCIA
        if perclos is None: return None
        if perclos >= FATIGA_PERCLOS: return "FATIGA"
        if perclos >= 50.0 and yawns_win >= YAWNS_FOR_FATIGA_WIN: return "FATIGA"
        if SOMNO_MIN_PERCLOS <= perclos < FATIGA_PERCLOS: return "SOMNOLENCIA"
        if yawns_win >= YAWNS_FOR_SOMNO_WIN: return "SOMNOLENCIA"
        return None

    def _flush_window_and_insert(self, perclos):
        yawns_win  = self.yawn_count  - self.prev_yawns
        blinks_win = self.blink_count - self.prev_blinks
        estado = self._decide_state_window(perclos, yawns_win)
        if estado is not None:
            db_queue.put({
                "device_id": DEVICE_ID,
                "estado": estado,
                "perclos": round(perclos,2) if perclos is not None else None,
                "blinks": int(blinks_win),
                "yawns": int(yawns_win),
                "ts": datetime.now(timezone.utc)
            })
        # reset ventana
        self.prev_yawns  = self.yawn_count
        self.prev_blinks = self.blink_count
        self.closed_frames = 0
        self.frames_cnt = 0

    def run(self):
        while not self.stop_evt.is_set():
            try: rgb = frame_queue.get(timeout=0.2)
            except queue.Empty: continue

            state_txt = "AWAKE"
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.detector.detect(mp_image)

            if result.face_landmarks:
                draw_landmarks_on_frame_rgb(rgb, result.face_landmarks)
                eyes_closed, mar, ear_l, ear_r = self._process_metrics(result.face_landmarks)
                if eyes_closed: state_txt = "EYES CLOSED"
                if mar > YAWN_MAR_THRESHOLD: state_txt = "YAWNING"

                # MICROSUEÑO (evento inmediato con fila dedicada)
                if self.closed_streak >= MICROSLEEP_FRAMES:
                    db_queue.put({
                        "device_id": DEVICE_ID,
                        "estado": "MICROSUEÑO",
                        "perclos": None,
                        "blinks": int(self.blink_count - self.prev_blinks),
                        "yawns": int(self.yawn_count  - self.prev_yawns),
                        "ts": datetime.now(timezone.utc)
                    })
                    self.closed_streak = 0

            # Ventana por frames: PERCLOS + decisión
            self.frames_cnt += 1
            if self.frames_cnt >= FRAMES_INTERVAL:
                perclos = 100.0 * self.closed_frames / max(FRAMES_INTERVAL,1)
                self._flush_window_and_insert(perclos)

            # UI
            put_text_rgb(rgb, state_txt, (12,28))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if annot_queue.full():
                try: annot_queue.get_nowait()
                except queue.Empty: pass
            annot_queue.put(bgr)
            frame_queue.task_done()

# ================== Main / UI ==================
def main():
    print("Iniciando hilos…")
    cap_t = CaptureWorker(stop_event)
    inf_t = InferenceWorker(stop_event)
    db_t  = DBWorker(stop_event, batch_size=30, flush_sec=2.0)

    cap_t.start(); inf_t.start(); db_t.start()
    print("🟢 Presiona 'q' para salir.")
    try:
        while not stop_event.is_set():
            try:
                frame = annot_queue.get(timeout=0.3)
                cv2.imshow("Fatigue - RSP (MySQL estados)", frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    stop_event.set()
            except queue.Empty:
                pass
    finally:
        stop_event.set()
        cap_t.join(timeout=2.0); inf_t.join(timeout=2.0); db_t.join(timeout=2.0)
        cv2.destroyAllWindows(); print("Finalizado.")

if __name__ == "__main__":
    main()
