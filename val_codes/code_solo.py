# ============================================
# Batch de videos con EAR clásico (6 pts) + CSVs
# Genera:
#   - base_per_frame.csv
#   - base_per_{AGG_INTERVAL_S}s.csv  (mismo esquema que anotaciones)
# ============================================
import os, glob, csv, math
from collections import deque, Counter
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from tqdm import tqdm

# -----------------------------
# RUTAS / CONFIG
# -----------------------------
MODEL_PATH   = "/content/face_landmarker.task"      # <-- pon aquí tu .task
INPUT_DIR    = "/content/videos"                    # <-- carpeta con .mp4/.mov/.avi...
OUT_DIR      = "/content/salidas"                   # <-- carpeta de salida
EXTS         = (".mp4", ".mov", ".avi", ".mkv")
RECURSIVE    = False
SKIP_IF_EXISTS = True

# Agregación por tiempo para comparar con anotaciones
WRITE_PER_INTERVAL_CSV = True
AGG_INTERVAL_S = 1  # usa 1 si tus anotaciones son por segundo exacto

# -----------------------------
# Umbrales y ventanas (tiempo)
# -----------------------------
EAR_THR_CLOSE = 0.20
EAR_HYST      = 0.06
BLINK_MIN_MS  = 60
BLINK_MAX_MS  = 500
YAWN_MIN_MS   = 1000
MICRO_MS      = 1500
RATES_WINDOW_S = 60.0   # ventana deslizante para tasas/min (si quieres añadirlas al per-frame)

# Umbral de decisión PERCLOS por ventana agregada (si te interesa etiquetar estado)
PERCLOS_THR   = 60.0    # %

# -----------------------------
# Landmarks (FaceMesh)
# -----------------------------
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX     = [13, 14, 78, 308]

# -----------------------------
# Utilidades
# -----------------------------
def hypot2(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)

def aspect_ratio_6(lms, idxs, w, h):
    """
    EAR clásico:
      EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
    """
    try:
        p = [(lms[i].x * w, lms[i].y * h) for i in idxs[:6]]
        (x1,y1),(x2,y2),(x3,y3),(x4,y4),(x5,y5),(x6,y6) = p
        v1 = hypot2(x2,y2,x6,y6)
        v2 = hypot2(x3,y3,x5,y5)
        h1 = hypot2(x1,y1,x4,y4)
        return (v1 + v2) / (2.0 * h1) if h1 > 0 else None
    except:
        return None

def mouth_aspect_ratio(lms, idxs, w, h):
    try:
        top = (lms[idxs[0]].x * w, lms[idxs[0]].y * h)
        bot = (lms[idxs[1]].x * w, lms[idxs[1]].y * h)
        lef = (lms[idxs[2]].x * w, lms[idxs[2]].y * h)
        rig = (lms[idxs[3]].x * w, lms[idxs[3]].y * h)
        vertical   = hypot2(top[0], top[1], bot[0], bot[1])
        horizontal = hypot2(lef[0], lef[1], rig[0], rig[1])
        return vertical / horizontal if horizontal > 0 else None
    except:
        return None

def safe_open_csv(path, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, 'w', newline='', encoding='utf-8')
    w = csv.writer(f)
    w.writerow(header)
    return f, w

def discover_videos(folder: str, exts=EXTS, recursive=False):
    pattern = "**/*" if recursive else "*"
    paths = [p for p in glob.glob(os.path.join(folder, pattern), recursive=recursive)
             if os.path.splitext(p)[1].lower() in exts]
    return sorted(paths)

def mode_estado_from_perclos(perclos_pct, closed_run_ms):
    # opcional: etiqueta simple por intervalo; microsueño tiene prioridad
    if closed_run_ms >= MICRO_MS:
        return "MICROSUEÑO"
    return "SLEEPY" if perclos_pct > PERCLOS_THR else "AWAKE"

# -----------------------------
# Detector (Face Landmarker)
# -----------------------------
def build_detector(model_path: str):
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)

# -----------------------------
# Procesar UN video
# -----------------------------
def analyze_video(video_path: str, out_dir: str, agg_interval_s: int = 1, write_per_interval_csv: bool = True):
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_csv_frame = os.path.join(out_dir, f"{base}_per_frame.csv")
    out_csv_int   = os.path.join(out_dir, f"{base}_per_{agg_interval_s}s.csv")

    if SKIP_IF_EXISTS and os.path.exists(out_csv_frame) and (not write_per_interval_csv or os.path.exists(out_csv_int)):
        print(f"⏭️  Ya existe salida para '{base}', omitiendo.")
        return

    detector = build_detector(MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ No se pudo abrir: {video_path}")
        detector.close()
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    fps = max(10.0, min(120.0, fps))

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 640)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Histeresis
    open_thr = EAR_THR_CLOSE * (1.0 + EAR_HYST)

    # Contadores
    eye_closed = 0
    eye_closed_prev = 0
    closed_run_ms = 0.0
    yawn_open_ms  = 0.0
    yawn_count = 0
    blink_count = 0

    # Para tasas/min (si quieres añadir después)
    RATES_WIN_FR = int(RATES_WINDOW_S * fps)
    events_window = deque()  # (frame_id, 'blink'|'yawn')

    # Buffers por intervalo
    # Esquema para comparar con anotaciones:
    # ["Second","PERCLOS","Blinks","Yawns","Avg_EAR","Avg_MAR","Mode_Estado"]
    interval = {
        'start_s': None,
        'elapsed_ms': 0.0,
        'closed_ms': 0.0,
        'blinks': 0,
        'yawns': 0,
        'ear_sum': 0.0,
        'ear_cnt': 0,
        'mar_sum': 0.0,
        'mar_cnt': 0,
        'max_closed_run_ms': 0.0,  # para priorizar microsueño si ocurre en el intervalo
    }

    # CSVs
    fout, wframe = safe_open_csv(
        out_csv_frame,
        ["Frame_ID","Timestamp_s","EAR","MAR","EyeClosed",
         "BlinkEvent","YawnEvent","Blinks_acc","Yawns_acc","ClosedRun_ms"]
    )
    if write_per_interval_csv:
        fsec, winter = safe_open_csv(
            out_csv_int,
            ["Second","PERCLOS","Blinks","Yawns","Avg_EAR","Avg_MAR","Mode_Estado"]
        )
    else:
        fsec = winter = None

    print(f"\n🎥 {os.path.basename(video_path)} | FPS={fps:.2f} | {W}x{H} | Frames={total_frames or 'desconocido'}")

    idx = 0
    # Tiempo: sacamos dt por frame con 1/fps (suficiente para videos)
    dt_ms = 1000.0 / fps

    try:
        with tqdm(total=(total_frames if total_frames>0 else None), unit="f", dynamic_ncols=True) as pbar:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                # OpenCV entrega BGR → Mediapipe espera RGB
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                res = detector.detect(mp_img)

                ts = idx / fps  # segundos (float)

                ear = None
                mar = None
                blink_event = 0
                yawn_event  = 0

                if res.face_landmarks:
                    lms = res.face_landmarks[0]
                    ear_l = aspect_ratio_6(lms, LEFT_EYE_IDX,  W, H)
                    ear_r = aspect_ratio_6(lms, RIGHT_EYE_IDX, W, H)
                    mar   = mouth_aspect_ratio(lms, MOUTH_IDX,   W, H)

                    # limpieza mínima
                    if ear_l is not None and (ear_l < 0.02 or ear_l > 0.8): ear_l = None
                    if ear_r is not None and (ear_r < 0.02 or ear_r > 0.8): ear_r = None
                    if mar   is not None and (mar   < 0.10 or mar   > 1.20): mar   = None

                    if (ear_l is not None) and (ear_r is not None):
                        ear = min(ear_l, ear_r)
                        # histéresis cierre/apertura
                        if eye_closed:
                            eye_closed = 0 if ear >= open_thr else 1
                        else:
                            eye_closed = 1 if ear < EAR_THR_CLOSE else 0

                # ---- Integración temporal (dt por frame)
                if eye_closed:
                    closed_run_ms += dt_ms
                    interval['closed_ms'] += dt_ms
                    if closed_run_ms > interval['max_closed_run_ms']:
                        interval['max_closed_run_ms'] = closed_run_ms
                else:
                    if eye_closed_prev and (BLINK_MIN_MS <= closed_run_ms <= BLINK_MAX_MS):
                        blink_count += 1
                        blink_event = 1
                        events_window.append((idx, 'blink'))
                    closed_run_ms = 0.0
                eye_closed_prev = eye_closed

                # Bostezo por tiempo
                if mar is not None and mar > 0.65:
                    yawn_open_ms += dt_ms
                else:
                    if yawn_open_ms >= YAWN_MIN_MS:
                        yawn_count += 1
                        yawn_event = 1
                        events_window.append((idx, 'yawn'))
                    yawn_open_ms = 0.0

                # ----- CSV per-frame (para auditoría fina)
                def f(x, nd=4): return "" if x is None else f"{x:.{nd}f}"
                wframe.writerow([
                    idx, f"{ts:.3f}", f(ear), f(mar),
                    1 if eye_closed else 0, blink_event, yawn_event,
                    blink_count, yawn_count, f(closed_run_ms, 1)
                ])

                # ----- Buffer por intervalo
                if interval['start_s'] is None:
                    interval['start_s'] = int(ts // agg_interval_s) * agg_interval_s

                interval['elapsed_ms'] += dt_ms
                if ear is not None:
                    interval['ear_sum'] += ear
                    interval['ear_cnt'] += 1
                if mar is not None:
                    interval['mar_sum'] += mar
                    interval['mar_cnt'] += 1
                if blink_event:
                    interval['blinks'] += 1
                if yawn_event:
                    interval['yawns'] += 1

                # ¿cerramos intervalo?
                interval_end_s = interval['start_s'] + agg_interval_s
                if ts >= interval_end_s - 1e-6:
                    # perclos en el intervalo
                    perclos = 100.0 * (interval['closed_ms'] / max(1.0, interval['elapsed_ms']))
                    avg_ear = interval['ear_sum']/interval['ear_cnt'] if interval['ear_cnt'] > 0 else ""
                    avg_mar = interval['mar_sum']/interval['mar_cnt'] if interval['mar_cnt'] > 0 else ""

                    # Estado modal simple basado en perclos / microsueño del intervalo
                    mode_estado = mode_estado_from_perclos(perclos, interval['max_closed_run_ms'])

                    if write_per_interval_csv:
                        winter.writerow([
                            interval['start_s'],               # Second (inicio del intervalo)
                            f"{perclos:.3f}",                   # PERCLOS (0-1 si prefieres multiplica por 1.0 o déjalo en %)
                            interval['blinks'],                 # Blinks
                            interval['yawns'],                  # Yawns
                            f"{avg_ear:.4f}" if avg_ear != "" else "",
                            f"{avg_mar:.4f}" if avg_mar != "" else "",
                            mode_estado
                        ])

                    # reset intervalo
                    interval = {
                        'start_s': int(ts // agg_interval_s) * agg_interval_s,
                        'elapsed_ms': 0.0,
                        'closed_ms': 0.0,
                        'blinks': 0,
                        'yawns': 0,
                        'ear_sum': 0.0,
                        'ear_cnt': 0,
                        'mar_sum': 0.0,
                        'mar_cnt': 0,
                        'max_closed_run_ms': 0.0,
                    }

                idx += 1
                pbar.update(1)

    finally:
        cap.release()
        detector.close()
        fout.close()
        if write_per_interval_csv and fsec:
            fsec.close()

    print(f"✅ Guardado:\n - {out_csv_frame}")
    if write_per_interval_csv:
        print(f" - {out_csv_int}")

# -----------------------------
# BATCH
# -----------------------------
os.makedirs(OUT_DIR, exist_ok=True)
videos = discover_videos(INPUT_DIR, EXTS, RECURSIVE)
print(f"🔎 Encontrados {len(videos)} videos en {INPUT_DIR}")

for i, vp in enumerate(videos, 1):
    try:
        print(f"\n({i}/{len(videos)}) {vp}")
        analyze_video(
            video_path=vp,
            out_dir=OUT_DIR,
            agg_interval_s=AGG_INTERVAL_S,
            write_per_interval_csv=WRITE_PER_INTERVAL_CSV
        )
    except Exception as e:
        print(f"⚠️ Error procesando {vp}: {e}")
