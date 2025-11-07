#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch por videos (RPi ready) — SOLO EAR decide el Estado (conserva Microsueño por racha cerrada)
Genera por video:
  - base_per_frame.csv  (EAR, MAR, EyeClosed, Blink/Yawn events, tasas/min, ClosedRun_ms, Estado)
  - base_per_{AGG_INTERVAL_S}s.csv  (Start_s, PERCLOS, Blinks, Yawns, Avg_EAR, Avg_MAR, Mode_Estado)
  - base_perf_summary.json / .csv (CPU %, RAM, temp, FPS, etc. si disponibles)

Dependencias sugeridas:
  sudo apt-get update && sudo apt-get install -y python3-opencv
  pip install mediapipe tqdm psutil numpy
"""

import os, glob, csv, math, collections, json, time, statistics, shutil, subprocess, threading
import numpy as np
import cv2
from tqdm import tqdm

# psutil opcional (para CPU/RAM). Si no está, seguimos sin esas métricas.
try:
    import psutil
except Exception:
    psutil = None

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# -----------------------------
# RUTAS / CONFIG (EDITA A TU GUSTO)
# -----------------------------
MODEL_PATH   = "/home/lucianadelarosa/Desktop/proyecto-final/facemesh/face_landmarker.task"
INPUT_DIR    = "/home/lucianadelarosa/Desktop/UTA-RLDD/01"
OUT_DIR      = "/home/lucianadelarosa/Desktop/proyecto-final/val-annot/code_solo/UTA-RLDD/01"
EXTS         = (".mp4", ".mov", ".avi", ".mkv", ".MOV")
RECURSIVE    = False
SKIP_IF_EXISTS = False

# Agregación temporal para el CSV por ventanas
AGG_INTERVAL_S = 2

# Ventana para tasas/min y PERCLOS (fijo, NO adaptativo)
RATES_WINDOW_S = 60.0

# -----------------------------
# UMBRALES / TIEMPOS (ms)
# -----------------------------
# EAR clásico con histéresis (abre/cierra)
EYE_THR_CLOSE = 0.21   # EAR por debajo → cerrado
EYE_THR_OPEN  = 0.25   # EAR por encima → abierto

BLINK_MIN_MS  = 60     # duración mínima de cierre para contar blink
BLINK_MAX_MS  = 500
YAWN_MIN_MS   = 1000
MICRO_MS      = 1500   # ≥ 1.5 s de cierre continuo = Microsueño
YAWN_THR      = 0.65   # umbral MAR para activar conteo de bostezo

# -----------------------------
# EAR_BANDS (tabla fija para decisión por EAR)
# -----------------------------
EAR_BANDS = {
    "NORMAL":      (0.25, 1.00),
    "FATIGA":      (0.21, 0.25),
    "SOMNOLENCIA": (0.15, 0.21),
    "MICROSUEÑO":  (0.00, 0.15),
}
RANK = {"NORMAL": 0, "FATIGA": 1, "SOMNOLENCIA": 2, "MICROSUEÑO": 3}

# -----------------------------
# Tabla Blinks/Yawns por minuto (solo diagnóstico; no decide Estado)
# -----------------------------
def estado_by_blinks_yawns(blinks_pm, yawns_pm):
    # Normal
    if (17 <= blinks_pm <= 25) and (yawns_pm <= 1):
        return "NORMAL"
    # Fatiga
    if (12 <= blinks_pm <= 16) and (1 <= yawns_pm <= 4):
        return "FATIGA"
    # Somnolencia
    if (6 <= blinks_pm <= 12) and (yawns_pm > 4):
        return "SOMNOLENCIA"
    # fallback suave según blinks
    if blinks_pm < 6:
        return "SOMNOLENCIA"
    if blinks_pm <= 12:
        return "FATIGA"
    return "NORMAL"

# -----------------------------
# Landmarks (FaceMesh)
# -----------------------------
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_VERT_PAIRS = [(13,14), (82,87), (312,317)]
MOUTH_HORZ = (78, 308)

# -----------------------------
# Utilidades geométricas
# -----------------------------
def hypot2(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)

def aspect_ratio_6(lms, idxs, w, h):
    """EAR clásico: (|p2-p6| + |p3-p5|) / (2*|p1-p4|)"""
    try:
        p = [(lms[i].x * w, lms[i].y * h) for i in idxs[:6]]
        (x1,y1),(x2,y2),(x3,y3),(x4,y4),(x5,y5),(x6,y6) = p
        v1 = hypot2(x2,y2,x6,y6)
        v2 = hypot2(x3,y3,x5,y5)
        h1 = hypot2(x1,y1,x4,y4)
        return (v1 + v2) / (2.0 * h1) if h1 > 0 else None
    except Exception:
        return None

def mouth_aspect_ratio_robust(lms, w, h):
    """MAR robusto: promedio 3 verticales / horizontal 78–308, con clamp suave."""
    try:
        vds = []
        for a,b in MOUTH_VERT_PAIRS:
            ta = (lms[a].x*w, lms[a].y*h); tb = (lms[b].x*w, lms[b].y*h)
            vds.append(hypot2(ta[0], ta[1], tb[0], tb[1]))
        left = (lms[MOUTH_HORZ[0]].x*w, lms[MOUTH_HORZ[0]].y*h)
        righ = (lms[MOUTH_HORZ[1]].x*w, lms[MOUTH_HORZ[1]].y*h)
        hd = hypot2(left[0], left[1], righ[0], righ[1])
        if hd <= 0:
            return None
        mar = float(np.mean(vds)) / hd
        if not (0.10 <= mar <= 1.20):
            return None
        return mar
    except Exception:
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
             if os.path.splitext(p)[1].lower() in tuple(e.lower() for e in exts)]
    return sorted(paths)

# -----------------------------
# Detector
# -----------------------------
def build_detector(model_path: str, num_threads: int = 4):
    """Crea FaceLandmarker con compatibilidad de versiones para num_threads."""
    try:
        base_options = mp_python.BaseOptions(
            model_asset_path=model_path,
            num_threads=num_threads
        )
    except TypeError:
        base_options = mp_python.BaseOptions(model_asset_path=model_path)

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)

# -----------------------------
# Métricas de sistema (opcionales)
# -----------------------------
def read_cpu_temp_c():
    """Intenta leer temperatura CPU en Raspberry Pi via vcgencmd."""
    try:
        if shutil.which("vcgencmd"):
            out = subprocess.check_output(["vcgencmd", "measure_temp"]).decode("utf-8").strip()
            # formato: temp=48.2'C
            val = out.split("=")[1].split("'")[0]
            return float(val)
    except Exception:
        pass
    return None

# -----------------------------
# Helpers de estado
# -----------------------------
def estado_by_perclos(perclos_pct):
    """Referencial (no se usa para decidir Estado en esta versión SOLO EAR)."""
    if perclos_pct >= 80:
        return "MICROSUEÑO"
    if perclos_pct >= 60:
        return "SOMNOLENCIA"
    if perclos_pct >= 40:
        return "FATIGA"
    return "NORMAL"

def estado_by_ear(ear):
    if ear is None:
        return "NORMAL"
    for cls, (lo, hi) in EAR_BANDS.items():
        if lo <= ear < hi:
            return cls
    return "NORMAL"

def fuse_states(*candidatos):
    """Si en algún momento quisieras fusionar; aquí queda por compatibilidad."""
    best = "NORMAL"; best_r = -1
    for c in candidatos:
        r = RANK.get(c, 0)
        if r > best_r:
            best = c; best_r = r
    return best

# -----------------------------
# Procesar UN video
# -----------------------------
def analyze_video(video_path: str, out_dir: str, agg_interval_s: int = 2):
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_csv_frame = os.path.join(out_dir, f"{base}_per_frame.csv")
    out_csv_int   = os.path.join(out_dir, f"{base}_per_{agg_interval_s}s.csv")
    out_perf_json = os.path.join(out_dir, f"{base}_perf_summary.json")
    out_perf_csv  = os.path.join(out_dir, f"{base}_perf_summary.csv")

    if SKIP_IF_EXISTS and os.path.exists(out_csv_frame) and os.path.exists(out_csv_int):
        print(f"⏭️  Ya existe salida para '{base}', omitiendo.")
        return

    os.makedirs(out_dir, exist_ok=True)

    detector = build_detector(MODEL_PATH, num_threads=4)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ No se pudo abrir: {video_path}")
        detector.close()
        return

    fps_nom = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fps_nom = max(10.0, min(120.0, fps_nom))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 640)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    total_frames_nom = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Estados internos
    eye_closed = 0
    eye_closed_prev = 0
    closed_run_ms = 0.0
    yawn_open_ms  = 0.0
    yawns_acc = 0
    blinks_acc = 0

    # Ventanas deslizantes (60 s)
    events_win = collections.deque()     # (ts, 'blink'|'yawn')
    perclos_win = collections.deque()    # (ts, closed_ms, elapsed_ms)
    perclos_win_closed = 0.0
    perclos_win_elapsed = 0.0

    # Intervalo exacto por frames (para CSV por 2 s)
    FRAMES_PER_INT = int(round(agg_interval_s * fps_nom))
    interval_idx = 0
    int_closed_ms = 0.0
    int_elapsed_ms = 0.0
    int_blinks = 0
    int_yawns  = 0
    int_ear_sum = 0.0
    int_ear_cnt = 0
    int_mar_sum = 0.0
    int_mar_cnt = 0
    estados_intervalo = []

    # CSVs con headers EXACTOS
    f_frame, wframe = safe_open_csv(
        out_csv_frame,
        ["Frame_ID","Timestamp_s","EAR","MAR","EyeClosed","BlinkEvent","YawnEvent",
         "Blinks_acc","Yawns_acc","Blinks_per_min","Yawns_per_min","ClosedRun_ms","Estado"]
    )
    f_int, winter = safe_open_csv(
        out_csv_int,
        ["Start_2s","PERCLOS","Blinks","Yawns","Avg_EAR","Avg_MAR","Mode_Estado"]
    )

    print(f"\n🎥 {os.path.basename(video_path)} | FPS_nom={fps_nom:.2f} | {W}x{H} | Frames_nom={total_frames_nom or 'desconocido'}")

    idx = 0
    dt_ms = 1000.0 / fps_nom

    # PERF MON
    frame_times = []
    t0 = time.perf_counter()
    if psutil:
        proc = psutil.Process(os.getpid())
        cpu_samples = []
        rss_samples = []
        _ = proc.cpu_percent(interval=None)
    else:
        proc = None

    try:
        with tqdm(total=(total_frames_nom if total_frames_nom>0 else None), unit="f", dynamic_ncols=True) as pbar:
            while True:
                t_frame_start = time.perf_counter()

                ok, frame_bgr = cap.read()
                if not ok:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                res = detector.detect(mp_img)

                ts = idx / fps_nom
                ear = None
                mar = None
                blink_event = 0
                yawn_event  = 0

                if res.face_landmarks:
                    lms = res.face_landmarks[0]
                    ear_l = aspect_ratio_6(lms, LEFT_EYE_IDX,  W, H)
                    ear_r = aspect_ratio_6(lms, RIGHT_EYE_IDX, W, H)
                    mar   = mouth_aspect_ratio_robust(lms, W, H)

                    # limpieza
                    if ear_l is not None and (ear_l < 0.02 or ear_l > 0.80): ear_l = None
                    if ear_r is not None and (ear_r < 0.02 or ear_r > 0.80): ear_r = None

                    if (ear_l is not None) and (ear_r is not None):
                        ear = min(ear_l, ear_r)

                        # histéresis de cierre
                        if eye_closed:
                            eye_closed = 0 if ear >= EYE_THR_OPEN else 1
                        else:
                            eye_closed = 1 if ear < EYE_THR_CLOSE else 0

                # ---- Integración temporal por frame
                if eye_closed:
                    closed_run_ms += dt_ms
                    int_closed_ms += dt_ms
                else:
                    if eye_closed_prev and (BLINK_MIN_MS <= closed_run_ms <= BLINK_MAX_MS):
                        blinks_acc += 1
                        blink_event = 1
                        int_blinks += 1
                        events_win.append((ts, 'blink'))
                    closed_run_ms = 0.0
                eye_closed_prev = eye_closed

                if mar is not None and mar > YAWN_THR:
                    yawn_open_ms += dt_ms
                else:
                    if yawn_open_ms >= YAWN_MIN_MS:
                        yawns_acc += 1
                        yawn_event = 1
                        int_yawns += 1
                        events_win.append((ts, 'yawn'))
                    yawn_open_ms = 0.0

                # --- Ventanas deslizantes 60 s para tasas y PERCLOS ---
                # Purga eventos viejos (para tasas por minuto)
                limit_ev = ts - RATES_WINDOW_S
                while events_win and events_win[0][0] < limit_ev:
                    events_win.popleft()
                blinks_per_min = (sum(1 for t,e in events_win if e=='blink') * 60.0) / RATES_WINDOW_S
                yawns_per_min  = (sum(1 for t,e in events_win if e=='yawn')  * 60.0) / RATES_WINDOW_S

                # Ventana PERCLOS 60s (proporción de tiempo con ojo cerrado)
                closed_ms_this = dt_ms if eye_closed else 0.0
                perclos_win.append((ts, closed_ms_this, dt_ms))
                perclos_win_closed  += closed_ms_this
                perclos_win_elapsed += dt_ms
                limit_pc = ts - RATES_WINDOW_S
                while perclos_win and perclos_win[0][0] < limit_pc:
                    _, c_ms, e_ms = perclos_win.popleft()
                    perclos_win_closed  -= c_ms
                    perclos_win_elapsed -= e_ms
                perclos_prop_win = (perclos_win_closed / perclos_win_elapsed) if perclos_win_elapsed > 0 else 0.0
                perclos_pct_win  = perclos_prop_win * 100.0

                # ====== DECISIÓN DE ESTADO — SOLO EAR (con microsueño por racha) ======
                if closed_run_ms >= MICRO_MS:
                    estado = "MICROSUEÑO"
                else:
                    estado = estado_by_ear(ear)  # usa exclusivamente EAR_BANDS

                # CSV per-frame (exacto)
                def f(x, nd=4): return "" if x is None else f"{x:.{nd}f}"
                wframe.writerow([
                    idx, f"{ts:.3f}", f(ear), f(mar),
                    1 if eye_closed else 0, blink_event, yawn_event,
                    blinks_acc, yawns_acc, f"{blinks_per_min:.2f}", f"{yawns_per_min:.2f}",
                    f"{closed_run_ms:.1f}", estado
                ])

                # --- Acumular para el intervalo (CSV de 2s)
                int_elapsed_ms += dt_ms
                if ear is not None:
                    int_ear_sum += ear; int_ear_cnt += 1
                if mar is not None:
                    int_mar_sum += mar; int_mar_cnt += 1
                estados_intervalo.append(estado)

                # ¿cerramos intervalo exacto por frames?
                if ((idx + 1) % FRAMES_PER_INT) == 0:
                    perclos_prop_int = (int_closed_ms / max(1.0, int_elapsed_ms))      # 0–1
                    avg_ear = (int_ear_sum / int_ear_cnt) if int_ear_cnt > 0 else ""
                    avg_mar = (int_mar_sum / int_mar_cnt) if int_mar_cnt > 0 else ""

                    # moda del estado en el intervalo
                    if estados_intervalo:
                        mode_estado = max(set(estados_intervalo), key=estados_intervalo.count)
                    else:
                        mode_estado = "NORMAL"

                    start_2s = interval_idx * agg_interval_s
                    winter.writerow([
                        start_2s,
                        f"{perclos_prop_int:.3f}",
                        int_blinks,
                        int_yawns,
                        f"{avg_ear:.4f}" if avg_ear != "" else "",
                        f"{avg_mar:.4f}" if avg_mar != "" else "",
                        mode_estado
                    ])

                    # reset intervalo
                    interval_idx += 1
                    int_closed_ms = 0.0
                    int_elapsed_ms = 0.0
                    int_blinks = 0
                    int_yawns  = 0
                    int_ear_sum = 0.0; int_ear_cnt = 0
                    int_mar_sum = 0.0; int_mar_cnt = 0
                    estados_intervalo = []

                # ===== PERF por frame =====
                frame_ms = (time.perf_counter() - t_frame_start) * 1000.0
                frame_times.append(frame_ms)

                # muestreo CPU/RAM cada ~1s
                if psutil and (idx % int(max(1, round(fps_nom)))) == 0:
                    try:
                        cpu_samples.append(proc.cpu_percent(interval=None))  # %
                        rss_samples.append(proc.memory_info().rss / (1024*1024))  # MB
                    except Exception:
                        pass

                idx += 1
                pbar.update(1)

    finally:
        cap.release()
        detector.close()
        f_frame.close()
        f_int.close()

    # ===== Resumen de rendimiento =====
    t1 = time.perf_counter()
    elapsed_s = t1 - t0
    frames_done = len(frame_times)
    eff_fps = frames_done / elapsed_s if elapsed_s > 0 else 0.0

    if frame_times:
        avg_ms = sum(frame_times)/len(frame_times)
        med_ms = statistics.median(frame_times)
        p90_ms = np.percentile(frame_times, 90)
        max_ms = max(frame_times)
    else:
        avg_ms = med_ms = p90_ms = max_ms = 0.0

    cpu_pct_avg = float(np.mean(cpu_samples)) if psutil and cpu_samples else None
    rss_mb_avg  = float(np.mean(rss_samples)) if psutil and rss_samples else None
    cpu_temp_c  = read_cpu_temp_c()

    perf = {
        "video": os.path.basename(video_path),
        "frames_processed": frames_done,
        "wall_time_s": round(elapsed_s, 3),
        "effective_fps": round(eff_fps, 2),
        "frame_ms_avg": round(avg_ms, 2),
        "frame_ms_median": round(med_ms, 2),
        "frame_ms_p90": round(float(p90_ms), 2),
        "frame_ms_max": round(max_ms, 2),
        "cpu_percent_avg": round(cpu_pct_avg, 1) if cpu_pct_avg is not None else None,
        "rss_memory_mb_avg": round(rss_mb_avg, 1) if rss_mb_avg is not None else None,
        "cpu_temp_c": cpu_temp_c,
        "threads_used": threading.active_count(),
        "fps_nominal_from_file": round(fps_nom, 2),
        "width": W,
        "height": H
    }

    # Guardar JSON y CSV
    with open(out_perf_json, "w", encoding="utf-8") as jf:
        json.dump(perf, jf, ensure_ascii=False, indent=2)

    with open(out_perf_csv, "w", newline="", encoding="utf-8") as cf:
        cw = csv.writer(cf)
        cw.writerow(list(perf.keys()))
        cw.writerow(list(perf.values()))

    print(f"\n✅ Guardado:\n - {out_csv_frame}\n - {out_csv_int}\n - {out_perf_json}\n - {out_perf_csv}")

# -----------------------------
# BATCH
# -----------------------------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    videos = discover_videos(INPUT_DIR, EXTS, RECURSIVE)
    print(f"🔎 Encontrados {len(videos)} videos en {INPUT_DIR}")
    for i, vp in enumerate(videos, 1):
        try:
            print(f"\n({i}/{len(videos)}) {vp}")
            analyze_video(vp, OUT_DIR, agg_interval_s=AGG_INTERVAL_S)
        except Exception as e:
            print(f"⚠️ Error procesando {vp}: {e}")
