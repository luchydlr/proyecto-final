import os, glob, csv, math, time, collections, statistics, shutil, subprocess, json
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

try:
    import psutil
except Exception:
    psutil = None

# ========= RUTAS / CONFIG =========
MODEL_PATH   = "/home/lucianadelarosa/Desktop/proyecto-final/facemesh/face_landmarker.task"
INPUT_DIR    = "/home/lucianadelarosa/Desktop/NTHU/030"
OUT_DIR      = "/home/lucianadelarosa/Desktop/proyecto-final/val-annot/code_solo/NTHU/030"
EXTS         = (".mp4", ".mov", ".avi", ".mkv", ".MOV")
RECURSIVE    = False
SKIP_IF_EXISTS = False

# ========= UMBRALES =========
EAR_THR = 0.21                  # ojo cerrado si EAR < EAR_THR
MIN_BLINK_CLOSED_FRAMES = 2     # frames cerrados mínimos para contar parpadeo
YAWN_MAR_THR = 0.65             # umbral MAR para bostezo
YAWN_HOLD_FRAMES = 5            # frames consecutivos por encima para activar yawn
MICRO_SECONDS = 1.5             # duración para MICROSUEÑO (segundos)
RATES_WINDOW_S = 60.0           # ventana para tasas por minuto

# Bandas por EAR (idéntico a Colab)
EAR_BANDS = {
    "NORMAL":      (0.25,  10.0),  # 10.0 para cubrir valores altos
    "FATIGA":      (0.21,  0.25),
    "SOMNOLENCIA": (0.00,  0.21),
}

# Ranking de severidad
RANK = {"NORMAL": 0, "FATIGA": 1, "SOMNOLENCIA": 2, "MICROSUEÑO": 3}

# ========= LANDMARKS (MediaPipe FaceMesh) =========
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
MOUTH_VERT_PAIRS = [(13,14), (82,87), (312,317)]
MOUTH_HORZ = (78, 308)

# ========= UTILIDADES =========
def hypot2(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)

# EAR idéntico a Colab (por ojo)
def ear_one_eye(lms, idxs, w, h):
    try:
        (x1,y1) = (lms[idxs[0]].x * w, lms[idxs[0]].y * h)
        (x2,y2) = (lms[idxs[1]].x * w, lms[idxs[1]].y * h)
        (x3,y3) = (lms[idxs[2]].x * w, lms[idxs[2]].y * h)
        (x4,y4) = (lms[idxs[3]].x * w, lms[idxs[3]].y * h)
        (x5,y5) = (lms[idxs[4]].x * w, lms[idxs[4]].y * h)
        (x6,y6) = (lms[idxs[5]].x * w, lms[idxs[5]].y * h)
        v1 = hypot2(x2, y2, x6, y6)
        v2 = hypot2(x3, y3, x5, y5)
        h1 = hypot2(x1, y1, x4, y4)
        if h1 == 0:
            return None
        return (v1 + v2) / (2.0 * h1)
    except Exception:
        return None

def avg_ear(lms, w, h):
    er = ear_one_eye(lms, RIGHT_EYE_IDX, w, h)
    el = ear_one_eye(lms, LEFT_EYE_IDX,  w, h)
    if er is None or el is None:
        return None
    return (er + el) / 2.0

# MAR idéntico a Colab
def mar_value(lms, w, h):
    try:
        left  = (lms[MOUTH_HORZ[0]].x * w, lms[MOUTH_HORZ[0]].y * h)
        right = (lms[MOUTH_HORZ[1]].x * w, lms[MOUTH_HORZ[1]].y * h)
        mouth_w = hypot2(left[0], left[1], right[0], right[1])
        if mouth_w == 0:
            return None
        v_dists = []
        for a, b in MOUTH_VERT_PAIRS:
            A = (lms[a].x * w, lms[a].y * h)
            B = (lms[b].x * w, lms[b].y * h)
            v_dists.append(hypot2(A[0], A[1], B[0], B[1]))
        mar = (sum(v_dists) / len(v_dists)) / mouth_w
        return mar
    except Exception:
        return None

def discover_videos(folder: str, exts=EXTS, recursive=False):
    pattern = "**/*" if recursive else "*"
    paths = [p for p in glob.glob(os.path.join(folder, pattern), recursive=recursive)
             if os.path.splitext(p)[1].lower() in tuple(e.lower() for e in exts)]
    return sorted(paths)

def safe_open_csv(path, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, 'w', newline='', encoding='utf-8')
    w = csv.writer(f)
    w.writerow(header)
    return f, w

# ========= Health check MediaPipe =========
def rpi_vcgencmd(args):
    try:
        if shutil.which("vcgencmd"):
            out = subprocess.check_output(["vcgencmd"] + args, timeout=1.0).decode("utf-8").strip()
            return out
    except Exception:
        pass
    return None

def read_cpu_temp_c():
    """Raspberry Pi: vcgencmd measure_temp → float en °C, o None."""
    out = rpi_vcgencmd(["measure_temp"])
    if not out:
        return None
    try:
        # ejemplo: 'temp=48.2'C'
        val = out.split("=")[1].split("'")[0]
        return float(val)
    except Exception:
        return None

def read_rpi_gpu_info():
    """Devuelve (gpu_clock_hz, gpu_mem_mb, arm_mem_mb) si vcgencmd está; de lo contrario None."""
    clk = rpi_vcgencmd(["measure_clock", "v3d"])  # GPU 3D clock
    mem_gpu = rpi_vcgencmd(["get_mem", "gpu"])
    mem_arm = rpi_vcgencmd(["get_mem", "arm"])
    def parse_clock(s):
        try:
            # 'frequency(46)=500000000' → 500000000
            return int(s.split("=")[1])
        except Exception:
            return None
    def parse_mem_mb(s):
        try:
            # 'gpu=76M' / 'arm=948M' → 76 / 948
            return int(''.join(ch for ch in s if ch.isdigit()))
        except Exception:
            return None
    return (
        parse_clock(clk) if clk else None,
        parse_mem_mb(mem_gpu) if mem_gpu else None,
        parse_mem_mb(mem_arm) if mem_arm else None,
    )

# ========= MediaPipe Tasks Detector =========
def build_detector(model_path: str, num_threads: int = 4):
    try:
        base_options = mp_python.BaseOptions(model_asset_path=model_path, num_threads=num_threads)
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

# ========= Clasificadores por variable =========
def estado_by_ear(ear):
    if ear is None:
        return "NORMAL"
    for nombre, (lo, hi) in EAR_BANDS.items():
        if lo <= ear < hi:
            return nombre
    return "NORMAL"

def estado_by_blinks(blinks_pm: float):
    if blinks_pm is None:
        return "NORMAL"
    if 17 <= blinks_pm <= 25:
        return "NORMAL"
    if 12 <= blinks_pm <= 16:
        return "FATIGA"
    if 6  <= blinks_pm <= 12:
        return "SOMNOLENCIA"
    if blinks_pm < 6:
        return "SOMNOLENCIA"
    # >25: a menudo conversación/ruido; mantenemos NORMAL (ajusta si prefieres)
    return "NORMAL"

def estado_by_yawns(yawns_pm: float):
    if yawns_pm is None:
        return "NORMAL"
    if yawns_pm <= 1:
        return "NORMAL"
    if 1 < yawns_pm <= 4:
        return "FATIGA"
    return "SOMNOLENCIA"

# ========= Decisor (EAR principal; secundarios suben severidad por consenso) =========
def decidir_estado(ear, closed_run_frames, micro_frames, blinks_pm, yawns_pm):
    # 0) prioridad absoluta a microsueño por racha
    if closed_run_frames >= micro_frames:
        return "MICROSUEÑO"

    # 1) estado base: SOLO EAR
    base = estado_by_ear(ear)
    base_rank = RANK[base]

    # 2) secundarios (solo para reforzar si AMBOS coinciden en algo > base)
    s_b = estado_by_blinks(blinks_pm)
    s_y = estado_by_yawns(yawns_pm)
    r_b, r_y = RANK[s_b], RANK[s_y]
    both_min = min(r_b, r_y)

    if both_min > base_rank:
        # subir exactamente al nivel mínimo común de los dos secundarios (consenso)
        for nombre, rank in RANK.items():
            if rank == both_min:
                return nombre

    # 3) si no hay consenso superior, se respeta EAR (no se degrada)
    return base

# ========= Procesamiento de un video =========
def process_video_csv_colab(video_path: str, out_dir: str):
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_csv = os.path.join(out_dir, f"{base}_val.csv")
    out_health = os.path.join(out_dir, f"{base}_health.json")
    if SKIP_IF_EXISTS and os.path.exists(out_csv):
        print(f"⏭️  Ya existe: {out_csv}")
        return

    os.makedirs(out_dir, exist_ok=True)
    detector = build_detector(MODEL_PATH, num_threads=4)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ No se pudo abrir: {video_path}")
        detector.close()
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 640)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    MICRO_FRAMES = int(round(MICRO_SECONDS * fps))

    f, wr = safe_open_csv(out_csv, [
        "Frame_ID","time_s","EAR","eye_closed","blink_count","MAR","yawn_active","yawn_count",
        "estado","blinks_per_min","yawns_per_min"
    ])

    # estados por frames
    closed_run = 0
    blink_count = 0
    yawn_run = 0
    yawn_active = False
    yawn_count = 0

    # ventana deslizante para tasas/min (timestamp en segundos)
    events_win = collections.deque()  # (ts, 'blink'|'yawn')
    
    # -------- Health / performance trackers --------
    t0 = time.perf_counter()
    frame_times_ms = []

    # psutil sampling (opcional)
    if psutil:
        proc = psutil.Process(os.getpid())
        _ = proc.cpu_percent(interval=None)  # prime
        cpu_samples = []
        rss_samples = []
    else:
        proc = None
        cpu_samples = []
        rss_samples = []

    # tomar una lectura GPU/mem al inicio (mejor esfuerzo RPi)
    gpu_clock_hz, gpu_mem_mb, arm_mem_mb = read_rpi_gpu_info()

    frame_idx = 0
    print(f"🎥 {os.path.basename(video_path)} | FPS={fps:.2f} | {W}x{H} | micro_frames={MICRO_FRAMES}")

    try:
        while True:
            t_frame_start = time.perf_counter()
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            res = detector.detect(mp_img)

            ear = None
            mar = None
            eye_closed_flag = 0

            if res.face_landmarks:
                lms = res.face_landmarks[0]
                ear = avg_ear(lms, W, H)   # PROMEDIO (idéntico a Colab)
                mar = mar_value(lms, W, H) # MAR (idéntico a Colab)

            # timestamp preferido desde el contenedor; si no, derivado por FPS
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            ts = (pos_msec / 1000.0) if (pos_msec and pos_msec > 0) else (frame_idx / fps)

            # --- BLINKS: racha de frames cerrados
            if ear is not None and ear < EAR_THR:
                eye_closed_flag = 1
                closed_run += 1
            else:
                if closed_run >= MIN_BLINK_CLOSED_FRAMES:
                    blink_count += 1
                    events_win.append((ts, 'blink'))
                closed_run = 0
                eye_closed_flag = 0

            # --- YAWNS: hold de MAR
            if mar is not None and mar > YAWN_MAR_THR:
                yawn_run += 1
                if not yawn_active and yawn_run >= YAWN_HOLD_FRAMES:
                    yawn_active = True
            else:
                if yawn_active:
                    yawn_count += 1
                    events_win.append((ts, 'yawn'))
                yawn_active = False
                yawn_run = 0

            # --- Tasas por minuto (ventana 60s)
            limit_ev = ts - RATES_WINDOW_S
            while events_win and events_win[0][0] < limit_ev:
                events_win.popleft()
            blinks_pm = (sum(1 for t,e in events_win if e=='blink') * 60.0) / RATES_WINDOW_S
            yawns_pm  = (sum(1 for t,e in events_win if e=='yawn')  * 60.0) / RATES_WINDOW_S

            # --- Estado final (EAR principal; secundarios refuerzan por consenso)
            estado = decidir_estado(ear, closed_run, MICRO_FRAMES, blinks_pm, yawns_pm)

            # --- Escribir CSV
            wr.writerow([
                frame_idx,
                f"{ts:.3f}",
                "" if ear is None else f"{ear:.5f}",
                eye_closed_flag,
                blink_count,
                "" if mar is None else f"{mar:.5f}",
                1 if yawn_active else 0,
                yawn_count,
                estado,
                f"{blinks_pm:.2f}",
                f"{yawns_pm:.2f}"
            ])
            
            # ====== Health sampling ======
            frame_ms = (time.perf_counter() - t_frame_start) * 1000.0
            frame_times_ms.append(frame_ms)
            if psutil and (frame_idx % int(max(1, round(fps)))) == 0:
                try:
                    cpu_samples.append(proc.cpu_percent(interval=None))               # %
                    rss_samples.append(proc.memory_info().rss / (1024*1024))          # MB
                except Exception:
                    pass

            frame_idx += 1

    finally:
        cap.release()
        detector.close()
        f.close()

    # --------- Build health JSON ---------
    t1 = time.perf_counter()
    elapsed_s = t1 - t0
    frames_done = len(frame_times_ms)
    eff_fps = (frames_done / elapsed_s) if elapsed_s > 0 else 0.0

    if frame_times_ms:
        avg_ms = sum(frame_times_ms)/frames_done
        med_ms = statistics.median(frame_times_ms)
        p90_ms = float(np_percentile(frame_times_ms, 90.0))
        max_ms = max(frame_times_ms)
    else:
        avg_ms = med_ms = p90_ms = max_ms = 0.0

    cpu_pct_avg = float(sum(cpu_samples)/len(cpu_samples)) if cpu_samples else None
    rss_mb_avg  = float(sum(rss_samples)/len(rss_samples)) if rss_samples else None
    cpu_temp    = read_cpu_temp_c()

    health = {
        "video": os.path.basename(video_path),
        "frames_processed": frames_done,
        "wall_time_s": round(elapsed_s, 3),
        "effective_fps": round(eff_fps, 2),
        "frame_ms_avg": round(avg_ms, 2),
        "frame_ms_median": round(med_ms, 2),
        "frame_ms_p90": round(p90_ms, 2),
        "frame_ms_max": round(max_ms, 2),
        "cpu_percent_avg": round(cpu_pct_avg, 1) if cpu_pct_avg is not None else None,
        "rss_memory_mb_avg": round(rss_mb_avg, 1) if rss_mb_avg is not None else None,
        "cpu_temp_c": cpu_temp,
        # Raspberry Pi extras (mejor esfuerzo, pueden ser None):
        "rpi_gpu_clock_hz": gpu_clock_hz,
        "rpi_gpu_mem_mb": gpu_mem_mb,
        "rpi_arm_mem_mb": arm_mem_mb,
        # meta del archivo fuente
        "fps_nominal_from_file": round(fps, 2),
        "width": W,
        "height": H,
    }

    with open(out_health, "w", encoding="utf-8") as jf:
        json.dump(health, jf, ensure_ascii=False, indent=2)

    print(f"✅ CSV guardado: {out_csv}")
    print(f"🩺 Health JSON: {out_health}")

# ===== util pequeño (evita importar numpy solo por un percentil) =====
def np_percentile(seq, q):
    # q en [0,100]; implementación simple para no depender de numpy en RPi mínima
    if not seq:
        return 0.0
    data = sorted(seq)
    k = (len(data)-1) * (q/100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return float(data[int(k)])
    d0 = data[f] * (c-k)
    d1 = data[c] * (k-f)
    return float(d0+d1)

# ========= BATCH =========
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    videos = discover_videos(INPUT_DIR, EXTS, RECURSIVE)
    print(f"🔎 Encontrados {len(videos)} videos en {INPUT_DIR}")
    for i, vp in enumerate(videos, 1):
        try:
            print(f"\n({i}/{len(videos)}) {vp}")
            process_video_csv_colab(vp, OUT_DIR)
        except Exception as e:
            print(f"⚠️ Error procesando {vp}: {e}")

