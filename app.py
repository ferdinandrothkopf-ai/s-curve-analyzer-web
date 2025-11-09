# app.py
import os
import base64
import tempfile
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from streamlit_drawable_canvas import st_canvas

# =================== Utils ===================
def load_first_frame(video_path, max_w=1280):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = frame.shape[:2]
    if w > max_w:
        s = max_w / w
        frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return frame

def center_of_box(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def seg_intersection(p1, p2, q1, q2):
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))

def alpha_blend(frames, alphas):
    out = np.zeros_like(frames[0], dtype=np.float32)
    for f, a in zip(frames, alphas):
        out += f.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)

def resize_to_width(frame, width):
    h, w = frame.shape[:2]
    if w == width:
        return frame
    s = width / w
    return cv2.resize(frame, (width, int(h * s)), interpolation=cv2.INTER_AREA)

def line_from_canvas(json_data):
    """Extrahiere eine Linie (zwei Punkte) aus dem Canvas-JSON (Canvas-Koords)."""
    if not json_data or "objects" not in json_data:
        return None
    for obj in json_data["objects"]:
        if obj.get("type") == "line":
            x1 = obj["x1"] + obj["left"]; y1 = obj["y1"] + obj["top"]
            x2 = obj["x2"] + obj["left"]; y2 = obj["y2"] + obj["top"]
            return ((x1, y1), (x2, y2))
        if obj.get("type") == "path" and obj.get("path"):
            (mx, my, *_), (lx, ly, *_) = obj["path"][0], obj["path"][-1]
            return ((mx, my), (lx, ly))
    return None

def scale_line_to_frame(line, sx, sy):
    (x1, y1), (x2, y2) = line
    return ((x1 * sx, y1 * sy), (x2 * sx, y2 * sy))

def pil_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"

def trim_clip(src_path, dst_path, t0, t1):
    from moviepy.editor import VideoFileClip
    with VideoFileClip(src_path) as clip:
        start = max(0.0, float(t0))
        end = max(start, float(t1))
        sub = clip.subclip(start, end)
        sub.write_videofile(
            dst_path, codec="libx264", audio=False, fps=clip.fps,
            verbose=False, logger=None
        )

def detect_times(video_path, entry_line, exit_line, model):
    """YOLOv8 + ByteTrack → Liste[(tid, t_in, t_out, dt)], fps, first_frame."""
    try:
        import lapx  # noqa: F401
    except Exception:
        raise ImportError("Tracker-Abhängigkeit fehlt (lap/lapx). Bitte 'lapx==0.5.6' installieren.")

    timings, last_centers = {}, {}
    cap0 = cv2.VideoCapture(video_path)
    fps = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    ok, first = cap0.read()
    cap0.release()
    if not ok:
        return [], fps, None
    width_ref = first.shape[1]

    stream = model.track(
        source=video_path, stream=True, tracker="bytetrack.yaml",
        classes=[2, 3, 5, 7], conf=0.25, verbose=False
    )
    frame_i = 0
    for r in stream:
        ow = r.orig_shape[1] if getattr(r, "orig_shape", None) else first.shape[1]
        scale = width_ref / float(ow)
        if r.boxes is not None and hasattr(r.boxes, "id") and r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy().astype(int)
            xyxys = r.boxes.xyxy.cpu().numpy()
            for tid, box in zip(ids, xyxys):
                cx, cy = center_of_box(box)
                cx, cy = cx * scale, cy * scale
                if tid in last_centers:
                    p1 = last_centers[tid]; p2 = (cx, cy)
                    if tid not in timings:
                        timings[tid] = {"entry": None, "exit": None}
                    if timings[tid]["entry"] is None and seg_intersection(p1, p2, entry_line[0], entry_line[1]):
                        timings[tid]["entry"] = frame_i / fps
                    if timings[tid]["exit"] is None and seg_intersection(p1, p2, exit_line[0], exit_line[1]):
                        timings[tid]["exit"] = frame_i / fps
                last_centers[tid] = (cx, cy)
        frame_i += 1

    valid = []
    for tid, t in timings.items():
        if t["entry"] is not None and t["exit"] is not None and t["exit"] > t["entry"]:
            valid.append((tid, t["entry"], t["exit"], t["exit"] - t["entry"]))
    valid.sort(key=lambda x: x[3])
    return valid, fps, first

# ===== Pillow Resampling kompatibel =====
try:
    RESAMPLE = Image.Resampling.BILINEAR
except Exception:
    RESAMPLE = Image.BILINEAR

# ===== Compat-Wrapper für st_canvas =====
def draw_canvas_with_bg(bg_pil: Image.Image, *, height: int, width: int, key: str, stroke_color: str):
    """Robuster Wrapper: erst mit background_image_url probieren, sonst Fallback auf PIL."""
    common = dict(
        background_color=None,
        height=height,
        width=width,
        drawing_mode="line",
        stroke_width=4,
        stroke_color=stroke_color,
        update_streamlit=True,
        display_toolbar=False,
        key=key,
    )
    # 1) Neuer Weg: Data-URL (funktioniert mit neueren Streamlit-Versionen)
    try:
        return st_canvas(background_image=None, background_image_url=pil_to_data_url(bg_pil, "PNG"), **common)
    except TypeError:
        # 2) Älterer Weg: direktes PIL-Image (Lib ruft intern image_to_url)
        return st_canvas(background_image=bg_pil, **common)

# =================== UI ===================
st.set_page_config(page_title="S-Curve Analyzer (Web)", layout="wide")
st.title("🏎️ S-Curve Analyzer – Linien **ziehen** auf dem Vorschaubild")

st.markdown(
    "1) Lade **1–3** Clips der gleichen S-Kurve hoch.  \n"
    "2) Wähle den **Referenz-Clip**.  \n"
    "3) Zeichne **Einfahrt** (links) und **Ausfahrt** (rechts) **als Linie**.  \n"
    "4) **Analysieren** → YOLO-Tracking, Sektorzeiten, Auto-Trim & Overlay."
)

# Session State
if "tmp_paths" not in st.session_state: st.session_state.tmp_paths = []
if "names"     not in st.session_state: st.session_state.names = []
if "ref_idx"   not in st.session_state: st.session_state.ref_idx = 0

uploaded = st.file_uploader("Clips (MP4/MOV)", type=["mp4", "mov", "m4v"], accept_multiple_files=True)
alpha_top = st.slider("Deckkraft obere Ebenen", 0.3, 0.8, 0.5, 0.05)
out_width = st.select_slider("Exportbreite", options=[854, 960, 1280, 1600, 1920], value=1280)

# Upload → Tempfiles
if uploaded:
    for p in st.session_state.tmp_paths:
        try: os.remove(p)
        except Exception: pass
    st.session_state.tmp_paths, st.session_state.names = [], []
    for uf in uploaded[:3]:
        suffix = os.path.splitext(uf.name)[1].lower()
        t = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        t.write(uf.read()); t.flush(); t.close()
        st.session_state.tmp_paths.append(t.name)
        st.session_state.names.append(uf.name)
    st.session_state.ref_idx = 0

paths = st.session_state.tmp_paths
names = st.session_state.names

if paths:
    st.caption("Wähle den Clip, dessen erstes Frame als Zeichengrundlage dient.")
    st.session_state.ref_idx = st.selectbox(
        "Referenz-Clip", options=list(range(len(paths))),
        index=st.session_state.ref_idx, format_func=lambda i: names[i]
    )

    # 1. Frame des Referenz-Clips laden
    first_bgr = load_first_frame(paths[st.session_state.ref_idx], max_w=1280)
    if first_bgr is None:
        st.error("Konnte ersten Frame nicht laden."); st.stop()

    # Vorschau anzeigen (RGB)
    st.markdown("### Vorschau-Frame")
    preview = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
    st.image(preview, caption="Referenz-Frame", width=min(960, preview.shape[1]))

    # Canvas-Hintergrund (PIL) + Größe
    first_rgb = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
    bg_img = Image.fromarray(first_rgb).convert("RGB")
    canvas_w = min(640, bg_img.width)
    canvas_h = int(bg_img.height * canvas_w / bg_img.width)
    bg_canvas = bg_img.resize((canvas_w, canvas_h), RESAMPLE)

    # Skalierung Canvas → Originalframe
    sx = first_bgr.shape[1] / float(canvas_w)
    sy = first_bgr.shape[0] / float(canvas_h)

    st.subheader("Sektorlinien zeichnen")
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("**Einfahrt-Linie**")
        entry_canvas = draw_canvas_with_bg(
            bg_canvas, height=canvas_h, width=canvas_w,
            key=f"entry_canvas_{st.session_state.ref_idx}",
            stroke_color="#00ff00",
        )

    with c2:
        st.markdown("**Ausfahrt-Linie**")
        exit_canvas = draw_canvas_with_bg(
            bg_canvas, height=canvas_h, width=canvas_w,
            key=f"exit_canvas_{st.session_state.ref_idx}_exit",
            stroke_color="#ff0000",
        )

    # Linien extrahieren & auf Framegröße mappen
    entry_line_canvas = line_from_canvas(getattr(entry_canvas, "json_data", None))
    exit_line_canvas  = line_from_canvas(getattr(exit_canvas,  "json_data", None))
    entry_line = scale_line_to_frame(entry_line_canvas, sx, sy) if entry_line_canvas else None
    exit_line  = scale_line_to_frame(exit_line_canvas,  sx, sy) if exit_line_canvas else None

    ready = entry_line is not None and exit_line is not None
    if not ready:
        st.info("⚠️ Bitte in **beiden** Canvases je **eine Linie** ziehen (Start ↔ Ende).")

    # ------------------- Analyse -------------------
    if st.button("Analysieren", type="primary", disabled=not ready):
        model = YOLO("yolov8n.pt")
        results_table, trimmed = [], []

        try:
            with st.spinner("Analysiere Clips …"):
                for idx, vpath in enumerate(paths):
                    valid, fps, _ = detect_times(vpath, entry_line, exit_line, model)
                    if not valid:
                        results_table.append(
                            {"Clip": os.path.basename(vpath),
                             "VehicleID": "-", "Zeit": "-", "t_in": "-", "t_out": "-"}
                        )
                        continue
                    best_id, t_in, t_out, dt = valid[0]
                    results_table.append(
                        {"Clip": os.path.basename(vpath),
                         "VehicleID": int(best_id),
                         "Zeit": f"{dt:.3f}s",
                         "t_in": f"{t_in:.3f}",
                         "t_out": f"{t_out:.3f}"}
                    )
                    pad = 0.15
                    outp = os.path.join(tempfile.gettempdir(), f"trim_{idx}.mp4")
                    trim_clip(vpath, outp, t_in - pad, t_out + pad)
                    trimmed.append((outp, best_id, dt))
        except ImportError as e:
            st.error(str(e)); st.stop()

        st.subheader("Sektorzeiten (schnellstes Fahrzeug pro Clip)")
        st.dataframe(results_table, use_container_width=True)

        if not trimmed:
            st.warning("Keine gültigen Sektorzeiten erkannt."); st.stop()

        # ------------------- Overlay -------------------
        st.markdown("### Overlay erstellen (bis 3 Clips)")
        names_pretty = [f"{os.path.basename(p)} (ID {int(vid)}, {d:.3f}s)" for (p, vid, d) in trimmed]
        sel_idx = st.multiselect(
            "Wähle Clips",
            options=list(range(len(trimmed))),
            format_func=lambda i: names_pretty[i],
            default=list(range(min(3, len(trimmed))))
        )

        if sel_idx:
            chosen = [trimmed[i] for i in sel_idx][:3]
            caps = [cv2.VideoCapture(p) for (p, _, _) in chosen]
            fps_list = [max(1.0, c.get(cv2.CAP_PROP_FPS) or 30.0) for c in caps]
            out_fps = min(fps_list)

            ok, ref = caps[0].read()
            if not ok:
                for c in caps: c.release()
                st.error("Fehler beim Overlay."); st.stop()
            ref = resize_to_width(ref, out_width)
            h, w = ref.shape[:2]

            if len(chosen) == 1: alphas = [1.0]
            elif len(chosen) == 2: alphas = [1.0, alpha_top]
            else: alphas = [1.0, alpha_top, alpha_top]

            out_path = os.path.join(tempfile.gettempdir(), "overlay.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, out_fps, (w, h))

            init_frames = [ref]
            for i in range(1, len(caps)):
                ok, fr = caps[i].read()
                fr = np.zeros_like(ref) if not ok else resize_to_width(fr, out_width)
                init_frames.append(fr)
            writer.write(alpha_blend(init_frames, alphas[: len(init_frames)]))

            while True:
                frames, ended = [], 0
                for c in caps:
                    ok, fr = c.read()
                    if not ok:
                        ended += 1; continue
                    frames.append(resize_to_width(fr, out_width))
                if ended == len(caps):
                    break
                while len(frames) < len(caps):
                    frames.append(np.zeros((h, w, 3), dtype=np.uint8))
                writer.write(alpha_blend(frames, alphas[: len(frames)]))

            for c in caps: c.release()
            writer.release()

            st.success("Fertig! Overlay & Zeiten erzeugt.")
            st.video(out_path)
            with open(out_path, "rb") as f:
                st.download_button(
                    "Overlay-Video herunterladen",
                    data=f.read(),
                    file_name="overlay.mp4",
                    mime="video/mp4"
                )
