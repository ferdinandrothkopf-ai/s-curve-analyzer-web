# app.py
import os
import base64
from io import BytesIO
import tempfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from streamlit_drawable_canvas import st_canvas

# --- Robust-Shim für alte drawable-canvas Versionen ---
# Stellt st_image.image_to_url bereit (Signatur-kompatibel).
try:
    from streamlit.elements import image as _st_image
    if not hasattr(_st_image, "image_to_url"):
        import base64
        from io import BytesIO
        from PIL import Image as _PILImage
        import numpy as _np

        def image_to_url(*args, **kwargs):
            img = args[0] if args else kwargs.get("image")
            if isinstance(img, _np.ndarray):
                if img.ndim == 3 and img.shape[2] == 3:
                    img = _PILImage.fromarray(img.astype("uint8"), mode="RGB")
                else:
                    img = _PILImage.fromarray(img.astype("uint8"))
            elif not isinstance(img, _PILImage.Image):
                img = _PILImage.open(img)

            buf = BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            url = f"data:image/png;base64,{b64}"
            dims = {"width": getattr(img, "width", None), "height": getattr(img, "height", None)}
            return url, dims

        _st_image.image_to_url = image_to_url
except Exception:
    pass

# ========================= Utils =========================

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

def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def center_of_box(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

def seg_intersection(p1, p2, q1, q2):
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))

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

try:
    RESAMPLE = Image.Resampling.BILINEAR
except Exception:
    RESAMPLE = Image.BILINEAR

# ===== Kompatibilitäts-Wrapper für st_canvas =====
import inspect
def draw_canvas(bg_pil, *, height, width, key, stroke_color="#00ff00"):
    """Kompatibel mit alten & neuen drawable-canvas-Versionen."""
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
    params = inspect.signature(st_canvas).parameters
    if "background_image_url" in params:  # neuere Lib
        import base64
        from io import BytesIO
        buf = BytesIO(); bg_pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        url = f"data:image/png;base64,{b64}"
        return st_canvas(background_image=None, background_image_url=url, **common)
    else:  # ältere Lib
        return st_canvas(background_image=bg_pil, **common)

# ========================= Simple IoU Tracker =========================
class SimpleTracker:
    def __init__(self, iou_thresh=0.3, max_age=30):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.next_id = 1
        self.tracks = {}

    @staticmethod
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
        inter = iw * ih
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter + 1e-6)

    def update(self, detections):
        assigned = set()
        for tid, t in list(self.tracks.items()):
            best_iou, best_j = 0.0, -1
            for j, d in enumerate(detections):
                if j in assigned: continue
                i = self.iou(t["box"], d)
                if i > best_iou:
                    best_iou, best_j = i, j
            if best_iou >= self.iou_thresh:
                self.tracks[tid]["box"] = detections[best_j]
                self.tracks[tid]["age"] = 0
                assigned.add(best_j)
            else:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]

        for j, d in enumerate(detections):
            if j not in assigned:
                self.tracks[self.next_id] = {"box": d, "age": 0, "last_center": None}
                self.next_id += 1

        return [(tid, t["box"]) for tid, t in self.tracks.items()]

# ========================= App =========================
st.set_page_config(page_title="Sektorzeiten (einfach)", layout="wide")
st.title("🏁 Sektorzeiten aus Video")

st.markdown(
    "1) **Video hochladen** → 2) Im **ersten Frame** **Einfahrts-** und **Ausfahrtslinie** zeichnen → "
    "3) **Analysieren** → **Tabelle mit Zeiten pro Fahrzeug**."
)

uploaded = st.file_uploader("Video (MP4/MOV)", type=["mp4", "mov", "m4v"], accept_multiple_files=False)

if uploaded:
    suffix = os.path.splitext(uploaded.name)[1].lower()
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tfile.write(uploaded.read()); tfile.flush(); tfile.close()
    video_path = tfile.name

    first_bgr = load_first_frame(video_path, max_w=1280)
    if first_bgr is None:
        st.error("Konnte ersten Frame nicht laden."); st.stop()

    st.image(to_rgb(first_bgr), caption="Erster Frame", width=min(960, first_bgr.shape[1]))

    first_rgb = to_rgb(first_bgr)
    bg_img = Image.fromarray(first_rgb).convert("RGB")
    canvas_w = min(640, bg_img.width)
    canvas_h = int(bg_img.height * canvas_w / bg_img.width)
    bg_canvas = bg_img.resize((canvas_w, canvas_h), RESAMPLE)

    sx = first_bgr.shape[1] / float(canvas_w)
    sy = first_bgr.shape[0] / float(canvas_h)

    st.subheader("Sektorlinien zeichnen")
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("**Einfahrts-Linie**")
        entry_canvas = draw_canvas(bg_canvas, height=canvas_h, width=canvas_w, key="entry_canvas", stroke_color="#00ff00")

    with c2:
        st.markdown("**Ausfahrts-Linie**")
        exit_canvas = draw_canvas(bg_canvas, height=canvas_h, width=canvas_w, key="exit_canvas", stroke_color="#ff0000")

    entry_line_c = line_from_canvas(getattr(entry_canvas, "json_data", None))
    exit_line_c = line_from_canvas(getattr(exit_canvas, "json_data", None))
    entry_line = scale_line_to_frame(entry_line_c, sx, sy) if entry_line_c else None
    exit_line = scale_line_to_frame(exit_line_c, sx, sy) if exit_line_c else None

    ready = entry_line is not None and exit_line is not None
    if not ready:
        st.info("Bitte **beide** Linien ziehen (Start & Ende).")

    if st.button("Analysieren", type="primary", disabled=not ready):
        st.write("🔎 Erkenne & tracke Fahrzeuge …")
        model = YOLO("yolov8n.pt")
        tracker = SimpleTracker(iou_thresh=0.3, max_age=20)

        timings, last_centers = {}, {}
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_i = 0

        while True:
            ok, frame = cap.read()
            if not ok: break
            yres = model.predict(frame, classes=[2, 3, 5, 7], conf=0.25, imgsz=960, verbose=False)
            boxes = []
            if yres and len(yres) > 0 and yres[0].boxes is not None:
                xyxys = yres[0].boxes.xyxy.cpu().numpy()
                for b in xyxys:
                    boxes.append(tuple(b.tolist()))

            tracks = tracker.update(boxes)

            for tid, box in tracks:
                cx, cy = center_of_box(box)
                p2 = (cx, cy)
                p1 = last_centers.get(tid, p2)
                if tid not in timings:
                    timings[tid] = {"entry": None, "exit": None}

                if timings[tid]["entry"] is None and seg_intersection(p1, p2, entry_line[0], entry_line[1]):
                    timings[tid]["entry"] = frame_i / fps
                if timings[tid]["exit"] is None and seg_intersection(p1, p2, exit_line[0], exit_line[1]):
                    timings[tid]["exit"] = frame_i / fps

                last_centers[tid] = p2
            frame_i += 1
        cap.release()

        rows = []
        for tid, t in timings.items():
            if t["entry"] and t["exit"] and t["exit"] > t["entry"]:
                rows.append({
                    "Fahrzeug-ID": int(tid),
                    "t_in [s]": round(t["entry"], 3),
                    "t_out [s]": round(t["exit"], 3),
                    "Sektorzeit [s]": round(t["exit"] - t["entry"], 3),
                })

        if not rows:
            st.warning("Keine gültigen Sektorzeiten erkannt."); st.stop()

        rows = sorted(rows, key=lambda r: r["Sektorzeit [s]"])
        st.success(f"Fertig! {len(rows)} Fahrzeuge erkannt.")
        st.dataframe(rows, use_container_width=True)
