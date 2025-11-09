import os, io, base64, tempfile, time
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from moviepy.editor import VideoFileClip

import plotly.graph_objs as go
from streamlit_plotly_events import plotly_events

# =================== Utils ===================
def load_first_frame(video_path, max_w=960):
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

def line_from_two_points(p, q):
    # p=(x,y), q=(x,y)
    return (tuple(p), tuple(q))

def trim_clip(src_path, dst_path, t0, t1):
    with VideoFileClip(src_path) as clip:
        sub = clip.subclip(max(t0, 0), max(t1, 0))
        sub.write_videofile(
            dst_path, codec="libx264", audio=False, fps=clip.fps,
            verbose=False, logger=None
        )

def detect_times(video_path, entry_line, exit_line, model):
    timings, last_centers = {}, {}
    cap0 = cv2.VideoCapture(video_path)
    fps = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    ok, first = cap0.read()
    cap0.release()
    if not ok:
        return [], fps, first
    width_ref = first.shape[1]

    stream = model.track(
        source=video_path, stream=True, tracker="bytetrack.yaml",
        classes=[2, 3, 5, 7], conf=0.25, verbose=False
    )
    frame_i = 0
    for r in stream:
        if r.orig_shape is not None:
            oh, ow = r.orig_shape
        else:
            oh, ow = first.shape[:2]
        scale = width_ref / ow
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

# ---------- Plotly Helpers (Bild als Hintergrund + Klicks erfassen) ----------
def pil_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return "data:image/png;base64," + b64

def click_two_points_on_image(img_bgr, title: str, key: str):
    """
    Show the image as a Plotly Image trace (clickable) and return exactly two (x,y) clicks.
    - Uses dragmode='pan' so dragging doesn't create a zoom box.
    - Hides zoom/select/lasso tools in the modebar.
    - Returns (p1, p2) as tuples in pixel coords (x to the right, y down).
    """
    # Convert to RGB for display
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # 1) Use an Image trace (click events fire on it)
    fig = go.Figure(data=[go.Image(z=img_rgb)])

    # 2) Axes & layout
    fig.update_xaxes(
        visible=False,
        range=[0, w]
    )
    fig.update_yaxes(
        visible=False,
        range=[h, 0]
    )
    fig.update_layout(
        dragmode="pan",
        clickmode="event+select",
        margin=dict(l=0, r=0, t=30, b=0),
        width=min(900, int(w * 0.95)),
        height=int(min(900, int(w * 0.95)) * (h / w)),
        title=title
    )

    # 3) Clean modebar
    config = {
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "zoom2d", "select2d", "lasso2d", "autoscale", "toggleSpikelines",
            "zoomIn2d", "zoomOut2d", "resetScale2d"
        ],
        "scrollZoom": False
    }

    st.caption("❗ Klicke **genau 2 Punkte** (Start und Ende der Linie).")
        events = plotly_events(
        fig,
        click_event=True,
        hover_event=False,
        select_event=False,
        key=key
    )

    # Extract first two clicks (x,y)
    pts = [(float(e["x"]), float(e["y"])) for e in events if ("x" in e and "y" in e)]
    if len(pts) >= 2:
        return pts[0], pts[1]
    return None, None

# =================== UI ===================
st.set_page_config(page_title="S-Curve Analyzer (Web)", layout="wide")
st.title("🏎️ S-Curve Analyzer – komplett im Browser (Plotly-Klicks)")

st.markdown(
    "1. Lade **1–3 Stativ-Clips** der S-Kurve hoch.  \n"
    "2. Wähle den **Referenz-Clip**.  \n"
    "3. Klicke auf dem Vorschau-Frame je **2 Punkte** für **Einfahrt** und **Ausfahrt**.  \n"
    "4. **Analysieren** → YOLO-Tracking, Sektorzeiten, Auto-Trim & Overlay."
)

# Session State
if "tmp_paths" not in st.session_state: st.session_state.tmp_paths = []
if "names"     not in st.session_state: st.session_state.names = []
if "ref_idx"   not in st.session_state: st.session_state.ref_idx = 0
if "entry_line" not in st.session_state: st.session_state.entry_line = None
if "exit_line"  not in st.session_state: st.session_state.exit_line  = None

uploaded = st.file_uploader(
    "Clips (MP4/MOV)", type=["mp4", "mov", "m4v"], accept_multiple_files=True
)
alpha_top = st.slider("Deckkraft obere Ebenen", 0.3, 0.8, 0.5, 0.05)
out_width = st.select_slider("Exportbreite", options=[854, 960, 1280, 1600, 1920], value=1280)

# Upload → Tempfiles erzeugen
if uploaded:
    for p in st.session_state.tmp_paths:
        try: os.remove(p)
        except Exception: pass
    st.session_state.tmp_paths, st.session_state.names = [], []
    for uf in uploaded[:3]:
        suf = os.path.splitext(uf.name)[1].lower()
        t = tempfile.NamedTemporaryFile(delete=False, suffix=suf)
        t.write(uf.read()); t.flush(); t.close()
        st.session_state.tmp_paths.append(t.name)
        st.session_state.names.append(uf.name)
    st.session_state.ref_idx = 0
    st.session_state.entry_line = None
    st.session_state.exit_line  = None

paths = st.session_state.tmp_paths
names = st.session_state.names

if paths:
    st.caption("Wähle den Clip, dessen erstes Frame als Zeichengrundlage dient.")
    st.session_state.ref_idx = st.selectbox(
        "Referenz-Clip", options=list(range(len(paths))),
        index=st.session_state.ref_idx, format_func=lambda i: names[i]
    )

    first_frame = load_first_frame(paths[st.session_state.ref_idx], max_w=960)
    if first_frame is None:
        st.error("Konnte ersten Frame nicht laden."); st.stop()

    st.markdown("### Vorschau-Frame")
    st.image(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))

    # --------- Linien erfassen via Klicks ----------
    st.markdown("### Einfahr-Linie wählen")
    ep1, ep2 = click_two_points_on_image(first_frame, "Einfahrt: 2 Punkte klicken", key=f"entry_{st.session_state.ref_idx}")
    if ep1 and ep2:
        st.session_state.entry_line = (ep1, ep2)
        st.success(f"Einfahrt gesetzt: {ep1} → {ep2}")

    st.markdown("### Ausfahr-Linie wählen")
    xp1, xp2 = click_two_points_on_image(first_frame, "Ausfahrt: 2 Punkte klicken", key=f"exit_{st.session_state.ref_idx}")
    if xp1 and xp2:
        st.session_state.exit_line = (xp1, xp2)
        st.success(f"Ausfahrt gesetzt: {xp1} → {xp2}")

    # -------------------------------------------------
    if st.button("Analysieren", type="primary"):
        if not st.session_state.entry_line or not st.session_state.exit_line:
            st.error("Bitte erst **beide** Linien durch je 2 Klicks setzen.")
            st.stop()

        entry_line = st.session_state.entry_line
        exit_line  = st.session_state.exit_line

        model = YOLO("yolov8n.pt")
        results_table, trimmed = [], []

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
                     "t_in": f"{t_in:.3f}", "t_out": f"{t_out:.3f}"}
                )
                pad = 0.15
                outp = os.path.join(tempfile.gettempdir(), f"trim_{idx}.mp4")
                trim_clip(vpath, outp, t_in - pad, t_out + pad)
                trimmed.append((outp, best_id, dt))

        st.subheader("Sektorzeiten (schnellstes Fahrzeug pro Clip)")
        st.dataframe(results_table, use_container_width=True)

        if not trimmed:
            st.warning("Keine gültigen Sektorzeiten erkannt."); st.stop()

        st.markdown("### Overlay-Auswahl (bis 3)")
        pretty = [f"{names[i]} (ID {int(vid)}, {d:.3f}s)" for i, (p, vid, d) in enumerate(trimmed)]
        sel = st.multiselect("Wähle Clips", options=list(range(len(pretty))),
                             format_func=lambda i: pretty[i],
                             default=list(range(min(3, len(pretty)))))

        if sel:
            chosen = [trimmed[i] for i in sel][:3]
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
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))

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
                        ended += 1
                        continue
                    frames.append(resize_to_width(fr, out_width))
                if ended == len(caps): break
                while len(frames) < len(caps):
                    frames.append(np.zeros((h, w, 3), dtype=np.uint8))
                writer.write(alpha_blend(frames, alphas[: len(frames)]))

            for c in caps: c.release()
            writer.release()

            st.success("Fertig! Overlay & Zeiten erzeugt.")
            st.video(out_path)
            st.download_button(
                "Overlay-Video herunterladen",
                data=open(out_path, "rb").read(),
                file_name="overlay.mp4", mime="video/mp4"
            )
