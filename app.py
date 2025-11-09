# app.py
import os
import tempfile
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events


# ============ Utils ============
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

def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def center_of_box(xyxy):
    x1, y1, x2, y2 = xyxy
    return (0.5*(x1+x2), 0.5*(y1+y2))

def seg_intersection(p1, p2, q1, q2):
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    return (ccw(p1,q1,q2) != ccw(p2,q1,q2)) and (ccw(p1,p2,q1) != ccw(p1,p2,q2))

def make_fig_with_lines(img_rgb: np.ndarray,
                        entry_pts: List[Tuple[float,float]],
                        exit_pts:  List[Tuple[float,float]],
                        width_px: int = 800) -> go.Figure:
    """Plotly-Figure mit Bild und bereits gesetzten Linien (falls vorhanden)."""
    h, w = img_rgb.shape[:2]
    fig = go.Figure()
    fig.add_trace(go.Image(z=img_rgb))
    # Plotly-Bild-Koordinaten: x in [0,w], y in [0,h] mit y nach unten -> wir spiegeln Achsen passend
    fig.update_xaxes(showgrid=False, visible=False, range=[0, w])
    fig.update_yaxes(showgrid=False, visible=False, range=[h, 0])  # invert y
    shapes = []
    if len(entry_pts) == 2:
        shapes.append(dict(type="line",
                           x0=entry_pts[0][0], y0=entry_pts[0][1],
                           x1=entry_pts[1][0], y1=entry_pts[1][1],
                           line=dict(color="lime", width=4)))
    if len(exit_pts) == 2:
        shapes.append(dict(type="line",
                           x0=exit_pts[0][0], y0=exit_pts[0][1],
                           x1=exit_pts[1][0], y1=exit_pts[1][1],
                           line=dict(color="red", width=4)))
    fig.update_layout(width=min(width_px, w), margin=dict(l=0,r=0,t=0,b=0), shapes=shapes)
    return fig

def simple_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, x2-x1), max(0, y2-y1)
    inter = iw*ih
    area_a = (ax2-ax1)*(ay2-ay1)
    area_b = (bx2-bx1)*(by2-by1)
    return inter / (area_a + area_b - inter + 1e-6)

class SimpleTracker:
    """Ganz einfacher IoU-Tracker ohne externe Libs."""
    def __init__(self, iou_thresh=0.3, max_age=20):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.next_id = 1
        self.tracks = {}  # id -> {"box":..., "age":0}

    def update(self, detections):
        assigned = set()
        for tid, t in list(self.tracks.items()):
            best_iou, best_j = 0.0, -1
            for j, d in enumerate(detections):
                if j in assigned: continue
                i = simple_iou(t["box"], d)
                if i > best_iou: best_iou, best_j = i, j
            if best_iou >= self.iou_thresh:
                t["box"] = detections[best_j]
                t["age"] = 0
                assigned.add(best_j)
            else:
                t["age"] += 1
                if t["age"] > self.max_age:
                    del self.tracks[tid]
        for j, d in enumerate(detections):
            if j not in assigned:
                self.tracks[self.next_id] = {"box": d, "age": 0}
                self.next_id += 1
        return [(tid, t["box"]) for tid, t in self.tracks.items()]


# ============ App ============
st.set_page_config(page_title="Sektorzeiten – Klicklinien", layout="wide")
st.title("🏁 Sektorzeiten aus Video (Klick-Linien)")

st.markdown("1) **Video hochladen** → 2) Im ersten Frame **2 Klicks** für *Einfahrtslinie* und **2 Klicks** für *Ausfahrtslinie* → 3) **Analysieren**.")

# Session-State für Punkte
if "entry_pts" not in st.session_state: st.session_state.entry_pts = []
if "exit_pts"  not in st.session_state: st.session_state.exit_pts  = []
if "mode"      not in st.session_state: st.session_state.mode = "Einfahrt"

uploaded = st.file_uploader("Video (MP4/MOV)", type=["mp4", "mov", "m4v"], accept_multiple_files=False)

if uploaded:
    # Temp speichern
    suffix = os.path.splitext(uploaded.name)[1].lower()
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tfile.write(uploaded.read()); tfile.flush(); tfile.close()
    video_path = tfile.name

    first_bgr = load_first_frame(video_path, max_w=1280)
    if first_bgr is None:
        st.error("Konnte ersten Frame nicht laden."); st.stop()
    first_rgb = bgr2rgb(first_bgr)
    h, w = first_rgb.shape[:2]

    # UI: Modus wählen + Reset
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        st.session_state.mode = st.radio("Modus", ["Einfahrt", "Ausfahrt"], horizontal=True, index=0 if st.session_state.mode=="Einfahrt" else 1)
    with c2:
        if st.button("Reset Punkte"):
            st.session_state.entry_pts = []
            st.session_state.exit_pts = []

    # Plotly-Figur mit bereits vorhandenen Linien
    fig = make_fig_with_lines(first_rgb, st.session_state.entry_pts, st.session_state.exit_pts, width_px=900)

    # Klicks einsammeln
    events = plotly_events(fig, click_event=True, hover_event=False, select_event=False, override_height=h, override_width=min(900, w))
    if events:
        x = float(events[0]["x"])
        y = float(events[0]["y"])
        if st.session_state.mode == "Einfahrt" and len(st.session_state.entry_pts) < 2:
            st.session_state.entry_pts.append((x, y))
        elif st.session_state.mode == "Ausfahrt" and len(st.session_state.exit_pts) < 2:
            st.session_state.exit_pts.append((x, y))
        # Figur mit neuen Punkten sofort neu malen
        fig = make_fig_with_lines(first_rgb, st.session_state.entry_pts, st.session_state.exit_pts, width_px=900)
        st.plotly_chart(fig, use_container_width=False)

    ready = (len(st.session_state.entry_pts) == 2 and len(st.session_state.exit_pts) == 2)

    if not ready:
        st.info("⚠️ Klicke jeweils **zwei Punkte** für Einfahrt (grün) und Ausfahrt (rot).")

    # ---------- Analyse ----------
    if st.button("Analysieren", type="primary", disabled=not ready):
        entry_line = (st.session_state.entry_pts[0], st.session_state.entry_pts[1])
        exit_line  = (st.session_state.exit_pts[0],  st.session_state.exit_pts[1])

        model = YOLO("yolov8n.pt")
        tracker = SimpleTracker(iou_thresh=0.3, max_age=20)

        timings = {}     # tid -> {"entry": t, "exit": t}
        last_centers = {}  # tid -> (x,y)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_i = 0

        with st.spinner("Analysiere …"):
            while True:
                ok, frame = cap.read()
                if not ok: break

                yres = model.predict(frame, classes=[2, 3, 5, 7], conf=0.25, imgsz=960, verbose=False)
                dets = []
                if yres and len(yres) > 0 and yres[0].boxes is not None:
                    xyxys = yres[0].boxes.xyxy.cpu().numpy()
                    for b in xyxys:
                        dets.append(tuple(b.tolist()))

                tracks = tracker.update(dets)

                for tid, box in tracks:
                    cx, cy = center_of_box(box)
                    p2 = (cx, cy)
                    p1 = last_centers.get(tid, p2)
                    if tid not in timings: timings[tid] = {"entry": None, "exit": None}

                    if timings[tid]["entry"] is None and seg_intersection(p1, p2, entry_line[0], entry_line[1]):
                        timings[tid]["entry"] = frame_i / fps
                    if timings[tid]["exit"] is None and seg_intersection(p1, p2, exit_line[0], exit_line[1]):
                        timings[tid]["exit"] = frame_i / fps

                    last_centers[tid] = p2

                frame_i += 1

        cap.release()

        rows = []
        for tid, t in timings.items():
            t_in, t_out = t["entry"], t["exit"]
            if t_in is not None and t_out is not None and t_out > t_in:
                rows.append({
                    "Fahrzeug-ID": int(tid),
                    "t_in [s]": round(t_in, 3),
                    "t_out [s]": round(t_out, 3),
                    "Sektorzeit [s]": round(t_out - t_in, 3),
                })

        if not rows:
            st.warning("Keine gültigen Sektorzeiten erkannt. Prüfe Linienposition und Videoqualität.")
            st.stop()

        rows = sorted(rows, key=lambda r: r["Sektorzeit [s]"])
        st.success(f"Fertig! {len(rows)} Fahrzeuge mit gültiger Sektorzeit.")
        st.dataframe(rows, use_container_width=True)
