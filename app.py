# app.py
import os
import tempfile
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events


# ===================== Utils =====================

def load_first_frame(video_path, max_w=1280):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = frame.shape[:2]
    if w > max_w:
        s = max_w / w
        frame = cv2.resize(frame, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    return frame

def bgr2rgb(img): return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def center_of_box(xyxy):
    x1, y1, x2, y2 = xyxy
    return (0.5*(x1+x2), 0.5*(y1+y2))

def seg_intersection(p1, p2, q1, q2):
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    return (ccw(p1,q1,q2) != ccw(p2,q1,q2)) and (ccw(p1,p2,q1) != ccw(p1,p2,q2))


# ===================== Besserer leichter Tracker =====================
# Greedy Matching mit kombiniertem Score aus IoU und normierter Mittelpunkt-Distanz.
# Stabilere ID-Haltung als der ganz einfache IoU-only-Tracker.
class LightTracker:
    def __init__(self, iou_w=0.6, dist_w=0.4, dist_gate=0.6, score_gate=0.25, max_age=15):
        self.iou_w = iou_w
        self.dist_w = dist_w
        self.dist_gate = dist_gate        # 0..1 (1 = sehr weit ok)
        self.score_gate = score_gate      # minimaler Kombi-Score
        self.max_age = max_age
        self.next_id = 1
        # id -> dict(box, age, last_center, vel, trail)
        self.tracks = {}

    @staticmethod
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        x1, y1 = max(ax1, bx1), max(ay1, by1)
        x2, y2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, x2-x1), max(0, y2-y1)
        inter = iw*ih
        area_a = (ax2-ax1)*(ay2-ay1)
        area_b = (bx2-bx1)*(by2-by1)
        return inter / (area_a + area_b - inter + 1e-6)

    @staticmethod
    def norm_center_dist(a, b, img_w, img_h):
        ax, ay = center_of_box(a)
        bx, by = center_of_box(b)
        # normiert auf Bilddiagonale
        d = np.hypot(ax-bx, ay-by) / (np.hypot(img_w, img_h) + 1e-6)
        # clip auf [0,1]
        return float(np.clip(d, 0.0, 1.0))

    def update(self, detections, img_w, img_h):
        # detections: list of (x1,y1,x2,y2)
        assigned_dets = set()

        # 1) existierende Tracks matchen
        for tid, t in list(self.tracks.items()):
            best_score, best_j = -1.0, -1
            for j, d in enumerate(detections):
                if j in assigned_dets:
                    continue
                iou = self.iou(t["box"], d)
                nd  = 1.0 - self.norm_center_dist(t["box"], d, img_w, img_h)  # 1=nah, 0=weit
                # harte Distanz-Gate
                if (1.0 - nd) > self.dist_gate:
                    continue
                score = self.iou_w * iou + self.dist_w * nd
                if score > best_score:
                    best_score, best_j = score, j
            if best_j >= 0 and best_score >= self.score_gate:
                # Match
                new_box = detections[best_j]
                cx_old, cy_old = center_of_box(t["box"])
                cx_new, cy_new = center_of_box(new_box)
                vel = (cx_new - cx_old, cy_new - cy_old)
                t.update({"box": new_box, "age": 0, "last_center": (cx_new, cy_new), "vel": vel})
                trail = t.get("trail", [])
                trail.append((int(cx_new), int(cy_new)))
                if len(trail) > 50: trail = trail[-50:]
                t["trail"] = trail
                assigned_dets.add(best_j)
            else:
                # nicht gematcht -> altern lassen
                t["age"] = t.get("age", 0) + 1
                if t["age"] > self.max_age:
                    del self.tracks[tid]

        # 2) neue Tracks anlegen
        for j, d in enumerate(detections):
            if j in assigned_dets:
                continue
            cx, cy = center_of_box(d)
            self.tracks[self.next_id] = {
                "box": d, "age": 0, "last_center": (cx, cy), "vel": (0.0, 0.0), "trail": [(int(cx), int(cy))]
            }
            self.next_id += 1

        # 3) Ausgabe
        return [(tid, t["box"]) for tid, t in self.tracks.items()]


# -------- Plotly Bild + Linien (für die Punktwahl) --------
def make_fig_with_lines(img_rgb: np.ndarray,
                        entry_pts: List[Tuple[float,float]],
                        exit_pts:  List[Tuple[float,float]],
                        width_px: int = 1000) -> go.Figure:
    h, w = img_rgb.shape[:2]
    fig = px.imshow(img_rgb, binary_format="png", origin="upper")
    fig.update_xaxes(visible=False, range=[0, w])
    fig.update_yaxes(visible=False, range=[h, 0])  # invert y

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
    width = min(width_px, w)
    height = int(h * (width / w))
    fig.update_layout(width=width, height=height, margin=dict(l=0,r=0,t=0,b=0), shapes=shapes)
    return fig

def capture_plotly_clicks(fig, fallback_width, fallback_height, key):
    try:
        return plotly_events(fig, click_event=True, hover_event=False, select_event=False,
                             override_height=fallback_height, override_width=fallback_width, key=key)
    except TypeError:
        try:
            return plotly_events(fig, click_event=True, hover_event=False, select_event=False)
        except Exception:
            st.plotly_chart(fig, use_container_width=False)
            return []


# ===================== App =====================

st.set_page_config(page_title="Sektorzeiten – Live Tracking", layout="wide")
st.title("🏁 Sektorzeiten aus Video — Live-Tracking")

st.caption("1) **Video hochladen** → 2) Im ersten Frame **2 Punkte** für *Einfahrt* (grün) und **2 Punkte** für *Ausfahrt* (rot) → 3) **Analysieren** & Live-Tracking.")

# Session State
if "entry_pts" not in st.session_state: st.session_state.entry_pts = []
if "exit_pts"  not in st.session_state: st.session_state.exit_pts  = []
if "mode"      not in st.session_state: st.session_state.mode = "Einfahrt"

uploaded = st.file_uploader("Video (MP4/MOV)", type=["mp4","mov","m4v"], accept_multiple_files=False)

if uploaded:
    # Tempfile
    suffix = os.path.splitext(uploaded.name)[1].lower()
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tfile.write(uploaded.read()); tfile.flush(); tfile.close()
    video_path = tfile.name

    first_bgr = load_first_frame(video_path, max_w=1280)
    if first_bgr is None:
        st.error("Konnte ersten Frame nicht laden."); st.stop()
    first_rgb = bgr2rgb(first_bgr)
    h, w = first_rgb.shape[:2]

    # Linien setzen
    c1, c2, _ = st.columns([1,1,2])
    with c1:
        st.session_state.mode = st.radio("Modus", ["Einfahrt","Ausfahrt"],
                                         horizontal=True,
                                         index=0 if st.session_state.mode=="Einfahrt" else 1)
    with c2:
        if st.button("Reset Punkte"):
            st.session_state.entry_pts = []; st.session_state.exit_pts = []

    fig = make_fig_with_lines(first_rgb, st.session_state.entry_pts, st.session_state.exit_pts, width_px=1000)
    widget_key = f"click_{st.session_state.mode}_{len(st.session_state.entry_pts)}_{len(st.session_state.exit_pts)}"
    events = capture_plotly_clicks(fig, fallback_width=min(1000, w),
                                   fallback_height=int(h*(min(1000, w)/w)), key=widget_key)

    if events:
        x = float(events[0]["x"]); y = float(events[0]["y"])
        if st.session_state.mode == "Einfahrt" and len(st.session_state.entry_pts) < 2:
            st.session_state.entry_pts.append((x,y))
        elif st.session_state.mode == "Ausfahrt" and len(st.session_state.exit_pts) < 2:
            st.session_state.exit_pts.append((x,y))
        fig = make_fig_with_lines(first_rgb, st.session_state.entry_pts, st.session_state.exit_pts, width_px=1000)
        st.plotly_chart(fig, use_container_width=False)

    ready = (len(st.session_state.entry_pts)==2 and len(st.session_state.exit_pts)==2)
    if not ready:
        st.info("⚠️ Klicke jeweils **zwei Punkte** für Einfahrt (grün) und Ausfahrt (rot).")

    # ---------- Analyse + Live Anzeige ----------
    st.markdown("---")
    live = st.checkbox("Live anzeigen (kann langsamer sein)", value=True)
    conf = st.slider("YOLO Konfidenz", 0.1, 0.8, 0.25, 0.05)

    if st.button("Analysieren", type="primary", disabled=not ready):
        entry_line = (st.session_state.entry_pts[0], st.session_state.entry_pts[1])
        exit_line  = (st.session_state.exit_pts[0],  st.session_state.exit_pts[1])

        model = YOLO("yolov8n.pt")
        tracker = LightTracker(iou_w=0.6, dist_w=0.4, dist_gate=0.6, score_gate=0.25, max_age=12)

        timings, last_centers = {}, {}
        unique_ids_seen = set()
        ids_with_sector = set()

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Live-Widgets
        img_placeholder = st.empty()
        kpi1, kpi2, kpi3 = st.columns(3)
        prog = st.progress(0)
        info = st.empty()

        frame_i = 0
        while True:
            ok, frame = cap.read()
            if not ok: break

            # YOLO: nur Fahrzeuge (car, moto, bus, truck)
            yres = model.predict(frame, classes=[2,3,5,7], conf=conf, imgsz=960, verbose=False)
            dets = []
            if yres and len(yres)>0 and yres[0].boxes is not None:
                for b in yres[0].boxes.xyxy.cpu().numpy():
                    dets.append(tuple(b.tolist()))

            tracks = tracker.update(dets, img_w=frame.shape[1], img_h=frame.shape[0])

            # Visualisierung
            vis = frame.copy()
            # Linien
            cv2.line(vis, tuple(map(int, entry_line[0])), tuple(map(int, entry_line[1])), (0,255,0), 3)
            cv2.line(vis, tuple(map(int, exit_line[0])),  tuple(map(int, exit_line[1])),  (0,0,255), 3)

            # Tracks rendern & Schnittpunkte prüfen
            for tid, box in tracks:
                unique_ids_seen.add(tid)

                x1,y1,x2,y2 = map(int, box)
                cx, cy = map(int, center_of_box(box))

                # schicke Overlays: Box + ID + Symbol + Trail
                cv2.rectangle(vis, (x1,y1), (x2,y2), (255,255,0), 2)
                cv2.circle(vis, (cx,cy), 6, (0,255,255), -1)
                cv2.putText(vis, f"ID {tid}", (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

                # Trails (Bewegungsspur), falls vorhanden
                tr = tracker.tracks[tid].get("trail", [])
                for k in range(1, len(tr)):
                    cv2.line(vis, tr[k-1], tr[k], (255,255,0), 2)

                # Crossing-Logik
                p2 = (float(cx), float(cy))
                p1 = last_centers.get(tid, p2)
                if tid not in timings:
                    timings[tid] = {"entry": None, "exit": None}

                if timings[tid]["entry"] is None and seg_intersection(p1, p2, entry_line[0], entry_line[1]):
                    timings[tid]["entry"] = frame_i / fps
                    cv2.circle(vis, (cx, cy), 10, (0,255,0), 3)

                if timings[tid]["exit"] is None and seg_intersection(p1, p2, exit_line[0], exit_line[1]):
                    timings[tid]["exit"] = frame_i / fps
                    cv2.circle(vis, (cx, cy), 10, (0,0,255), 3)
                    if timings[tid]["entry"] is not None:
                        ids_with_sector.add(tid)

                last_centers[tid] = p2

            # sanftes Overlay (leichte Abdunklung + Zeichnung schärfer)
            overlay = vis.copy()
            cv2.rectangle(overlay, (0,0), (vis.shape[1], 40), (0,0,0), -1)
            vis = cv2.addWeighted(vis, 0.92, overlay, 0.08, 0)

            # KPIs aktualisieren
            kpi1.metric("Aktive Tracks", len(tracks))
            kpi2.metric("Verschiedene Fahrzeuge gesehen", len(unique_ids_seen))
            kpi3.metric("Mit Sektorzeit", len(ids_with_sector))

            # Live zeigen
            if live:
                img_placeholder.image(bgr2rgb(vis), use_container_width=True)

            # Fortschritt
            if total > 0:
                prog.progress(min(1.0, frame_i / float(total)))
            info.write(f"Frame {frame_i} / {total}")

            frame_i += 1

        cap.release()
        prog.empty(); info.empty()

        # Ergebnisse aggregieren
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
            st.warning("Keine gültigen Sektorzeiten erkannt. Probiere niedrigere Konfidenz oder andere Linienpositionen.")
        else:
            rows = sorted(rows, key=lambda r: r["Sektorzeit [s]"])
            st.success(f"Fertig! {len(rows)} Fahrzeuge mit gültiger Sektorzeit.  ·  Insgesamt gesehen: {len(unique_ids_seen)}")
            st.dataframe(rows, use_container_width=True)
