import queue
import threading
import struct
import numpy as np
import pandas as pd
import cv2
import streamlit as st
import serial
from sklearn.cluster import DBSCAN
import time

PORT = 'COM3'
BAUD_RATE = 230400
SIZE = 600
PX = 400

st.set_page_config(layout="wide")
st.title('Що бачить лідар D200')

def lidar_scans(port, q):
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        ser.setDTR(False)
        ser.setRTS(False)
        scan_angles, scan_dists = [], []
        previous_angle = None
        
        while True:
            if ser.read(1) != b'\x54': continue
            packet = ser.read(46)
            if len(packet) != 46 or packet[0] != 0x2C: continue
            
            start_angle = struct.unpack_from('<H', packet, 3)[0] / 100
            end_angle = struct.unpack_from('<H', packet, 41)[0] / 100
            
            if previous_angle is not None and start_angle < previous_angle:
                if len(scan_angles) > 100 and not q.full():
                    q.put((np.array(scan_angles), np.array(scan_dists)))
                scan_angles, scan_dists = [], []
            previous_angle = start_angle
            
            step = ((end_angle - start_angle) % 360) / 11
            for i in range(12):
                dist_mm = struct.unpack_from('<H', packet, 5 + 3 * i)[0]
                if dist_mm > 0:
                    scan_angles.append((start_angle + step * i) % 360)
                    scan_dists.append(dist_mm)
    except Exception as e:
        print(f"Помилка: {e}")

@st.cache_resource
def start_reader():
    q = queue.Queue(maxsize=5)
    threading.Thread(target=lidar_scans, args=(PORT, q), daemon=True).start()
    return q

q = start_reader()

st.sidebar.header("Легенда")
st.sidebar.markdown("🔵 **Сині точки** — Сигнал лідара\n\n🟢 **Зелені лінії** — Стіни (Гаф)\n\n🟠 **Помаранчеві рамки** — Предмети (DBSCAN)\n\n🔴 **Червоні рамки** — Зміни фону")

st.sidebar.header("Параметри")
eps = st.sidebar.slider('eps, м', 0.01, 0.20, 0.02)
min_samples = st.sidebar.slider('min_samples', 1, 10, 4)
z_threshold = st.sidebar.slider('поріг z', -5.0, -1.0, -3.0)

if 'background_scans' not in st.session_state: st.session_state.background_scans = []
if 'bg_mean' not in st.session_state: st.session_state.bg_mean = None
if 'bg_std' not in st.session_state: st.session_state.bg_std = None
if 'collect' not in st.session_state: st.session_state.collect = False
if 'changes_history' not in st.session_state: st.session_state.changes_history = []
if 'last_frame_data' not in st.session_state: st.session_state.last_frame_data = None

st.sidebar.header("Керування")
if st.sidebar.button('Зафіксувати сцену'):
    st.session_state.collect = True
    st.session_state.background_scans = []
    st.session_state.bg_mean = None
    st.session_state.changes_history = []
    
if st.sidebar.button('Скинути фон'):
    st.session_state.collect = False
    st.session_state.background_scans = []
    st.session_state.bg_mean = None
    st.session_state.changes_history = []

if st.sidebar.button('Зберегти знімок'):
    if st.session_state.last_frame_data:
        img, ang, dst = st.session_state.last_frame_data
        ts = int(time.time())
        cv2.imwrite(f'snapshot_{ts}.png', img)
        pd.DataFrame({'angle': ang, 'dist': dst}).to_csv(f'scan_{ts}.csv', index=False)
        st.sidebar.success(f"Збережено snapshot_{ts}!")

status_text = st.empty()
plan = st.empty()
col1, col2, col3, col4, col5 = st.columns(5)
m1, m2, m3, m4, m5 = col1.empty(), col2.empty(), col3.empty(), col4.empty(), col5.empty()
chart_spot = st.empty()

def process_scan(angles_deg, dists_mm, eps, min_samples, z_threshold):
    angles_rad = angles_deg * np.pi / 180
    dists_m = dists_mm / 1000.0
    x_m = dists_m * np.sin(angles_rad)
    y_m = dists_m * np.cos(angles_rad)
    px = (SIZE / 2 + x_m * PX).astype(int)
    py = (SIZE / 2 - y_m * PX).astype(int)
    
    img = np.ones((SIZE, SIZE, 3), dtype=np.uint8) * 255
    for r in range(int(0.1 * PX), int(2.0 * PX), int(0.1 * PX)):
        cv2.circle(img, (SIZE//2, SIZE//2), r, (220, 220, 220), 1)
    
    pts = np.array([[SIZE//2, SIZE//2-10], [SIZE//2-8, SIZE//2+8], [SIZE//2+8, SIZE//2+8]], np.int32)
    cv2.fillPoly(img, [pts], (0, 0, 0))
    
    for p_x, p_y in zip(px, py):
        if 0 <= p_x < SIZE and 0 <= p_y < SIZE:
            cv2.circle(img, (p_x, p_y), 2, (255, 0, 0), -1)
            
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    for p_x, p_y in zip(px, py):
        if 0 <= p_x < SIZE and 0 <= p_y < SIZE: mask[p_y, p_x] = 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)

    n_walls = 0
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=50, minLineLength=60, maxLineGap=30)
    
    if lines is not None:
        lines = np.asarray(lines).reshape(-1, 4)
        lines = sorted(lines, key=lambda line: np.hypot(line[2] - line[0], line[3] - line[1]), reverse=True)
        
        walls = []
        for x1, y1, x2, y2 in lines:
            direction = np.arctan2(y2 - y1, x2 - x1)
            duplicate = False
            
            for a, b, c, d in walls:
                other = np.arctan2(d - b, c - a)
                angle = abs(np.degrees(direction - other)) % 180
                angle = min(angle, 180 - angle)
                
                distance = abs((c - a) * ((y1 + y2) / 2 - b) - (d - b) * ((x1 + x2) / 2 - a)) / max(np.hypot(c - a, d - b), 1e-5)
                
                if angle < 8 and distance < 10:
                    duplicate = True
                    break
                    
            if not duplicate:
                walls.append((x1, y1, x2, y2))
                
        n_walls = len(walls)
        for x1, y1, x2, y2 in walls:
            cv2.line(img, (x1, y1), (x2, y2), (0, 200, 0), 3)
            
    points = np.column_stack((x_m, y_m))
    n_objects, n_noise, n_changes = 0, 0, 0
    
    if len(points) > 0:
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = db.labels_
        clusters = set(labels) - {-1}
        n_noise = np.sum(labels == -1)
        
        for i in clusters:
            c_pts = points[labels == i]
            rect_points = np.column_stack(((SIZE / 2 + c_pts[:, 0] * PX).astype(int), (SIZE / 2 - c_pts[:, 1] * PX).astype(int)))
            x, y, w, h = cv2.boundingRect(rect_points)
            if w < int(0.3 * PX) and h < int(0.3 * PX):
                cv2.rectangle(img, (x-5, y-5), (x+w+5, y+h+5), (0, 165, 255), 2)
                n_objects += 1
                
    if st.session_state.collect:
        st.session_state.background_scans.append((angles_deg.astype(int), dists_m))
        if len(st.session_state.background_scans) >= 10:
            st.session_state.collect = False
            bg_dists = [[] for _ in range(360)]
            for scan_a, scan_d in st.session_state.background_scans:
                for a, d in zip(scan_a, scan_d): bg_dists[a % 360].append(d)
            mean_arr, std_arr = np.zeros(360), np.zeros(360)
            for deg in range(360):
                if bg_dists[deg]:
                    mean_arr[deg], std_arr[deg] = np.mean(bg_dists[deg]), max(np.std(bg_dists[deg]), 0.01)
            st.session_state.bg_mean, st.session_state.bg_std = mean_arr, std_arr
            
    if st.session_state.bg_mean is not None:
        new_pts = []
        for a_deg, d_m in zip(angles_deg, dists_m):
            idx = int(a_deg) % 360
            mu, sigma = st.session_state.bg_mean[idx], st.session_state.bg_std[idx]
            if mu != 0 and ((d_m - mu) / sigma < z_threshold):
                new_pts.append([d_m * np.sin(a_deg * np.pi / 180), d_m * np.cos(a_deg * np.pi / 180)])
                
        if new_pts:
            new_pts_arr = np.array(new_pts)
            change_labels = DBSCAN(eps=eps, min_samples=min_samples).fit(new_pts_arr).labels_
            change_clusters = set(change_labels) - {-1}
            n_changes = len(change_clusters)
            
            for i in change_clusters:
                c_pts = new_pts_arr[change_labels == i]
                c_px, c_py = (SIZE / 2 + c_pts[:, 0] * PX).astype(int), (SIZE / 2 - c_pts[:, 1] * PX).astype(int)
                rx, ry, rw, rh = cv2.boundingRect(np.column_stack((c_px, c_py)))
                cv2.rectangle(img, (rx-10, ry-10), (rx+rw+10, ry+rh+10), (0, 0, 255), 2)
                cv2.putText(img, "ZMINA", (rx-10, ry-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return img, n_walls, n_objects, n_noise, n_changes

last_time = time.time()

while True:
    if not q.empty():
        angles, dists = q.get()
        
        curr_time = time.time()
        fps = round(1.0 / (curr_time - last_time), 1) if curr_time - last_time > 0 else 0
        last_time = curr_time
        
        if st.session_state.collect:
            status = f"Збираю {len(st.session_state.background_scans)} / 10"
        elif st.session_state.bg_mean is not None:
            status = "зафіксовано"
        else:
            status = "немає"
            
        status_text.markdown(f"**Стан фону:** {status}")
        
        img, n_walls, n_objects, n_noise, n_changes = process_scan(angles, dists, eps, min_samples, z_threshold)
        
        st.session_state.last_frame_data = (img, angles, dists)
        
        if st.session_state.bg_mean is not None:
            st.session_state.changes_history.append(n_changes)
            if len(st.session_state.changes_history) > 50:
                st.session_state.changes_history.pop(0)
            chart_spot.line_chart(st.session_state.changes_history, height=150)
        
        plan.image(img, channels='BGR', use_container_width=True)
        m1.metric('Стін (OpenCV)', n_walls)
        m2.metric('Предметів (DBSCAN)', n_objects)
        m3.metric('ЗМІН НА СЦЕНІ', n_changes)
        m4.metric('Шумових точок', n_noise)
        m5.metric('Обертів/сек (FPS)', fps)
        
    time.sleep(0.01)