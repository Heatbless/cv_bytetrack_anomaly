import cv2
import numpy as np

INPUT_VIDEO = 'CCTV_Sleman.mp4'

# =========================================================
# Ambil frame pertama dari video
# =========================================================
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise FileNotFoundError(f'Cannot open: {INPUT_VIDEO}')
ret, frame = cap.read()
cap.release()
if not ret:
    raise RuntimeError('Gagal membaca frame.')

h, w = frame.shape[:2]
print(f'Frame size: {w}x{h}')
print('='*55)
print('PETUNJUK:')
print('  Klik 4 titik untuk ZONA KIRI  (searah jarum jam)')
print('  Klik 4 titik untuk ZONA KANAN (searah jarum jam)')
print('  Total = 8 klik')
print('  Tekan R  = reset / ulang dari awal')
print('  Tekan Q  = selesai & print koordinat')
print('='*55)

# =========================================================
# State
# =========================================================
ZONE_NAMES  = ['LEFT SIDEWALK', 'RIGHT SIDEWALK']
ZONE_COLORS = [(0, 0, 255), (0, 0, 255)]   # merah keduanya
DOT_COLOR   = (0, 255, 255)                 # kuning titik
PTS_PER_ZONE = 4
TOTAL_ZONES  = 2

all_points = []   # list of list
current_zone_pts = []

def draw_state(img, all_points, current_zone_pts):
    canvas = img.copy()

    # Zona yang sudah selesai
    for zi, pts in enumerate(all_points):
        arr = np.array(pts, np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [arr], ZONE_COLORS[zi])
        canvas = cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0)
        cv2.polylines(canvas, [arr], True, ZONE_COLORS[zi], 3)
        for j, pt in enumerate(pts):
            cv2.circle(canvas, pt, 7, DOT_COLOR, -1)
            cv2.putText(canvas, f'P{j}({pt[0]},{pt[1]})',
                        (pt[0]+6, pt[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, DOT_COLOR, 1)
        cx = int(arr[:,0].mean())
        cy = int(arr[:,1].mean())
        cv2.putText(canvas, ZONE_NAMES[zi], (cx-60, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # Titik zona sedang dikerjakan
    zi_current = len(all_points)
    for j, pt in enumerate(current_zone_pts):
        cv2.circle(canvas, pt, 7, DOT_COLOR, -1)
        cv2.putText(canvas, f'P{j}({pt[0]},{pt[1]})',
                    (pt[0]+6, pt[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, DOT_COLOR, 1)
    if len(current_zone_pts) > 1:
        for k in range(len(current_zone_pts)-1):
            cv2.line(canvas, current_zone_pts[k], current_zone_pts[k+1],
                     ZONE_COLORS[zi_current] if zi_current < TOTAL_ZONES else (200,200,200), 2)

    # HUD info
    zones_done = len(all_points)
    pts_done   = len(current_zone_pts)
    if zones_done < TOTAL_ZONES:
        zone_label = ZONE_NAMES[zones_done]
        remaining  = PTS_PER_ZONE - pts_done
        msg = f'Zona: {zone_label}  |  Klik {remaining} titik lagi'
    else:
        msg = 'Semua zona selesai! Tekan Q untuk print koordinat.'

    cv2.rectangle(canvas, (0, h-38), (w, h), (30,30,30), -1)
    cv2.putText(canvas, msg, (10, h-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,180), 2)
    cv2.putText(canvas, 'R=Reset  Q=Selesai', (w-220, h-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    return canvas

def mouse_callback(event, x, y, flags, param):
    global all_points, current_zone_pts, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(all_points) >= TOTAL_ZONES:
            return
        current_zone_pts.append((x, y))
        if len(current_zone_pts) == PTS_PER_ZONE:
            all_points.append(current_zone_pts.copy())
            current_zone_pts.clear()
            zi = len(all_points) - 1
            print(f'✅ {ZONE_NAMES[zi]} selesai: {all_points[zi]}')
        canvas = draw_state(frame, all_points, current_zone_pts)
        cv2.imshow(WINDOW, canvas)

WINDOW = 'Zone Calibrator  |  Klik 4 titik per zona  |  R=Reset  Q=Selesai'
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW, mouse_callback)

canvas = draw_state(frame, all_points, current_zone_pts)
cv2.imshow(WINDOW, canvas)

while True:
    key = cv2.waitKey(20) & 0xFF

    if key == ord('r'):
        all_points.clear()
        current_zone_pts.clear()
        print('🔄 Reset. Mulai dari awal.')
        canvas = draw_state(frame, all_points, current_zone_pts)
        cv2.imshow(WINDOW, canvas)

    elif key == ord('q'):
        break

    # Redraw setiap frame
    canvas = draw_state(frame, all_points, current_zone_pts)
    cv2.imshow(WINDOW, canvas)

cv2.destroyAllWindows()

# =========================================================
# Print koordinat siap pakai
# =========================================================
print()
print('='*55)
print('COPY KOORDINAT INI KE CELL 5:')
print('='*55)
print('SIDEWALK_ZONES = [')
for zi, pts in enumerate(all_points):
    print(f'    # {ZONE_NAMES[zi]}')
    print('    np.array([')
    for j, pt in enumerate(pts):
        corner = ['top-left','top-right','bottom-right','bottom-left'][j]
        print(f'        [{pt[0]:4d}, {pt[1]:4d}],   # {corner}')
    print('    ], np.int32),')
    print()
print(']')
print('='*55)