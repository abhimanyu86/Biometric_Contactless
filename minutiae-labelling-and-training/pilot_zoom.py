"""Zoomed, contrast-enhanced crop with a 10 px grid and current points: python pilot_zoom.py NAME X0 Y0 SIZE [ZOOM]"""
import sys, os, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_tiles as pt
name, x0, y0, size = sys.argv[1], *map(int, sys.argv[2:5])
zoom = int(sys.argv[5]) if len(sys.argv) > 5 else 8
d = pt.load(name)
img = cv2.imread(d["image"], cv2.IMREAD_GRAYSCALE)
pad = 16
X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
big = img[Y0:y0 + size + pad, X0:x0 + size + pad]
big = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2, 2)).apply(big)
crop = big[y0 - Y0:y0 - Y0 + size, x0 - X0:x0 - X0 + size]
vis = cv2.cvtColor(cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
for g in range((x0 // 10 + 1) * 10, x0 + crop.shape[1], 10):
    X = (g - x0) * zoom; cv2.line(vis, (X, 0), (X, vis.shape[0]), (0, 200, 255) if g % 50 else (0, 120, 255), 1)
    cv2.putText(vis, str(g), (X + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 160, 255), 1)
for g in range((y0 // 10 + 1) * 10, y0 + crop.shape[0], 10):
    Y = (g - y0) * zoom; cv2.line(vis, (0, Y), (vis.shape[1], Y), (0, 200, 255) if g % 50 else (0, 120, 255), 1)
    cv2.putText(vis, str(g), (2, Y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 160, 255), 1)
pt.draw_points(vis, [p for p in d["points"] if x0 <= p["x"] < x0 + size and y0 <= p["y"] < y0 + size], zoom, x0, y0, 0.55)
out = pt.path(name, "zoom.png"); cv2.imwrite(out, vis); print(out)
