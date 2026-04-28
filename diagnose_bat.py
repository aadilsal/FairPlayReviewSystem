from ultralytics import YOLO
import cv2, numpy as np

primary = YOLO('weights/bat_weights_new.pt')
fallback = YOLO('weights/Bat_detection2/weights/best.pt')

print("=== PRIMARY model.names ===")
print(primary.names)
print(f"    nc (num classes): {primary.model.nc}")

print("\n=== FALLBACK model.names ===")
print(fallback.names)
print(f"    nc (num classes): {fallback.model.nc}")

# Test on a white frame — a healthy model will produce garbage detections
# A broken/wrong model will produce zero even here
blank = np.ones((640, 640, 3), dtype=np.uint8) * 200

r_primary  = primary.predict(blank,  conf=0.001, verbose=False)
r_fallback = fallback.predict(blank, conf=0.001, verbose=False)

print(f"\n=== Blank frame test (conf=0.001) ===")
print(f"PRIMARY  boxes on blank frame: {len(r_primary[0].boxes)}")
print(f"FALLBACK boxes on blank frame: {len(r_fallback[0].boxes)}")

# Test on an actual video frame
cap = cv2.VideoCapture('test_vids/t4.mp4')
ret, frame = cap.read()
cap.release()

if ret:
    r_primary  = primary.predict(frame,  conf=0.001, verbose=False)
    r_fallback = fallback.predict(frame, conf=0.001, verbose=False)
    print(f"\n=== Real frame test (conf=0.001) ===")
    print(f"PRIMARY  boxes: {len(r_primary[0].boxes)}")
    if len(r_primary[0].boxes) > 0:
        for b in r_primary[0].boxes:
            print(f"  cls={int(b.cls[0])}  conf={float(b.conf[0]):.4f}  box={b.xyxy[0].tolist()}")
    print(f"FALLBACK boxes: {len(r_fallback[0].boxes)}")
    if len(r_fallback[0].boxes) > 0:
        for b in r_fallback[0].boxes:
            print(f"  cls={int(b.cls[0])}  conf={float(b.conf[0]):.4f}  box={b.xyxy[0].tolist()}")