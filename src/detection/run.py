from ultralytics import YOLO
import cv2
import time

model = YOLO("runs/detect/train/weights/best.pt")  # or path to downloaded .pt
cap = cv2.VideoCapture(0)  # change to RTSP or file if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break
    t0 = time.time()
    results = model(frame, stream=False)   # returns list-like
    annotated = results[0].plot()
    fps = 1.0 / (time.time() - t0)
    cv2.putText(annotated, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
    cv2.imshow("VigilHawk - Weapons", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
