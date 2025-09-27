import cv2
from flask import Flask, Response, render_template_string
from ultralytics import YOLO

# Load your trained model
MODEL_PATH = "runs/detect/train/weights/best.pt"
model = YOLO(MODEL_PATH)

# Initialize Flask app
app = Flask(__name__)

# HTML template (very simple)
HTML_PAGE = """
<!doctype html>
<html>
<head>
  <title>VigilHawk Weapon Detection</title>
</head>
<body>
  <h1>VigilHawk Live Feed (Press Q to stop server)</h1>
  <img src="/video_feed" width="800">
</body>
</html>
"""

def generate_frames():
    cap = cv2.VideoCapture(0)  # 0 = default webcam, change to RTSP/file if needed
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)  # inference
        annotated_frame = results[0].plot()

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        # Yield frame in HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Run Flask server
    app.run(host="0.0.0.0", port=5000, debug=False)

