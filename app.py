from ultralytics import YOLO
from flask import request, Response, Flask
from waitress import serve
from PIL import Image
import json

app = Flask(__name__)

model = YOLO("best.pt")

@app.route("/")
def root():
    try:
        with open("index.html", encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return Response(
            json.dumps({"error": "index.html not found"}), 
            mimetype='application/json'
        ), 404
    except UnicodeDecodeError:
        return Response(
            json.dumps({"error": "Error reading HTML file - encoding issue"}), 
            mimetype='application/json'
        ), 500

@app.route("/detect", methods=["POST"])
def detect():
    if "image_file" not in request.files:
        return Response(
            json.dumps({"error": "No image file provided"}),  
            mimetype='application/json'
        ), 400

    try:
        buf = request.files["image_file"]
        image = Image.open(buf.stream)
        boxes, safety_status = detect_objects_on_image(image)
        response_data = {
            "boxes": boxes,
            "safety_status": safety_status
        }
        return Response(
            json.dumps(response_data),  
            mimetype='application/json'
        )
    except Exception as e:
        return Response(
            json.dumps({"error": str(e)}),  
            mimetype='application/json'
        ), 500

def detect_objects_on_image(image):
    results = model.predict(image)
    result = results[0]
    output = []
    person_detected = vest_detected = False
    helmet_detected = {
        "helmet": False,
        "yellow": False,
        "white": False,
        "blue": False
    }

    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        class_name = result.names[class_id]
        output.append([
            x1, y1, x2, y2, class_name, prob
        ])

        if class_name == "person":
            person_detected = True
        elif class_name == "vest":
            vest_detected = True
        elif class_name in helmet_detected:
            helmet_detected[class_name] = True

    safety_status = "safe" if person_detected and vest_detected and any(helmet_detected.values()) else "unsafe"
    return output, safety_status

if __name__ == '__main__':
    serve(app, host='127.0.0.1', port=5000)
