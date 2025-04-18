
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import base64
from PIL import Image
import io
import cv2

app = Flask(__name__)

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height, width = input_details[0]['shape'][1:3]

def detect(image_np, threshold=0.5):
    input_data = np.expand_dims(image_np, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])  # shape: (1, 5, 8400)
    output = output[0]  # shape: (5, 8400)
    x_center, y_center, w, h, conf = output

    boxes = []
    for i in range(conf.shape[0]):
        if conf[i] >= threshold:
            xc, yc, bw, bh = x_center[i], y_center[i], w[i], h[i]
            xmin = max(int((xc - bw / 2) * width), 0)
            ymin = max(int((yc - bh / 2) * height), 0)
            xmax = min(int((xc + bw / 2) * width), width)
            ymax = min(int((yc + bh / 2) * height), height)

            boxes.append({
                'box': [xmin, ymin, xmax, ymax],
                'confidence': round(float(conf[i]), 2)
            })

    return boxes

def draw_boxes(image_np, boxes):
    image_np = (image_np * 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    for box in boxes:
        x1, y1, x2, y2 = box['box']
        conf = box['confidence']
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{conf:.2f}"
        cv2.putText(image_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

    _, buffer = cv2.imencode('.jpg', image_bgr)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return encoded_image

@app.route('/upload', methods=['POST'])
def upload_image():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'Image data not provided'}), 400

    try:
        image_data = data['image'].split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Invalid image data: {str(e)}'}), 400

    image = image.resize((width, height))
    image_np = np.array(image) / 255.0

    boxes = detect(image_np)
    image_with_boxes = draw_boxes(image_np, boxes)

    return jsonify({
        'jumlah_benur': len(boxes),
        'deteksi': boxes,
        'image_with_boxes': f"data:image/jpeg;base64,{image_with_boxes}"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
