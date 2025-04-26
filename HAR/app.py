from flask import Flask, render_template, Response, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
import moviepy.editor as mp

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model
model = load_model('my_model.h5')

# Labels (update based on your model)
labels = ["Walking", "Running", "Sitting", "Jumping"]

# Webcam capture
camera = cv2.VideoCapture(0)

# Live video feed route
def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            img = cv2.resize(frame, (64, 64))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Duplicate image 20 times to match input shape
            video_like_input = np.repeat(img_array, 20, axis=0)
            video_like_input = np.expand_dims(video_like_input, axis=0)  # Shape: (1, 20, 64, 64, 3)

            try:
                prediction = model.predict(video_like_input)
                predicted_index = np.argmax(prediction)

                if predicted_index >= len(labels):
                    result = "Unknown"
                else:
                    result = labels[predicted_index]
            except Exception as e:
                result = f"Error: {str(e)}"

            # Overlay prediction
            cv2.putText(frame, f'Activity: {result}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        filename = secure_filename(file.filename)
        filepath = os.path.join('static', filename)
        file.save(filepath)

        if filename.lower().endswith(('.mp4', '.avi', '.mov')):
            video = mp.VideoFileClip(filepath)
            frames = []
            for i, frame in enumerate(video.iter_frames(fps=1)):
                if len(frames) >= 20:
                    break
                img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                img = cv2.resize(img, (64, 64))
                img_array = image.img_to_array(img) / 255.0
                frames.append(img_array)

            if len(frames) < 20:
                return "Video too short. At least 20 frames needed."

            frames_np = np.array(frames)
            frames_np = np.expand_dims(frames_np, axis=0)  # Shape: (1, 20, 64, 64, 3)

            prediction = model.predict(frames_np)
            predicted_index = np.argmax(prediction)

            if predicted_index >= len(labels):
                final_prediction = "Unknown"
            else:
                final_prediction = labels[predicted_index]

            return render_template('index.html', prediction=final_prediction)

        else:
            try:
                img = Image.open(filepath)
                img = img.resize((64, 64))
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                img_array = np.repeat(img_array, 20, axis=0)
                img_array = np.expand_dims(img_array, axis=0)  # (1, 20, 64, 64, 3)

                prediction = model.predict(img_array)
                predicted_index = np.argmax(prediction)

                if predicted_index >= len(labels):
                    result_label = "Unknown"
                else:
                    result_label = labels[predicted_index]

                return render_template('index.html', prediction=result_label)

            except UnidentifiedImageError:
                return "Unable to identify image format."

    except Exception as e:
        return f"Internal Server Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
