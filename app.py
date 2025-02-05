import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load trained model
model = load_model('leaf (1).h5')
labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

# Function to process image and predict
def getResult(image_path):
    try:
        img = load_img(image_path, target_size=(224, 224))  # Ensure correct input size
        x = img_to_array(img) / 255.0  # Normalize
        x = np.expand_dims(x, axis=0)

        predictions = model.predict(x)[0]
        predicted_label = labels[np.argmax(predictions)]
        return predicted_label
    except Exception as e:
        print("Error in prediction:", str(e))
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save file
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)

    # Get prediction
    predicted_label = getResult(file_path)
    if predicted_label:
        return jsonify({'prediction': predicted_label})
    else:
        return jsonify({'error': 'Prediction failed'})

if __name__ == '__main__':
    app.run(debug=False)  

