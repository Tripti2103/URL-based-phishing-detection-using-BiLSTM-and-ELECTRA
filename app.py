from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
import pandas as pd
from flask_cors import CORS

app = Flask(__name__, template_folder="templates")  # Set template folder
CORS(app)  # Allow frontend to access backend

# Load Trained Model
model = tf.keras.models.load_model('phishing_detection_model.h5')

# Load Dataset and Initialize Tokenizer
df = pd.read_csv('C:/Users/TRIPTI/Downloads/Final Year Project/proj3/url_dataset.csv')
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(df['url'])

# Set Maximum URL Length
max_len = 200  

# ✅ Route to Serve Home Page
@app.route('/')
def home():
    return render_template("index.html")

# ✅ API Route for URL Prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    url = data.get('url', '')

    # Tokenize & Pad URL
    sequence = tokenizer.texts_to_sequences([url])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')

    # Predict Phishing or Legitimate
    prediction = model.predict(padded_sequence)[0][0]
    result = "Phishing" if prediction >= 0.5 else "Legitimate"

    return jsonify({'result': result})

# ✅ Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
