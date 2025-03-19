import numpy as np
import tensorflow as tf
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm  # For progress tracking

# Preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd


# Load trained model
model = tf.keras.models.load_model('phishing_detection_hybrid_model.h5')

# Load ELECTRA tokenizer & model
electra_model_name = "google/electra-small-discriminator"
electra_tokenizer = AutoTokenizer.from_pretrained(electra_model_name)
electra_model = AutoModel.from_pretrained(electra_model_name)

# Load dataset to reinitialize tokenizer
df = pd.read_csv('C:/Users/TRIPTI/Downloads/Final Year Project/proj3/url_dataset.csv')
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(df['url'])

# Define max sequence length
max_len = 200  

# Function to extract ELECTRA embedding
def get_electra_embedding(url):
    inputs = electra_tokenizer(url, return_tensors="pt", padding=True, truncation=True, max_length=50)
    with torch.no_grad():
        outputs = electra_model(**inputs)
    return np.squeeze(outputs.last_hidden_state[:, 0, :].numpy(), axis=0)  # ✅ Remove extra dim (Fix shape issue)

# Function to preprocess URL for BiLSTM
def preprocess_url(url):
    sequence = tokenizer.texts_to_sequences([url])
    return pad_sequences(sequence, maxlen=max_len, padding='post')

# Prediction function
def predict_url(url):
    # Get BiLSTM input
    url_seq = preprocess_url(url)

    # Get ELECTRA input (Fix shape issue)
    electra_features = np.expand_dims(get_electra_embedding(url), axis=0)  # ✅ Ensure correct shape (1, 256)

    # Predict using both inputs
    prediction = model.predict([url_seq, electra_features])[0][0]

    return "Phishing" if prediction >= 0.5 else "Legitimate"

# Example usage
url_input = input("Enter a URL to check: ")
result = predict_url(url_input)
print(f"🔍 Prediction: {result}")
