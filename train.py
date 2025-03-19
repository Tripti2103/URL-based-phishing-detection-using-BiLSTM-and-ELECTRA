import os
import pandas as pd
import numpy as np
import tensorflow as tf
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm  # For progress tracking

# Model and Layers
Model = tf.keras.models.Model
Input = tf.keras.layers.Input
Embedding = tf.keras.layers.Embedding
LSTM = tf.keras.layers.LSTM
Bidirectional = tf.keras.layers.Bidirectional
Conv1D = tf.keras.layers.Conv1D
MaxPooling1D = tf.keras.layers.MaxPooling1D
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Concatenate = tf.keras.layers.Concatenate

# Preprocessing
Tokenizer = tf.keras.preprocessing.text.Tokenizer
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

# Load Dataset
print("📥 Loading dataset...")
df = pd.read_csv('C:/Users/TRIPTI/Downloads/Final Year Project - 1/proj3/url_dataset.csv')
print("✅ Dataset loaded! First 5 rows:\n", df.head())

# Encode Labels
print("🔄 Encoding labels...")
label_encoder = LabelEncoder()
df['type'] = label_encoder.fit_transform(df['type'])
print("✅ Labels encoded. Unique values:", np.unique(df['type']))

# Tokenize URLs (Character-Level)
print("🔡 Tokenizing URLs...")
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(df['url'])
sequences = tokenizer.texts_to_sequences(df['url'])
max_len = 200
X_seq = pad_sequences(sequences, maxlen=max_len, padding='post')
print("✅ Tokenization complete. Sample sequence:", X_seq[:2])

# Load ELECTRA Model
print("🔄 Loading ELECTRA model...")
electra_model_name = "google/electra-small-discriminator"
electra_tokenizer = AutoTokenizer.from_pretrained(electra_model_name)
electra_model = AutoModel.from_pretrained(electra_model_name)
print("✅ ELECTRA model loaded!")

# Function to Get ELECTRA Embeddings
def get_electra_embedding(url):
    inputs = electra_tokenizer(url, return_tensors="pt", padding=True, truncation=True, max_length=50)
    with torch.no_grad():
        outputs = electra_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Extract CLS token embedding

# Load or Compute ELECTRA Features
if os.path.exists("electra_embeddings.npy"):
    print("📂 Loading precomputed ELECTRA embeddings...")
    X_electra = np.load("electra_embeddings.npy")
    print("✅ ELECTRA embeddings loaded successfully! Shape:", X_electra.shape)
else:
    print("⏳ Extracting ELECTRA embeddings (this may take time)...")
    X_electra = np.array([get_electra_embedding(url) for url in tqdm(df['url'], desc="Processing URLs")])
    print("✅ ELECTRA embeddings extracted. Shape:", X_electra.shape)

    # Save embeddings for future runs
    np.save("electra_embeddings.npy", X_electra)
    print("💾 ELECTRA embeddings saved to 'electra_embeddings.npy'!")

# ✅ Fix the shape issue
print("🔄 Reshaping ELECTRA embeddings...")
X_electra = np.squeeze(X_electra, axis=1)  # Removes the extra (1) dimension if needed
print("✅ New ELECTRA shape:", X_electra.shape)

# Combine Inputs
y = df['type'].values

# Split Dataset
X_seq_train, X_seq_test, X_electra_train, X_electra_test, y_train, y_test = train_test_split(
    X_seq, X_electra, y, test_size=0.2, random_state=42
)

# Hybrid Model Definition
def create_hybrid_model(input_dim_seq, input_dim_electra, embedding_dim, max_len):
    # **BiLSTM Branch**
    input_seq = Input(shape=(max_len,))
    embedding = Embedding(input_dim=input_dim_seq, output_dim=embedding_dim, input_length=max_len)(input_seq)
    bilstm = Bidirectional(LSTM(64, return_sequences=True))(embedding)
    conv1d = Conv1D(64, 3, activation='relu')(bilstm)
    pooling = MaxPooling1D(pool_size=2)(conv1d)
    lstm_flat = Flatten()(pooling)

    # **ELECTRA Embedding Branch**
    input_electra = Input(shape=(input_dim_electra,))
    electra_dense = Dense(128, activation='relu')(input_electra)

    # **Merge Both Outputs**
    merged = Concatenate()([lstm_flat, electra_dense])
    merged = Dense(128, activation='relu')(merged)
    merged = Dropout(0.5)(merged)
    merged = Dense(64, activation='relu')(merged)
    merged = Dropout(0.5)(merged)

    # **Final Output**
    output = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[input_seq, input_electra], outputs=output)
    return model

# Define Model Parameters
input_dim_seq = len(tokenizer.word_index) + 1
input_dim_electra = X_electra.shape[1]
embedding_dim = 128

# Create Model
print("🔧 Building model...")
model = create_hybrid_model(input_dim_seq, input_dim_electra, embedding_dim, max_len)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
print("✅ Model built successfully!")

# Custom Callback to Track Each Epoch
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n🚀 Starting Epoch {epoch + 1}...")

    def on_epoch_end(self, epoch, logs=None):
        print(f"✅ Epoch {epoch + 1} Complete - Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}")

# Train Model with Progress Tracking
print("🏋️‍♂️ Training the model...")
history = model.fit(
    [X_seq_train, X_electra_train], y_train,
    epochs=10, batch_size=32, validation_split=0.2,
    verbose=1, callbacks=[CustomCallback()]
)

# Evaluate Model on Test Data
print("📊 Evaluating model on test data...")
loss, accuracy = model.evaluate([X_seq_test, X_electra_test], y_test)
print(f"📉 Test Loss: {loss:.4f}")
print(f"🎯 Test Accuracy: {accuracy:.4f}")

# Save the Model
model.save('phishing_detection_hybrid_model.h5')
print("✅ Model saved successfully!")
