from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
nltk.download('stopwords')

app = Flask(__name__)

# Load your trained model
model = joblib.load('model/disease_model.pkl')

# Load your tokenizer and label encoder
tokenizer = joblib.load('model/tokenizer.pkl')
label_encoder = joblib.load('model/encoder.pkl')
maxlen = joblib.load('model/max_len.pkl')

# Define your preprocessing functions
def remove_digits(text):
    pattern = r'[^a-zA-z.,!?/:;\"\'\s]'
    return re.sub(pattern, '', text)

def remove_special_characters(text):
    pattern = r'[^a-zA-z0-9.,!?/:;\"\'\s]'
    return re.sub(pattern, '', text)

def non_ascii(s):
    return "".join(i for i in s if ord(i) < 128)

def lower(text):
    return text.lower()

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

def clean_text(text):
    text = remove_digits(text)
    text = remove_special_characters(text)
    text = non_ascii(text)
    text = lower(text)
    text = remove_stopwords(text)
    return text

# Preprocess input text
def preprocess_input(text):
    text = clean_text(text)
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=maxlen, padding='post')  
    return padded

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']
    
    # Preprocess the text
    processed_text = preprocess_input(user_input)
    
    # Predict the class
    predictions = model.predict(processed_text)
    predicted_label = np.argmax(predictions, axis=1)
    predicted_class = label_encoder.inverse_transform(predicted_label)[0]
    
    return jsonify({'response': f'I think it might be {predicted_class}'})

if __name__ == '__main__':
    app.run(debug=True)
