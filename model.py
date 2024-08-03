### import all the neccessary libries
import tensorflow as tf 
import pandas as pd
import numpy as np

import joblib, os
from tensorflow import keras 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.corpus import stopwords

## download neccesarry resoucres
nltk.download('stopwords')

## loading the dataset 
data = pd.read_csv('data/Symptom2Disease.csv') 

def remove_digits(text):
    pattern = r'[^a-zA-z.,!?/:;\"\'\s]' 
    #when the ^ is on the inside of []; we are matching any character that is not included in this expression within the []
    return re.sub(pattern, '', text)

def remove_special_characters(text):
    pattern = r'[^a-zA-z0-9.,!?/:;\"\'\s]'   # define the pattern to keep
    return re.sub(pattern, '', text)

def non_ascii(s):
  return "".join(i for i in s if ord(i)<128)

def non_ascii(s):
  return "".join(i for i in s if ord(i)<128)

def lower(text):
  return text.lower()

## function to remove stop words. 
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

def clean_data(df, col): 
    df[col] = df[col].apply(remove_digits)
    df[col] = df[col].apply(remove_special_characters)
    df[col] = df[col].apply(non_ascii)
    df[col] = df[col].apply(lower)
    return df

df = clean_data(data, 'text')

## breaking the columns into text for training and data labels 
labels = df['label']
text = df['text']

## encoding the labels. 
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)

### removing stop words before breaking into test and training data set. 
text = text.apply(remove_stopwords)

X_train, X_test, y_train, y_test = train_test_split(text, labels, test_size=0.2, random_state=42,)

## further preprocessing of our training dataset
tokenize = Tokenizer(num_words=10000, oov_token='<00V>')
tokenize.fit_on_texts(X_train)
X_train = tokenize.texts_to_sequences(X_train)
word_indexes = tokenize.word_index

# Determine maxlen from training data
max_len = max(len(seq) for seq in X_train)
X_train_padded = pad_sequences(X_train, maxlen=max_len, padding='post')

# further preprocessing for our test data 
X_test = tokenize.texts_to_sequences(X_test)

# Determine maxlen from training data
max_len = max(len(seq) for seq in X_test)
X_test_padded = pad_sequences(X_test, maxlen=max_len, padding='post')

## let's define some variables 

embedding_dimession = 50
input_dim = len(word_indexes)
ouput_layer = len(encoder.classes_)
#modeling
model = keras.Sequential([
    keras.layers.Embedding(10000, embedding_dimession),
    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    #keras.layers.LSTM(64, return_sequences=False),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(ouput_layer, activation='softmax')
])

## compiling the model 
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True)

history = model.fit(
    X_train_padded, y_train,
    epochs = 100, 
    validation_data = (X_test_padded, y_test),
    callbacks = [early_stopping],
    batch_size=512,
    verbose = 0
)

predictions = model.predict(X_test_padded)

# Convert probabilities to class labels
predicted_labels_ = np.argmax(predictions, axis=1)

# Inverse transform the predicted labels
predicted_labels = encoder.inverse_transform(predicted_labels_)

## checking the accuracy of my model 
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
accuracy = accuracy_score(y_test, predicted_labels_)
precision = precision_score(y_test, predicted_labels_, average='weighted')
recall = recall_score(y_test, predicted_labels_, average='weighted')
f1 = f1_score(y_test, predicted_labels_, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

## make directory 
os.makedirs('model', exist_ok=True)

## dump model using joblib 
joblib.dump(model, 'model/disease_model.pkl')

joblib.dump(encoder, 'model/encoder.pkl')
joblib.dump(tokenize, 'model/tokenizer.pkl')
joblib.dump(max_len, 'model/max_len.pkl')