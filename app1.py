from flask import Flask, request, jsonify, render_template
from datasets import load_dataset
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

app = Flask(__name__)

# Load the dataset
file_path = r"C:\Users\mervi\OneDrive\Desktop\TASK-03\archive (27)\IMDB Dataset.csv"
dataset = load_dataset("csv", data_files=file_path)
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

# Convert to pandas
train_data = pd.DataFrame(dataset["train"])
test_data = pd.DataFrame(dataset["test"])

# Map labels
label_mapping = {"negative": 0, "positive": 1}
train_data["sentiment"] = train_data["sentiment"].map(label_mapping)
test_data["sentiment"] = test_data["sentiment"].map(label_mapping)

### Logistic Regression Model ###
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(train_data["review"])
X_test_tfidf = vectorizer.transform(test_data["review"])

log_model = LogisticRegression()
log_model.fit(X_train_tfidf, train_data["sentiment"])

# Save models
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(log_model, "logistic_model.pkl")

# Load the pre-trained BERT model (TensorFlow version)
bert_model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

#save model
bert_model.save_pretrained("sentiment_model_bert", save_format="tf")


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# API Endpoint for Predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("review", "")

    if not text:
        return jsonify({"error": "No review provided"}), 400

    # Logistic Regression Prediction
    vectorizer = joblib.load("vectorizer.pkl")
    log_model = joblib.load("logistic_model.pkl")
    text_tfidf = vectorizer.transform([text])
    prediction_log = log_model.predict(text_tfidf)[0]

    # BERT Prediction
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)
    outputs = bert_model(inputs.input_ids)
    prediction_bert = tf.argmax(outputs.logits, axis=1).numpy()[0]

    return jsonify({
        "logistic_regression": "Positive" if prediction_log == 1 else "Negative",
        "bert": "Positive" if prediction_bert == 1 else "Negative"
    })

from flask import Flask, render_template, request
from textblob import TextBlob

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    if request.method == 'POST':
        review = request.form['review']
        blob = TextBlob(review)
        sentiment = blob.sentiment.polarity
        if sentiment > 0:
            sentiment = "Positive 😊"
        elif sentiment < 0:
            sentiment = "Negative 😠"
        else:
            sentiment = "Neutral 😐"
    return render_template('index.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)

