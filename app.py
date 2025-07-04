import nltk
import numpy as np
import json
import random
import os 

from nltk.stem import WordNetLemmatizer
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Dropout
try:
    nltk.data.find('tokenizers/punkt')
except LookupError: 
    print("Downloading NLTK 'punkt' tokenizer...")
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK 'wordnet' corpus...")
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError: 
    print("Downloading NLTK 'punkt_tab' tokenizer...")
    nltk.download('punkt_tab')


# NLP Preprocessing
lemmatizer = WordNetLemmatizer()
words = []       
classes = []     
documents = []   
ignore_letters = ['?', '!', '.', ',']

# Load intents from the JSON file
try:
    with open('intents.json', encoding='utf-8') as file:
        intents = json.load(file)
except FileNotFoundError:
    print("Error: 'intents.json' not found. Please make sure it exists in the same directory.")
    exit() 
except json.JSONDecodeError:
    print("Error: Could not decode 'intents.json'. Please check its format.")
    exit()

# Process intents to build vocabulary and documents
for intent in intents['intents']:
    for pattern in intent['patterns']:
       
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list) 
        documents.append((word_list, intent['tag'])) 
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words))) 
classes = sorted(list(set(classes))) 

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = [] 
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    bag = [1 if w in word_patterns else 0 for w in words]

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training) 
training = np.array(training, dtype=object)

# Separate features (X) and labels (Y)
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Model training or loading
model_filename = "chatbot_model.h5" # Filename for saving/loading the model

# Check if a pre-trained model exists
if os.path.exists(model_filename):
    model = load_model(model_filename)
    print("Loaded existing model from 'chatbot_model.h5'.")
else:
    # If no model exists, create and train a new one
    print("No existing model found. Training a new model...")
    model = Sequential()
    # Input layer with ReLU activation and dropout for regularization
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    # Hidden layer with ReLU activation and dropout
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    # Output layer with softmax activation for multi-class classification
    model.add(Dense(len(train_y[0]), activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Train the model
    history = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

    # Save the trained model for future use
    save_model(model, model_filename)
    print("Trained and saved the model to 'chatbot_model.h5'.")

    # plt.figure(figsize=(12, 6))
    # plt.plot(history.history['accuracy'], label='Training Accuracy')
    # plt.plot(history.history['loss'], label='Training Loss')
    # plt.title('Model Training History')
    # plt.xlabel('Epoch')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.show()


# Chatbot functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    try:
        bow = bag_of_words(sentence)
        res = model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25 
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    except Exception as e:
        print(f"Error in predict_class: {e}")
        return [] 

def get_response(intents_list, intents_json):
    try:
        if not intents_list:
            return "Sorry, I didn't understand that." 
        tag = intents_list[0]['intent'] 
        for i in intents_json['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses'])
        return "Sorry, I didn't find a suitable response for that intent." 
    except Exception as e:
        print(f"Error in get_response: {e}")
        return "An error occurred while processing the response."

def chatbot_response(text):
    try:
        ints = predict_class(text)
        return get_response(ints, intents)
    except Exception as e:
        print(f"Error in chatbot_response: {e}")
        return "An internal error occurred. Please try again later."


# Flask App
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["GET"])
def get_bot_response():
    user_text = request.args.get('msg')
    response = chatbot_response(user_text)
    return jsonify({"response": response})


@app.route("/plot_data")
def plot_data():
    tag_freq = {'greeting': 20, 'goodbye': 15, 'thanks': 10, 'technical_issue': 30, 'billing_inquiry': 25}
    intents_list = list(tag_freq.keys())
    frequency = list(tag_freq.values())
    return jsonify({'intents': intents_list, 'frequency': frequency})


if __name__ == "__main__":
    app.run(debug=True)
