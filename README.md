# 🤖 Customer Support Chatbot

This project is part of my internship at **Future Interns**. It showcases a **chatbot application** built using **Natural Language Processing (NLP)** and **deep learning**, designed to handle basic customer service queries efficiently.

## 📌 Project Overview

The chatbot is trained on customer support logs and FAQ data to recognize user intents and respond accordingly. It can handle queries like order status, technical issues, billing inquiries, and more. The system is built with Python, NLTK, TensorFlow, and Flask for web integration.

## 🧠 Features

- Understands customer messages using NLP
- Classifies intents using a neural network (TensorFlow)
- Responds with predefined replies from a structured JSON file
- Web interface using Flask
- Fallback response for unrecognized queries
- Endpoint to visualize intent frequency distribution

## 🛠️ Tech Stack

- **Programming:** Python
- **NLP Libraries:** NLTK (Tokenization, Lemmatization)
- **Machine Learning:** TensorFlow (Keras)
- **Web Framework:** Flask
- **Visualization:** JavaScript (for plotting), JSON
- **Dataset:** `intents.json` + sample support logs

## 📂 Project Structure

```plaintext
├── app.py                # Main Flask application
├── chatbot_model.h5      # Trained deep learning model
├── intents.json          # Intent patterns and responses
├── customer_support_tickets.csv  # Sample dataset (not directly used in this demo)
├── templates/
│   └── index.html        # Frontend chat interface
