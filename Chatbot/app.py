#!/usr/bin/env python
# coding: utf-8

# In[8]:


import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model 
model = tf.keras.models.load_model(r"C:\Users\Asus\OneDrive\Desktop\Chatbot\chatbot_model.keras") 

# Load the saved tokenizer 
with open(r"C:\Users\Asus\OneDrive\Desktop\Chatbot\tokenizer.pkl", 'rb') as f:
    tokenizer = pickle.load(f)

max_len = 125 

# Define the prediction function
def predict_response(model, tokenizer, input_text, max_len):
    """
    Predicts the response for a given input text using the trained model.
    """
    # Tokenize and pad the input text
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_padded = pad_sequences(input_sequence, maxlen=max_len, padding='post')

    # Generate prediction
    prediction = model.predict(input_padded, verbose=0)
    predicted_indices = prediction.argmax(axis=-1).flatten()

    # Reverse the word index to map token indices to words
    reverse_word_map = {index: word for word, index in tokenizer.word_index.items()}
    response = " ".join([reverse_word_map.get(idx, '') for idx in predicted_indices if idx != 0])

    return response

# Set up the Streamlit interface
st.title("AI Chatbot 🤖")
st.write("Chat with the AI!", " \n Type your message below:")

# Text input for the user to type their message
user_input = st.text_input("You: ", "")

if user_input:
    response = predict_response(model, tokenizer, user_input, max_len)
    st.write(f"Chatbot: {response}")



# In[ ]:




