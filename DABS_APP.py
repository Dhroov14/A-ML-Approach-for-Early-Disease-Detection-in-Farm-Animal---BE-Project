import tkinter as tk
from tkinter import Text, Scrollbar, Button, END, Label, filedialog, messagebox, StringVar, OptionMenu
import numpy as np
import joblib
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import pickle
from PIL import Image, ImageTk
import nltk
from nltk.stem import WordNetLemmatizer
import random
import json

# Load Machine Learning Models
model_files = {
    "SVC": "SVC_Cow_Model.pkl",
    "Naive Bayes": "Naive Bayes_Cow_Model.pkl",
    "Random Forest": "Random Forest_Cow_Model.pkl"
}
available_models = {name: joblib.load(file) for name, file in model_files.items() if os.path.exists(file)}

# Initialize NLP Tools
lemmatizer = WordNetLemmatizer()
selected_language = "English"

# Function to Load Chatbot Model & Data Based on Language
def load_chatbot(language):
    model_file = next((f for f in os.listdir() if f.startswith(f"chatbot_model_{language.lower()}") and f.endswith(".h5")), None)
    if model_file:
        chatbot_model = tf.keras.models.load_model(model_file)
        words, classes = pickle.load(open(f"words_{language.lower()}.pkl", "rb"))
        intents = json.load(open(f"intents_{language.lower()}.json", encoding="utf-8"))
        return chatbot_model, words, classes, intents
    else:
        messagebox.showerror("Error", f"Chatbot model for {language} not found!")
        return None, None, None, None

# Load Default Chatbot Data
chatbot_model, words, classes, intents = load_chatbot(selected_language)

def change_language(*args):
    global chatbot_model, words, classes, intents, selected_language
    selected_language = language_var.get()
    chatbot_model, words, classes, intents = load_chatbot(selected_language)
    messagebox.showinfo("Language Changed", f"Selected Language: {selected_language}")

def process_image(image_path):
    image = load_img(image_path, target_size=(64, 64))
    image = img_to_array(image) / 255.0
    return image.flatten().reshape(1, -1)

def classify_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png")])
    if not file_path:
        return

    image = Image.open(file_path).resize((250, 250))
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo

    input_data = process_image(file_path)
    predictions, confidence_scores = [], []
    for model_name, model in available_models.items():
        pred = model.predict(input_data)[0]
        predictions.append(pred)
        if hasattr(model, "predict_proba"):
            confidence_scores.append(np.max(model.predict_proba(input_data)) * 100)

    final_prediction = round(np.mean(predictions))
    avg_confidence = np.mean(confidence_scores) if confidence_scores else None
    result_text = "Diseased (Lumpy Skin Disease)" if final_prediction == 1 else "Healthy"
    result_label.config(text=f"Diagnosis: {result_text}\nConfidence: {avg_confidence:.2f}%")

def predict_class(sentence, model):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    bag = [1 if word in sentence_words else 0 for word in words]
    res = model.predict(np.array([bag]))[0]
    results = [[i, r] for i, r in enumerate(res) if r > 0.25]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intent):
    for i in intents['intents']:
        if i['tag'] == intent:
            return random.choice(i['responses'])
    return "I'm not sure."

def chatbot_response(msg):
    if not chatbot_model:
        return "Chatbot model not loaded."
    ints = predict_class(msg, chatbot_model)
    return get_response(ints[0]['intent']) if ints else "I'm not sure."

def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)
    if msg:
        ChatLog.config(state=tk.NORMAL)
        ChatLog.insert(tk.END, "You: " + msg + '\n\n')
        res = chatbot_response(msg)
        ChatLog.insert(tk.END, "Bot: " + res + '\n\n')
        ChatLog.config(state=tk.DISABLED)
        ChatLog.yview(tk.END)

# Initialize GUI
root = tk.Tk()
root.title("Lumpy Skin Disease Detection & Chatbot")
root.geometry("600x650")

# Language Selector
language_var = StringVar(root)
language_var.set("English")
languages = ["English", "Marathi"]
language_menu = OptionMenu(root, language_var, *languages, command=change_language)
language_menu.pack()

# Chatbot Section
ChatLog = Text(root, bd=0, bg="white", height=10, width=50, font="Arial")
ChatLog.config(state=tk.DISABLED)
scrollbar = Scrollbar(root, command=ChatLog.yview)
ChatLog['yscrollcommand'] = scrollbar.set
EntryBox = Text(root, bd=0, bg="white", width=40, height=3, font="Arial")
SendButton = Button(root, font=("Verdana", 12, 'bold'), text="Send", width=12, height=3, bg="#25cdf7", fg='#ffffff', command=send)

# Image Classification Section
image_label = Label(root, text="Upload an image")
image_label.pack()
classify_button = Button(root, text="Classify Image", command=classify_image)
classify_button.pack()
result_label = Label(root, text="")
result_label.pack()

# Layout Placement
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
ChatLog.pack()
EntryBox.pack()
SendButton.pack()

root.mainloop()
