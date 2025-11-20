import nltk
from nltk.stem import WordNetLemmatizer
from indicnlp.tokenize import indic_tokenize
import json
import pickle
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import datetime
import os

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def load_intents(language):
    file_name = f'intents_{language.lower()}.json'
    if os.path.exists(file_name):
        with open(file_name, encoding='utf-8') as file:
            return json.load(file)
    else:
        raise FileNotFoundError(f"{file_name} not found!")

def preprocess_data(language):
    words, classes, documents = [], [], []
    ignore_words = ['?', '!', '.', ',']

    intents = load_intents(language)
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            if any('\u0900' <= ch <= '\u097F' for ch in pattern):
                w = indic_tokenize.trivial_tokenize(pattern)  # Marathi Tokenization
            else:
                w = nltk.word_tokenize(pattern)  # English Tokenization

            w = [lemmatizer.lemmatize(word.lower()) for word in w if word not in ignore_words]
            words.extend(w)
            documents.append((w, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = sorted(set(words))
    classes = sorted(set(classes))

    # Save language-specific words and classes
    with open(f'words_{language.lower()}.pkl', 'wb') as f:
        pickle.dump((words, classes), f)

    return words, classes, documents

def prepare_training_data(words, classes, documents):
    training = []
    output_empty = [0] * len(classes)

    for doc in documents:
        bag = [1 if w in doc[0] else 0 for w in words]
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype=object)
    return np.array(list(training[:, 0])), np.array(list(training[:, 1]))

def train_chatbot(train_x, train_y, language):
    model = Sequential([
        Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(len(train_y[0]), activation='softmax')
    ])

    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(f'chatbot_model_{language.lower()}_{timestamp}.h5')
    print(f"Model saved as chatbot_model_{language.lower()}_{timestamp}.h5")

# Train for both languages
for lang in ["English", "Marathi"]:
    print(f"Processing {lang} data...")
    words, classes, documents = preprocess_data(lang)
    train_x, train_y = prepare_training_data(words, classes, documents)
    train_chatbot(train_x, train_y, lang)
