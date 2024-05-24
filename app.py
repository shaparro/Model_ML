from sklearn.linear_model import Perceptron
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Wczytanie danych
data = load_iris()
X = data.data
y = data.target

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Trenowanie modelu
model = Perceptron()
model.fit(X_train, y_train)

# Ocena modelu
predictions = model.predict(X_test)
print("Dokładność modelu perceptronu:", accuracy_score(y_test, predictions))

# Zapis modelu
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Ładowanie modelu
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Przekształć dane wejściowe do odpowiedniej formy
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
