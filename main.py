# Supply and Transfer Monitoring

# checks supply speed, loss of packages and sends a recommendation to change processing

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# 1. Daten-Simulation (Warenstrom im Lagerhaus)
# Wir simulieren eine Sinus-Welle (Tagesrhythmus der Pakete)
time = np.linspace(0, 100, 1000)
packages = np.sin(time) + np.random.normal(0, 0.1, 1000)

# Daten für LSTM vorbereiten (Fenster-Ansatz)
def create_sequences(data, window=10):
    X, y = [], []
    for i in range(len(data)-window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

X, y = create_sequences(packages)
X = X.reshape((X.shape[0], X.shape[1], 1))

# 2. Modell-Setup (Deep Learning für Zeitreihen)
model = Sequential([
    LSTM(50, activation='relu', input_shape=(10, 1)),
    Dense(1)
])
model.compile(optimizer='sgd', loss='mse')

# 3. Training (In Colab schnell erledigt)
model.fit(X, y, epochs=5, verbose=0)
print("Modell für Mujin-Durchsatzvorhersage ist bereit.")
