# Login Anomaly Detector

# creates a csv of each login with email address, time and date and activity and alerts if it is very different from previous sessions

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# 1. Daten-Simulation (HENNGE One Access Logs)
# Merkmale: [Stunde (0-23), Anzahl_Fehlversuche, IP_Risiko_Score (1-10)]
normal_logins = np.random.normal(loc=[14, 0, 1], scale=[3, 0.5, 0.5], size=(100, 3))
attack_logins = [[3, 15, 9], [4, 12, 8], [2, 20, 10]] # Nachts, viele Versuche, hohe Risiko-IP

X = np.vstack([normal_logins, attack_logins])

# 2. Modell-Setup (Isolation Forest findet Ausreißer ohne Training auf Labels)
model = IsolationForest(contamination=0.03, random_state=42)
model.fit(X)

# 3. Vorhersage (-1 bedeutet Anomalie/Angriff)
predictions = model.predict(X)

print(f"Gefundene Anomalien an den letzten Positionen: {predictions[-3:]}")
