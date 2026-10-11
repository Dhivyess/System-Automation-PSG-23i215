import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_path = r"C:\Users\PL\Documents\System-Automation-PSG-23i215\Day-2\Data\day2_sample.csv"
df = pd.read_csv(data_path)

X = df[['temperature', 'pressure', 'vibration', 'rpm', 'current']]
y = df['state']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")