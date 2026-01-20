import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. LOAD DATA

train_df = pd.read_csv('mitbih_train.csv', header=None)
test_df = pd.read_csv('mitbih_test.csv', header=None)

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")


# 2. MODEL BUILDING (Random Forest)
print("\nTraining Random Forest model")

model = RandomForestClassifier(n_estimators=100, 
                               random_state=42, 
                               n_jobs=-1,
                               class_weight='balanced') 

model.fit(X_train, y_train)
print("Done!")


# 3. EVALUATION
print("\nPrediting Test")
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"--> Accuracy: {acc*100:.2f}%")

print("\n--- Classification Report ---")
target_names = ['Normal', 'SVEB', 'VEB', 'Fusion', 'Unknown']
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()