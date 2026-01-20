import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ---------------------------------------------------------
# 1. LOAD DATA
# Đảm bảo file csv nằm cùng thư mục với file code này
# ---------------------------------------------------------
print("Đang tải dữ liệu...")
train_df = pd.read_csv('mitbih_train.csv', header=None)
test_df = pd.read_csv('mitbih_test.csv', header=None)

# Tách features (187 cột đầu) và label (cột cuối)
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# ---------------------------------------------------------
# 2. MODEL BUILDING (Random Forest)
# Không cần reshape phức tạp như CNN
# ---------------------------------------------------------
print("\nĐang huấn luyện mô hình Random Forest...")

# n_estimators=100: Số lượng cây trong rừng
# n_jobs=-1: Dùng tất cả nhân CPU để chạy cho nhanh
model = RandomForestClassifier(n_estimators=100, 
                               random_state=42, 
                               n_jobs=-1,
                               class_weight='balanced') # Giúp cân bằng dữ liệu

model.fit(X_train, y_train)
print("Huấn luyện xong!")

# ---------------------------------------------------------
# 3. EVALUATION
# ---------------------------------------------------------
print("\nĐang dự đoán trên tập Test...")
y_pred = model.predict(X_test)

# Tính độ chính xác
acc = accuracy_score(y_test, y_pred)
print(f"--> Accuracy: {acc*100:.2f}%")

# Báo cáo chi tiết
print("\n--- Classification Report ---")
target_names = ['Normal', 'SVEB', 'VEB', 'Fusion', 'Unknown']
print(classification_report(y_test, y_pred, target_names=target_names))

# Vẽ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()