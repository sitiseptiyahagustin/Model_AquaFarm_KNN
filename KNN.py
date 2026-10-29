import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Correct method to read Excel files
data = pd.read_excel('C:/Users/septi/OneDrive/Documents/KULIAH/SKRIPSI/Project Skrips/Project Skrips/Extraksi Fitur Jamur.xlsx')

# Display the actual column names in the DataFrame to verify
print("Columns in the dataset:")
print(data.columns)

# Check the number of columns
print(f"Number of columns: {len(data.columns)}")

# Rename the columns if necessary
data.columns = ['Warna R', 'Warna G', 'Warna B', 'GLCM Contrast', 'GLCM Homogeneity', 'GLCM Energy', 'GLCM Entropy', 'kondisi', 'jenis jamur']  # Adjust as necessary

# Check the unique values in the 'kondisi' column
print("Unique values in 'kondisi':", data['kondisi'].unique())

# Ensure that 'kondisi' contains valid labels (no need to map since it's already categorical)
# If you want to map to 'Keruh' and 'Jernih', you can, but since your goal is to predict "Beracun" or "Tidak Beracun", leave as is.

# 2. Ekstrak fitur RGB serta label kondisi
# Here, the 'kondisi' will be used as the target variable for classification
rgb_features = data[['Warna R', 'Warna G', 'Warna B', 'GLCM Contrast', 'GLCM Homogeneity', 'GLCM Energy', 'GLCM Entropy']].values
labels = data['kondisi'].values  # 'kondisi' is the target (Beracun or Tidak Beracun)

# 3. Pisahkan data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(rgb_features, labels, test_size=0.3, random_state=42)

# Check for missing values
print("Missing values in each column:")
print(data.isna().sum())

# Handle missing values (if any)
data = data.dropna()  # Or fill missing values with data.fillna() if needed

# 4. Normalisasi data untuk meningkatkan performa KNN
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Inisialisasi model KNN dan latih model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 6. Prediksi dan evaluasi model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Menghitung probabilitas prediksi untuk menghitung log_loss
y_pred_proba = knn.predict_proba(X_test)

# Menghitung log loss
loss = log_loss(y_test, y_pred_proba)

# Menampilkan akurasi
print("Accuracy:", accuracy)

# Menampilkan log loss
print("Log Loss:", loss)

# Menghitung dan menampilkan matriks konfusi
print("\nConfusion Matrix:")
print(conf_matrix)

# Menampilkan laporan klasifikasi (precision, recall, f1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Beracun', 'Tidak Beracun']))

# 7. Tampilkan Confusion Matrix dalam bentuk heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Beracun', 'Tidak Beracun'], yticklabels=['Beracun', 'Tidak Beracun'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix of KNN Model")
plt.show()

# 8. Fungsi untuk memprediksi kondisi air dari gambar
# 8. Fungsi untuk memprediksi kondisi air dari gambar
def predict_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return None
    
    # Resize the image to 224x224 as expected by the model
    resized_image = cv2.resize(image, (224, 224))  
    
    # Calculate the average RGB values
    avg_color_per_row = np.average(resized_image, axis=0)
    avg_rgb = np.average(avg_color_per_row, axis=0)
    
    # Add dummy GLCM values to make the feature vector length equal to 7
    # Dummy values for GLCM features, assuming you don't have them
    features = np.concatenate([avg_rgb, [0, 0, 0, 0]])  # Add dummy GLCM features
    
    # Reshape the feature vector to be in the same shape as the training data
    features = features.reshape(1, -1)

    # Normalize the features using the same scaler
    features = scaler.transform(features)
    
    # Get the prediction from the trained model
    prediction = knn.predict(features)
    return prediction[0]

# 9. Contoh prediksi dari gambar input
image_path = 'C:/Users/septi/OneDrive/Documents/KULIAH/SKRIPSI/arsip/Jamur trucuk (8).jpg'  # Ganti dengan path gambar Anda
prediksi_kondisi = predict_image(image_path)
print("Prediksi kondisi jamur:", prediksi_kondisi)
