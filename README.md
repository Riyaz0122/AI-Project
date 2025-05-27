# 📘 AI Project Documentation

## 📟 Table of Contents
1. Project contant
2. Project code
3. Key technologies
3. Description
4. Output 
5. Further research 
---

# 🐶🐱 Dog and Cat Classification
## 📌 Project Content
```
This script is designed to mount Google Drive in a Google Colab environment and then load and display images from a specific folder in your Drive.
Drive Mounting:
drive.mount('/content/drive') connects your Google Drive to the Colab workspace, allowing access to files stored there.
Folder Path:
The variable folder_path points to the directory containing dog images (/content/drive/MyDrive/dogs).
Image Listing:
It scans the folder for image files with extensions .jpg, .jpeg, and .png.
Image Loading & Display:
The script loads up to the first 50 images, resizing each to 200x200 pixels using Keras’ load_img function. Each image is then displayed one by one using Matplotlib, with the filename shown as the title.
```

## 🛠 Code
```python
from google.colab import drive
drive.mount('/content/drive')

code for Dog:
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
folder_path = '/content/drive/MyDrive/dogs'
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
for i in range(min(50, len(image_files))):
    img_path = os.path.join(folder_path, image_files[i])
    img = image.load_img(img_path, target_size=(200, 200))
    plt.imshow(img)
    plt.title(f"Image: {image_files[i]}")
    plt.axis('off')
    plt.show()
```
## 🌟 Output:
![image](https://github.com/user-attachments/assets/19436fa3-47f9-4cae-896e-3f83761b28f5)

code for cat:
```python
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
folder_path = '/content/drive/MyDrive/cats'
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
for i in range(min(50, len(image_files))):
    img_path = os.path.join(folder_path, image_files[i])
    img = image.load_img(img_path, target_size=(200, 200))
    plt.imshow(img)
    plt.title(f"Image: {image_files[i]}")
    plt.axis('off')
    plt.show()
 ```
## 🌟 Output:
![image](https://github.com/user-attachments/assets/599cf87f-368e-4b18-a4a2-ab6501383b07)


## 🚀 Key Technologies
```
Google Colab
Cloud-based Jupyter notebook environment that supports free GPU/TPU usage and easy integration with Google Drive.
Google Drive API (via google.colab.drive)
Used to mount and access Google Drive files directly within Colab.
Python os Module
For file and directory operations like listing image files.
Matplotlib
A popular Python library for creating static, animated, and interactive visualizations.
TensorFlow Keras Preprocessing
Specifically tensorflow.keras.preprocessing.image.load_img for loading and resizing images easily in deep learning workflows.

## 📌 Description
This script enables you to quickly preview a collection of images stored in a Google Drive folder when working in a Google Colab environment. After mounting your Google Drive, it scans a specified directory for common image file formats (.jpg, .jpeg, .png). It then loads each image, resizes it to a uniform size (200x200 pixels), and displays it using Matplotlib.
This visual inspection step is crucial for verifying dataset contents before proceeding with tasks like model training or data preprocessing. It helps identify any corrupted files, mislabeled images, or inconsistencies in the dataset, improving the overall quality of your machine learning pipeline.
```
## 🚀 Further research
```
Advanced Image Preprocessing:
Explore additional preprocessing techniques such as normalization, data augmentation (flipping, rotation, zoom), and color adjustments to improve model robustness.
Automated Dataset Validation:
Implement scripts to automatically detect and flag corrupted or mislabeled images, helping to clean large datasets without manual inspection.
Batch Visualization:
Create grid views or interactive galleries to preview many images simultaneously rather than one by one, improving dataset exploration efficiency.
Integration with Annotation Tools:
Combine with image annotation tools or labeling platforms to streamline dataset preparation and ground-truth labeling.
Model Integration:
Extend this visualization pipeline to include real-time model predictions on displayed images, assisting in debugging and understanding model performance.
Cross-Platform Dataset Access:
Research ways to mount and visualize datasets stored on other cloud platforms such as AWS S3, Azure Blob Storage, or local servers.
Performance Optimization:
Investigate efficient loading and rendering methods for very large datasets, including caching and lazy loading..

```

# 🏥 Healthcare XAI

### Project Content
```
### Project Content
This project focuses on analyzing a healthcare dataset to predict key medical outcomes such as test results. It utilizes machine learning techniques including logistic regression and random forest, combined with model interpretability tools like LIME, SHAP, and ELI5 for deep insights.`, `openai`, `webrtcvad`
```

###Code

```python
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
# Load dataset
df = pd.read_csv("healthcare_dataset.csv")
# Drop 'Loan_ID' column
df.drop(columns=['Hospital'], inplace=True)
df
```
## 🌟 Output:
![image](https://github.com/user-attachments/assets/25f83920-4d71-4460-956a-165cb8b6c3b8)

```python
numerical_cols = ['Room Number', 'Billing Amount', 'Age']
for col in numerical_cols:
    df[col].fillna(df[col].median(), inplace=True)
# Example categorical to numeric conversion
df.replace({
    'Gender': {'Male': 0, 'Female': 1},
    'Admission Type': {'Emergency': 0, 'Urgent': 1, 'Elective': 2},
    'Test Results': {'Normal': 0, 'Abnormal': 1, 'Inconclusive': 2},
    'Blood Type': {'O+': 0, 'A+': 1, 'B+': 2, 'AB+': 3, 'O-': 4, 'A-': 5, 'B-': 6, 'AB-': 7}
}, inplace=True)
# Define the target column
target_col = 'Billing Amount'

# Split the dataset
X = df.drop(columns=[target_col])
y = df[target_col]

# Print first 5 rows of features and target
print(X.head())
print(y.head())
```
## 🌟 Output:
![image](https://github.com/user-attachments/assets/6eddde46-ff2c-4933-a424-800fe6d5a427)

```python

from sklearn.model_selection import train_test_split
# Split the data (90% train, 10% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# Optional: print the shapes of the resulting splits
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
y_test
```
## 🌟 Output:
![image](https://github.com/user-attachments/assets/fbfe8563-7dc2-464a-baf4-aacfd43d5570)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# Example: Let's say we want to predict 'Test Results'
# Step 1: Encode the target variable
le = LabelEncoder()
df['Test Results'] = le.fit_transform(df['Test Results'])  # e.g., Normal=1, Abnormal=0, etc.
# Step 2: Select features (dropping non-numeric or irrelevant columns for now)
features = ['Age', 'Gender', 'Blood Type', 'Medical Condition', 'Billing Amount']
X = pd.get_dummies(df[features], drop_first=True)
y = df['Test Results']
# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 4: Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
pip install lime
from sklearn.metrics import accuracy_score, classification_report

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the predictions
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)
```
## 🌟 Output:
![image](https://github.com/user-attachments/assets/b082acbe-1ba3-438c-8ff6-df04ce1a0bc6)

![image](https://github.com/user-attachments/assets/13a61eab-829e-4bbf-8059-028870d69512)

```python




