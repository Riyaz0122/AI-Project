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
Specifically tensorflow.keras.preprocessing.image.load_img for loading and resizing images easily in deep learning workflows.```

```
## 📌 Description
```
```
This script enables you to quickly preview a collection of images stored in a Google Drive folder when working in a Google Colab environment. After mounting your Google Drive, it scans a specified directory for common image file formats (.jpg, .jpeg, .png). It then loads each image, resizes it to a uniform size (200x200 pixels), and displays it using Matplotlib.
This visual inspection step is crucial for verifying dataset contents before proceeding with tasks like model training or data preprocessing. It helps identify any corrupted files, mislabeled images, or inconsistencies in the dataset, improving the overall quality of your machine learning pipeline.
```
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
Investigate efficient loading and rendering methods for very large datasets, including caching and lazy loading.
```

# 🏥 Healthcare XAI

### Project Content
```
This project focuses on analyzing a healthcare dataset to predict key medical outcomes such as test results. It utilizes machine learning techniques including logistic regression and random forest, combined with model interpretability tools like LIME, SHAP, and ELI5 for deep insights.`, `openai`, `webrtcvad`
```


### Code

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
y_pred
a=([1, 1, 0, ..., 1, 1, 0])
len(a)
y_test
```
## 🌟 Output:
![image](https://github.com/user-attachments/assets/b7ef2a2a-346d-460f-b6fe-e3131413c2a8)

```python
# Compare actual vs predicted results
comparison = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
}).reset_index(drop=True)
# Show the first 10 rows
print(comparison.head(10))
```
## 🌟 Output:
![image](https://github.com/user-attachments/assets/48caec64-7854-401a-a6e2-c2e3ca4ec55e)

```python
# Print evaluation results
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("\nClassification Report:\n")
print(report)
```
## 🌟 Output:
![image](https://github.com/user-attachments/assets/3df35638-81bd-4749-9d04-9339dd5e39f7)

```python
import pickle
with open("logistic_model.pkl", "wb") as f:
    pickle.dump((model, X.columns.tolist()), f)
!pip install shap
!pip install lime
!pip install eli5
!pip install alibi
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import lime
import lime.lime_tabular

# Drop non-informative columns
df = df.drop(columns=['Name', 'Date of Admission', 'Discharge Date', 'Doctor', 'Hospital', 'Room Number'])

# Encode categorical columns
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Split features and target
X = df.drop('Test Results', axis=1)
y = df['Test Results']

# Encode the target
y_encoded = LabelEncoder().fit_transform(y)
class_names = LabelEncoder().fit(y).classes_

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Initialize LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X.columns.tolist(),
    class_names=class_names.tolist(),
    mode='classification'
)

# Explain a test instance
i = 0
exp = explainer.explain_instance(
    data_row=X_test.iloc[i].values,
    predict_fn=model.predict_proba,
    num_features=5
)

# Show explanation (in Jupyter Notebook)
exp.show_in_notebook(show_table=True)
```
## 🌟 Output:
![image](https://github.com/user-attachments/assets/2ebdc07c-4f25-4651-9fe9-41b7f3e2ad33)

```python
import eli5
from eli5.sklearn import PermutationImportance
# Fit PermutationImportance on the model using the test set
perm = PermutationImportance(model, random_state=42).fit(X_test, y_test)
# Show weights (feature importance)
eli5.show_weights(perm, feature_names=X_test.columns.tolist())
```
## 🌟 Output:
![image](https://github.com/user-attachments/assets/6733f47f-7bfa-4042-8e53-4f383a91fcfa)


```python
import matplotlib.pyplot as plt
import numpy as np
# Get feature importances from the model
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]  # Sort by importance descending
# Plot
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center", color='skyblue')
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=45, ha='right')
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()
```
## 🌟 Output:
![image](https://github.com/user-attachments/assets/e51a3339-ae69-446a-9821-611b5022e954)

```python
from alibi.explainers import ALE
from alibi.explainers.ale import plot_ale
import numpy as np
import matplotlib.pyplot as plt
# Define prediction function
def predict_fn(X):
    return model.predict_proba(X)
# Create ALE explainer
ale = ALE(predict_fn, feature_names=X.columns.tolist(), target_names=['Normal', 'Abnormal', 'Inconclusive'])
# Explain model's behavior on feature 0 (e.g., Gender)
ale_exp = ale.explain(X_test.values, features=[0])  # Index 0 is the first feature
# Check shape of ALE values
print("Feature bins:", ale_exp.feature_values[0])
print("ALE values shape:", np.array(ale_exp.ale_values).shape)
# Extract ALE values for class 1 (Abnormal, if index 1)
ale_class_1 = np.array(ale_exp.ale_values[0])[:, 1]
# Plot ALE for class 1
plt.plot(ale_exp.feature_values[0], ale_class_1, marker='o')
plt.xlabel('Gender')  # Ensure feature 0 is indeed "Gender"
plt.ylabel('ALE for class 1 (Abnormal)')
plt.title('ALE Plot for Gender Feature (Class: Abnormal)')
plt.grid(True)
plt.show()
```
## 🌟 Output:
![image](https://github.com/user-attachments/assets/565f69a2-6340-4dc0-9438-d9d3ab5b8d2a)

## 📌 Description
```
The goal of this project is to build a predictive model that can classify healthcare-related outcomes like 'Test Results' based on patient attributes such as age, gender, blood type, medical condition, and billing amount. The workflow includes loading data, cleaning, feature engineering, training models, evaluating performance, and applying explainable AI techniques.
```
## 🚀 Key Technologies
```
Python (Pandas, NumPy): Data manipulation and analysis
Scikit-learn: Machine learning modeling and evaluation
LIME / SHAP / ELI5 / Alibi: Model explanation and interpretability
Matplotlib: Visualization of model results and explanations
```
## 🚀 Further research

```
Explore Deep Learning Models: Implement deep learning architectures such as Convolutional Neural Networks (CNNs) or Recurrent Neural Networks (RNNs) for improved performance on complex healthcare data.
Integrate with EHR Systems: Connect the model with real-world Electronic Health Record (EHR) systems for real-time insights and deployment in clinical environments.
NLP on Clinical Notes: Use Natural Language Processing (NLP) techniques like BERT or BioBERT to extract information from unstructured medical text such as physician notes or discharge summaries.
Time-Series Analysis: Analyze patient vitals or medical history data over time using time-series modeling techniques like LSTMs or Prophet.
Dashboard for Stakeholders: Create a web-based interactive dashboard using tools like Streamlit, Gradio, or Dash for doctors and administrators to monitor and interpret predictions.
Fairness and Ethics: Conduct fairness audits and evaluate potential biases in predictions, especially across sensitive features like gender or ethnicity.
Explainability Benchmarks: Compare performance of different model interpretability techniques (e.g., LIME vs SHAP vs ALE) on clinical decision-making effectiveness.
```

## 🎬 IMDB DataSet

## project content

```
This project involves preprocessing and analyzing a movie reviews dataset (IMDb) to prepare it for sentiment classification using Natural Language Processing (NLP) techniques. The steps include text normalization, tokenization, stopword removal, contraction expansion, and TF-IDF feature extraction.
```
## Code

```python
pip install contractions
# Read from file (e.g., dataset_book.txt)
with open('IMDB Dataset.csv', 'r', encoding='utf-8') as file:
    text = file.read()
# Convert all text to lowercase
text_lower = text.lower()
# Print first 500 characters for preview
print(text_lower[:50001])
```

## 🌟 Output:
![image](https://github.com/user-attachments/assets/654855ae-086e-44a5-a9f8-0f8730181054)

```python
import re
# Read the file content
with open('IMDB Dataset.csv', 'r', encoding='utf-8') as file:
    text = file.read()
# Remove punctuation using regex
text_no_punct = re.sub(r'[^\w\s]', '', text)
# Print the first 500 characters to preview
print(text_no_punct[:50001])
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
# Load English stopwords
stop_words = set(stopwords.words('english'))
# Read content from the file
with open('IMDB Dataset.csv', 'r', encoding='utf-8') as file:
    text = file.read()
# Convert to lowercase and split into words
words = text.lower().split()
# Filter out stopwords
filtered = [word for word in words if word not in stop_words]
# Print the first 50 filtered words
print(filtered[:50001])
```
## 🌟 Output:
![image](https://github.com/user-attachments/assets/b116deb5-1d75-4053-9d57-693c1346b26e)
![image](https://github.com/user-attachments/assets/6a7670b5-992c-4470-8149-eaafdce07d6b)

```python

import re
# Read content from the file
with open('IMDB Dataset.csv', 'r', encoding='utf-8') as file:
    text = file.read()
# Remove numbers using regex
text_no_numbers = re.sub(r'\d+', '', text)
# Print the first 500 characters to preview
print(text_no_numbers[:50001])
import re
# Read content from the file
with open('IMDB Dataset.csv', 'r', encoding='utf-8') as file:
    text = file.read()
# Remove URLs using regex
text_no_url = re.sub(r"http\S+|www\S+|https\S+", "", text)
# Print the first 500 characters to preview
print(text_no_url[:50001])
```
## 🌟 Output:
![image](https://github.com/user-attachments/assets/a787ad75-bfe7-403a-b74d-2bb76beaeff7)

![image](https://github.com/user-attachments/assets/876b893e-84ea-447f-8a15-3c93f1bb13f9)

```python
# Read the content from the file
with open('IMDB Dataset.csv', 'r', encoding='utf-8') as file:
    text = file.read()
# Tokenize using split (basic method)
tokens = text.split()
# Print the first 50 tokens
print(tokens[:50001])
with open("IMDB Dataset.csv", "r", encoding="utf-8") as file:
    docs = [line.strip() for line in file if line.strip()]
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
# Load the IMDB dataset using pandas
df = pd.read_csv("IMDB Dataset.csv")
# Extract review texts and clean HTML tags
docs = [re.sub(r"<.*?>", "", review) for review in df["review"].dropna()]
# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=50001)
# Fit and transform the reviews
X = vectorizer.fit_transform(docs)
# Display vocabulary
print("Vocabulary:")
print(vectorizer.vocabulary_)
# Display TF-IDF matrix shape (avoid printing full matrix for large data)
print("\nTF-IDF Matrix shape:")
print(X.shape)
# Optionally, print TF-IDF vector for first review
print("\nTF-IDF Vector for first review:")
print(X[0])
```
## 🌟 Output:
![image](https://github.com/user-attachments/assets/15621fec-fcc8-4d38-8e36-ae9e5e4a2690)

```python
pip install contractions
import pandas as pd
import contractions
# Load the CSV
df = pd.read_csv("IMDB Dataset.csv")
# Expand contractions in the 'review' column
df["expanded_review"] = df["review"].apply(lambda x: contractions.fix(x) if pd.notnull(x) else "")
# Print a few examples
print(df[["review", "expanded_review"]].head())
```
## 🌟 Output:
![image](https://github.com/user-attachments/assets/ea4a55a2-e2d8-4fcd-8bf4-4bc3f0f29c56)

```python
import re
with open("IMDB Dataset.csv", "r", encoding="utf-8") as file:
    lines = file.readlines()
# Clean up spaces line by line
cleaned_lines = [re.sub(r'\s+', ' ', line).strip() for line in lines]
# Print cleaned lines
for line in cleaned_lines:
    print(line)
```
## 🌟 Output:
![image](https://github.com/user-attachments/assets/b574aead-8e90-4406-95df-8dec7c855777)


## 📌 Description
```
The goal of this project is to clean and transform raw IMDb movie review text data to prepare it for machine learning tasks like sentiment classification. We load the dataset, normalize the text, and convert it into TF-IDF vectors. This prepares the data for downstream tasks such as building classification models.
```

## 🚀 Key Technologies

```
Python (Pandas, re): Data manipulation and regular expressions
NLTK: Natural language processing and stopword removal
Scikit-learn (TfidfVectorizer): Feature extraction via TF-IDF
Contractions: Expanding English contractions for normalization
```
## 🚀 Further research

```
Sentiment Classification Models: Train and compare models like Logistic Regression, Naive Bayes, SVM, and deep learning (e.g., LSTM, CNN) on the TF-IDF features.
Transformer-Based Approaches: Fine-tune pre-trained transformer models such as BERT or RoBERTa for more accurate sentiment predictions.
Explainability: Use LIME or SHAP to interpret model predictions and understand which words contribute most to sentiment classification.
Data Augmentation: Apply techniques such as back-translation or synonym replacement to expand the training dataset.
Streaming Analysis: Analyze live movie reviews from platforms like Twitter or Reddit and classify them in real-time.
Deployment: Develop an interactive web interface using Streamlit, Gradio, or Flask to deploy the sentiment analysis tool for public use.
Multilingual Sentiment Analysis: Expand the dataset to include reviews in multiple languages and explore multilingual NLP models.
```

