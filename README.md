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
This script is designed to mount Google Drive in a Google Colab environment and then load and display images from a specific folder in your Drive.
Drive Mounting:
drive.mount('/content/drive') connects your Google Drive to the Colab workspace, allowing access to files stored there.
Folder Path:
The variable folder_path points to the directory containing dog images (/content/drive/MyDrive/dogs).
Image Listing:
It scans the folder for image files with extensions .jpg, .jpeg, and .png.
Image Loading & Display:
The script loads up to the first 50 images, resizing each to 200x200 pixels using Keras’ load_img function. Each image is then displayed one by one using Matplotlib, with the filename shown as the title.

## 🛠 Code
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
code for cat:
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

## 🚀 Key Technologies

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

---

## 🌟 Output:
![image](https://github.com/user-attachments/assets/19436fa3-47f9-4cae-896e-3f83761b28f5)


---

## 🚀 Further research
To get started with MeetMate, ensure you have Python and Google Chrome installed. The system uses a combination of browser automation and speech recognition to process meetings.

---

## 💠 Installation
### Requirements
- Python 3.8+
- Chrome browser
- pip dependencies: `pyaudio`, `whisper`, `gradio`, `selenium`, `openai`, `webrtcvad`

### Steps
```bash
# Clone the repository
git clone https://github.com/your-username/meetmate.git

# Install dependencies
cd meetmate
pip install -r requirements.txt

