# Interactive Handwritten Digit Recognizer using CNN

This is a desktop-based machine learning application built with Python Tkinter.

A desktop application for recognizing handwritten digits using a Convolutional Neural Network (CNN) trained on the MNIST dataset. The project includes a Tkinter-based graphical user interface where users can draw a digit, upload a single-digit image, or upload an image containing multiple digits.

The app preprocesses input images using OpenCV, predicts the digit or number, shows the confidence scores, and stores prediction history and user feedback for later review.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Training the Model](#training-the-model)
- [How Prediction Works](#how-prediction-works)
- [Output Files and Data Storage](#output-files-and-data-storage)
- [Tips for Better Accuracy](#tips-for-better-accuracy)
- [Known Limitations](#known-limitations)
- [Author](#author)
- [License](#license)

---

## Project Overview

This project is a complete handwritten digit recognition system built with Python. It combines:

- A CNN model trained on MNIST data
- OpenCV-based image preprocessing
- A Tkinter desktop GUI
- Prediction history logging
- Feedback capture for incorrect predictions

It is suitable for learning, demonstration, and experimentation with deep learning and computer vision.

---

## Features

### 1. Draw a Digit on Canvas
- Users can draw a handwritten digit directly inside the app.
- The drawing is converted into an image and passed through the recognition pipeline.

### 2. Upload a Single Digit Image
- Users can upload a clear image containing one handwritten digit.
- The app preprocesses the image and predicts the digit.

### 3. Upload a Multi-Digit Image
- Users can upload an image containing multiple separated digits.
- The app detects individual digit regions, recognizes each one, and combines them into a final number.

### 4. Preview Original and Processed Images
- The app displays:
  - the original input image
  - the processed version used by the model

### 5. Confidence Scores and Prediction Details
- After prediction, the app shows:
  - predicted class
  - confidence percentage
  - probability values for all digits from 0 to 9

### 6. Save Prediction History
- Every prediction is logged in a CSV file for reference.

### 7. Feedback System
- If the prediction is wrong, users can enter the correct value and save it.
- Feedback is stored for future improvement and analysis.

---

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Pillow
- Tkinter
- Pandas

---

## Project Structure

```text
Handwritten_Digit_Recognizer/
│
├── app.py                  # Main Tkinter GUI application
├── predict_digit.py        # Script for predicting a single image from file
├── train_model.py          # Script to train the CNN model
├── requirements.txt        # Python dependencies
├── model/
│   └── digit_model.h5      # Trained CNN model
├── feedback/
│   └── feedback.csv        # User feedback records
├── history/
│   └── prediction_history.csv
├── test_images/            # Sample images for testing
├── screenshots/            # Screenshots of the app
└── README.md               # Project documentation
```

---

## Installation

### 1. Clone the Project

```bash
git clone <repository-url>
cd Handwritten_Digit_Recognizer
```

### 2. Create a Virtual Environment

On Windows:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Ensure the Model File Exists

The app expects the trained model at:

```text
model/digit_model.h5
```

If the file is missing, train the model first using the training script.

---

## How to Run

### Run the Desktop Application

```bash
python app.py
```

This will open the Tkinter GUI where you can:
- draw a digit
- upload a single-digit image
- upload a multi-digit image
- view predictions and confidence
- save feedback

### Run Prediction on a Single Image File

```bash
python predict_digit.py
```

This script uses a sample image from the test_images folder and prints the predicted digit and confidence.

---

## Training the Model

To train the CNN model from scratch, run:

```bash
python train_model.py
```

### What the training script does
- Downloads the MNIST dataset
- Reshapes the images into a format suitable for CNN input
- Normalizes the pixel values
- Builds a convolutional neural network
- Trains the model for several epochs
- Saves the trained model as:

```text
model/digit_model.h5
```

> Training may take some time depending on your hardware and environment.

---

## How Prediction Works

The recognition pipeline works in the following steps:

1. Input image is loaded.
2. The image is converted to grayscale.
3. The image is inverted and thresholded to improve digit visibility.
4. The digit region is cropped and resized.
5. The image is centered on a 28x28 canvas.
6. The image is normalized and passed to the CNN model.
7. The model outputs probabilities for digits 0 through 9.
8. The highest-probability digit is selected as the prediction.

For multi-digit recognition:
- the image is processed to find separate connected components
- each digit region is isolated and recognized individually
- the results are concatenated into one number

---

## Output Files and Data Storage

### Prediction History
The app stores prediction records in:

```text
history/prediction_history.csv
```

Each row includes:
- timestamp
- input type
- prediction value
- confidence

### Feedback Data
Incorrect predictions can be saved in:

```text
feedback/feedback.csv
```

Each feedback entry contains:
- timestamp
- input type
- predicted value
- actual value
- confidence
- image path

### Test Images
Sample input images are stored in:

```text
test_images/
```

---

## Tips for Better Accuracy

For best results:

- Use clear, centered handwriting.
- Make the digit fill most of the image area.
- Avoid heavy background noise.
- For multi-digit images, ensure digits are separated clearly.
- Use a high-contrast image with a light background and dark digit.

---

## Known Limitations

- The app performs best with clean and simple handwritten digits.
- Very noisy images may reduce accuracy.
- Multi-digit recognition requires the digits to be visibly separated.
- The model is trained on the MNIST dataset, so it is optimized for digit-like handwriting rather than complex handwriting styles.

---

## Author

Created by Rittik Basak.

---

## License

This project is intended for educational and personal use. If you plan to reuse or distribute it, add an appropriate license file based on your requirements.
