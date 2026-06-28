import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model("model/digit_model.h5")


def preprocess_image(image_path):
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError("Image not found. Check the image path.")

    # Invert image: black digit on white background becomes white digit on black background
    img = 255 - img

    # Threshold
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    # Find digit pixels
    coords = cv2.findNonZero(img)

    if coords is None:
        raise ValueError("No digit found in the image.")

    # Crop digit
    x, y, w, h = cv2.boundingRect(coords)
    digit = img[y:y+h, x:x+w]

    # Resize while keeping aspect ratio
    h, w = digit.shape

    if h > w:
        new_h = 20
        new_w = int(w * (20 / h))
    else:
        new_w = 20
        new_h = int(h * (20 / w))

    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Put digit in 28x28 black canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = digit

    # Center the digit using center of mass
    moments = cv2.moments(canvas)

    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        shift_x = 14 - cx
        shift_y = 14 - cy

        matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        canvas = cv2.warpAffine(canvas, matrix, (28, 28))

    # Normalize
    canvas = canvas / 255.0

    # Reshape for CNN
    canvas = canvas.reshape(1, 28, 28, 1)

    return canvas


def predict_digit(image_path):
    processed_img = preprocess_image(image_path)

    prediction = model.predict(processed_img)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    print(f"Predicted Digit: {predicted_digit}")
    print(f"Confidence: {confidence:.2f}%")

    # Show probabilities for all digits
    print("\nPrediction scores:")
    for i, score in enumerate(prediction[0]):
        print(f"{i}: {score * 100:.2f}%")

    # Show processed image
    plt.imshow(processed_img.reshape(28, 28), cmap="gray")
    plt.title(f"Predicted Digit: {predicted_digit}")
    plt.axis("off")
    plt.show()


predict_digit("test_images/digit.png")