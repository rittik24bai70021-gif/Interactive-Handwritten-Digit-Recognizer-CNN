
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk
import tensorflow as tf
import numpy as np
import cv2
import os
import csv
from datetime import datetime


# ==============================
# Load Model
# ==============================

MODEL_PATH = "model/digit_model.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found. Please run train_model.py first.")

model = tf.keras.models.load_model(MODEL_PATH)


# ==============================
# Create Required Folders
# ==============================

os.makedirs("feedback", exist_ok=True)
os.makedirs("history", exist_ok=True)
os.makedirs("test_images", exist_ok=True)


# ==============================
# Global Variables
# ==============================

last_prediction = None
last_confidence = None
last_input_type = None
last_image_path = None

canvas_image = None
draw = None


# ==============================
# Colors and Styling
# ==============================

BG_MAIN = "#eef3f9"
CARD_BG = "#ffffff"
CARD_BORDER = "#cbd5e1"

TEXT_DARK = "#0f172a"
TEXT_MID = "#475569"
TEXT_LIGHT = "#64748b"

BLUE = "#2563eb"
BLUE_DARK = "#1d4ed8"
RED = "#ef4444"
RED_DARK = "#dc2626"
GREEN = "#16a34a"
GREEN_DARK = "#15803d"
PURPLE = "#7c3aed"
PURPLE_DARK = "#6d28d9"
ORANGE = "#f97316"
ORANGE_DARK = "#ea580c"

TITLE_BLUE = "#2563eb"
TITLE_PINK = "#ec4899"

CARD_WIDTH = 440
CARD_HEIGHT = 665
CARD_GAP = 14

CANVAS_SIZE = 350
PREVIEW_WIDTH = 260
PREVIEW_HEIGHT = 210


# ==============================
# Image Helper Functions
# ==============================

def create_placeholder(width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT):
    image = Image.new("RGB", (width, height), "#e5e7eb")
    return ImageTk.PhotoImage(image)


def show_image_on_label(label, image_array, size=(PREVIEW_WIDTH, PREVIEW_HEIGHT)):
    if image_array is None:
        return

    if len(image_array.shape) == 2:
        image = Image.fromarray(image_array)
    else:
        image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))

    image = image.resize(size, Image.Resampling.NEAREST)
    image_tk = ImageTk.PhotoImage(image)

    label.config(image=image_tk)
    label.image = image_tk


def preprocess_single_digit_from_array(img):
    """
    Converts a handwritten digit image into MNIST-style 28x28 format.
    Output: white digit on black background.
    """

    if img is None:
        return None

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert black digit on white background to white digit on black background
    img = 255 - img

    # Threshold to remove background noise
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(img)

    if coords is None:
        return None

    # Crop digit area
    x, y, w, h = cv2.boundingRect(coords)
    digit = img[y:y + h, x:x + w]

    h, w = digit.shape

    if h == 0 or w == 0:
        return None

    # Resize while keeping aspect ratio
    if h > w:
        new_h = 20
        new_w = int(w * (20 / h))
    else:
        new_w = 20
        new_h = int(h * (20 / w))

    new_w = max(new_w, 1)
    new_h = max(new_h, 1)

    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Place digit on 28x28 black canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = digit

    # Center using image moments
    moments = cv2.moments(canvas)

    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        shift_x = 14 - cx
        shift_y = 14 - cy

        matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        canvas = cv2.warpAffine(canvas, matrix, (28, 28))

    return canvas


def predict_processed_digit(processed_img):
    input_img = processed_img / 255.0
    input_img = input_img.reshape(1, 28, 28, 1)

    prediction = model.predict(input_img, verbose=0)[0]

    predicted_digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)

    return predicted_digit, confidence, prediction


def update_probability_text_single(prediction_scores):
    probability_text.config(state="normal")
    probability_text.delete("1.0", tk.END)

    probability_text.insert(tk.END, "Single Digit Confidence\n")
    probability_text.insert(tk.END, "-----------------------\n")

    for i, score in enumerate(prediction_scores):
        probability_text.insert(tk.END, f"{i}: {score * 100:.2f}%\n")

    probability_text.config(state="disabled")


def update_probability_text_multi(details):
    probability_text.config(state="normal")
    probability_text.delete("1.0", tk.END)

    probability_text.insert(tk.END, "Multi-Digit Details\n")
    probability_text.insert(tk.END, "--------------------\n")

    for index, digit, confidence in details:
        probability_text.insert(
            tk.END,
            f"Digit {index}: {digit}   ({confidence:.2f}%)\n"
        )

    probability_text.config(state="disabled")


def save_prediction_history(predicted_value, confidence, input_type):
    history_path = "history/prediction_history.csv"
    file_exists = os.path.isfile(history_path)

    with open(history_path, mode="a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["Time", "Input Type", "Prediction", "Confidence"])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            input_type,
            predicted_value,
            f"{confidence:.2f}%"
        ])


def update_result_color(confidence):
    if confidence >= 80:
        result_label.config(fg=GREEN)
    elif confidence >= 50:
        result_label.config(fg=ORANGE)
    else:
        result_label.config(fg=RED)


def update_single_prediction_ui(original_img, processed_img, predicted_digit, confidence, prediction_scores):
    result_label.config(
        text=f"Predicted Digit: {predicted_digit}\nConfidence: {confidence:.2f}%"
    )

    update_result_color(confidence)

    multi_result_label.config(text="Multi-Digit Result:")

    show_image_on_label(original_label, original_img)
    show_image_on_label(processed_label, processed_img)

    update_probability_text_single(prediction_scores)


# ==============================
# Single Digit Prediction
# ==============================

def predict_from_canvas():
    global last_prediction, last_confidence, last_input_type, last_image_path

    canvas_image_path = "test_images/canvas_digit.png"
    canvas_image.save(canvas_image_path)

    original_img = cv2.imread(canvas_image_path, cv2.IMREAD_GRAYSCALE)

    processed_img = preprocess_single_digit_from_array(original_img)

    if processed_img is None:
        messagebox.showerror("Error", "No digit found. Please draw a clear digit.")
        return

    predicted_digit, confidence, prediction_scores = predict_processed_digit(processed_img)

    update_single_prediction_ui(
        original_img,
        processed_img,
        predicted_digit,
        confidence,
        prediction_scores
    )

    last_prediction = str(predicted_digit)
    last_confidence = confidence
    last_input_type = "Canvas"
    last_image_path = canvas_image_path

    save_prediction_history(predicted_digit, confidence, "Canvas")


def predict_from_uploaded_image():
    global last_prediction, last_confidence, last_input_type, last_image_path

    file_path = filedialog.askopenfilename(
        title="Select Single Digit Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )

    if not file_path:
        return

    original_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    processed_img = preprocess_single_digit_from_array(original_img)

    if processed_img is None:
        messagebox.showerror("Error", "No digit found. Please upload a clear image.")
        return

    predicted_digit, confidence, prediction_scores = predict_processed_digit(processed_img)

    update_single_prediction_ui(
        original_img,
        processed_img,
        predicted_digit,
        confidence,
        prediction_scores
    )

    last_prediction = str(predicted_digit)
    last_confidence = confidence
    last_input_type = "Uploaded Single Digit"
    last_image_path = file_path

    save_prediction_history(predicted_digit, confidence, "Uploaded Single Digit")


# ==============================
# Multi-Digit Recognition
# ==============================

def recognize_multi_digit_image():
    global last_prediction, last_confidence, last_input_type, last_image_path

    file_path = filedialog.askopenfilename(
        title="Select Multi-Digit Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )

    if not file_path:
        return

    original_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    if original_img is None:
        messagebox.showerror("Error", "Image could not be loaded.")
        return

    inverted = 255 - original_img
    _, thresh = cv2.threshold(inverted, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    digit_boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Ignore tiny noise
        if w > 8 and h > 15:
            digit_boxes.append((x, y, w, h))

    if len(digit_boxes) == 0:
        messagebox.showerror("Error", "No digits found. Use clear separated digits.")
        return

    # Sort digits from left to right
    digit_boxes = sorted(digit_boxes, key=lambda box: box[0])

    final_number = ""
    confidence_values = []
    details = []

    preview_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

    for index, (x, y, w, h) in enumerate(digit_boxes, start=1):
        padding = 8

        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, original_img.shape[1])
        y2 = min(y + h + padding, original_img.shape[0])

        digit_crop = original_img[y1:y2, x1:x2]

        processed_img = preprocess_single_digit_from_array(digit_crop)

        if processed_img is None:
            continue

        predicted_digit, confidence, _ = predict_processed_digit(processed_img)

        final_number += str(predicted_digit)
        confidence_values.append(confidence)
        details.append((index, predicted_digit, confidence))

        cv2.rectangle(preview_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            preview_img,
            str(predicted_digit),
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

    if final_number == "":
        messagebox.showerror("Error", "Could not recognize the digits properly.")
        return

    average_confidence = sum(confidence_values) / len(confidence_values)

    result_label.config(
        text=f"Predicted Number: {final_number}\nAvg Confidence: {average_confidence:.2f}%"
    )

    update_result_color(average_confidence)

    multi_result_label.config(
        text=f"Multi-Digit Result: {final_number}\nAvg Confidence: {average_confidence:.2f}%"
    )

    show_image_on_label(original_label, preview_img)
    show_image_on_label(processed_label, thresh)

    update_probability_text_multi(details)

    last_prediction = final_number
    last_confidence = average_confidence
    last_input_type = "Multi-Digit Image"
    last_image_path = file_path

    save_prediction_history(final_number, average_confidence, "Multi-Digit Image")


# ==============================
# Feedback System
# ==============================

def save_feedback():
    if last_prediction is None:
        messagebox.showwarning("Warning", "Please make a prediction first.")
        return

    actual_value = actual_digit_entry.get().strip()

    if actual_value == "":
        messagebox.showwarning("Warning", "Please enter the correct digit or number.")
        return

    if not actual_value.isdigit():
        messagebox.showwarning("Warning", "Only numbers are allowed.")
        return

    feedback_path = "feedback/feedback.csv"
    file_exists = os.path.isfile(feedback_path)

    with open(feedback_path, mode="a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Time",
                "Input Type",
                "Predicted Value",
                "Actual Value",
                "Confidence",
                "Image Path"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            last_input_type,
            last_prediction,
            actual_value,
            f"{last_confidence:.2f}%",
            last_image_path
        ])

    messagebox.showinfo("Saved", "Feedback saved successfully.")
    actual_digit_entry.delete(0, tk.END)


# ==============================
# Canvas Drawing
# ==============================

def paint(event):
    x1, y1 = event.x - 10, event.y - 10
    x2, y2 = event.x + 10, event.y + 10

    drawing_canvas.create_oval(
        x1,
        y1,
        x2,
        y2,
        fill="black",
        outline="black"
    )

    draw.ellipse(
        (x1, y1, x2, y2),
        fill="black"
    )


def clear_canvas():
    global canvas_image, draw, last_prediction, last_confidence, last_input_type, last_image_path

    drawing_canvas.delete("all")

    canvas_image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
    draw = ImageDraw.Draw(canvas_image)

    result_label.config(
        text="Prediction will appear here",
        fg=TEXT_DARK
    )

    multi_result_label.config(text="Multi-Digit Result:")

    probability_text.config(state="normal")
    probability_text.delete("1.0", tk.END)
    probability_text.config(state="disabled")

    original_label.config(image=placeholder_img)
    original_label.image = placeholder_img

    processed_label.config(image=placeholder_img)
    processed_label.image = placeholder_img

    actual_digit_entry.delete(0, tk.END)

    last_prediction = None
    last_confidence = None
    last_input_type = None
    last_image_path = None


# ==============================
# Hover Effects
# ==============================

def add_hover_effect(button, normal_color, hover_color):
    button.bind("<Enter>", lambda event: button.config(bg=hover_color))
    button.bind("<Leave>", lambda event: button.config(bg=normal_color))


# ==============================
# GUI Design
# ==============================

root = tk.Tk()
root.title("Interactive Handwritten Digit Recognizer")
root.geometry("1450x850")
root.minsize(1250, 780)
root.configure(bg=BG_MAIN)
root.resizable(True, True)

try:
    root.state("zoomed")
except Exception:
    pass

placeholder_img = create_placeholder()


# ==============================
# Root Grid Center Setup
# ==============================

root.grid_rowconfigure(0, weight=0)
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(2, weight=0)
root.grid_columnconfigure(0, weight=1)


# ==============================
# Title Section
# ==============================

title_frame = tk.Frame(root, bg=BG_MAIN)
title_frame.grid(row=0, column=0, pady=(22, 5), sticky="n")

title_label = tk.Label(
    title_frame,
    text="Interactive Handwritten Digit Recognizer",
    font=("Georgia", 32, "bold"),
    fg=TITLE_BLUE,
    bg=BG_MAIN
)
title_label.pack()

subtitle_label = tk.Label(
    title_frame,
    text="using CNN",
    font=("Georgia", 19, "bold italic"),
    fg=TITLE_PINK,
    bg=BG_MAIN
)
subtitle_label.pack(pady=(0, 6))


# ==============================
# Center Wrapper
# ==============================

center_wrapper = tk.Frame(root, bg=BG_MAIN)
center_wrapper.grid(row=1, column=0, sticky="nsew")

center_wrapper.grid_rowconfigure(0, weight=1)
center_wrapper.grid_columnconfigure(0, weight=1)

main_frame = tk.Frame(center_wrapper, bg=BG_MAIN)
main_frame.grid(row=0, column=0)

main_frame.grid_columnconfigure(0, weight=1)
main_frame.grid_columnconfigure(1, weight=1)
main_frame.grid_columnconfigure(2, weight=1)


# ==============================
# Left Card
# ==============================

left_frame = tk.Frame(
    main_frame,
    bg=CARD_BG,
    width=CARD_WIDTH,
    height=CARD_HEIGHT,
    relief="solid",
    bd=1,
    highlightbackground=CARD_BORDER,
    highlightthickness=1
)
left_frame.grid(row=0, column=0, padx=CARD_GAP, pady=10, sticky="n")
left_frame.grid_propagate(False)

left_title = tk.Label(
    left_frame,
    text="Draw a Single Digit",
    font=("Segoe UI", 18, "bold"),
    bg=CARD_BG,
    fg=TEXT_DARK
)
left_title.pack(pady=(18, 12))

drawing_canvas = tk.Canvas(
    left_frame,
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    bg="white",
    cursor="cross",
    highlightthickness=2,
    highlightbackground="#1f2937"
)
drawing_canvas.pack(pady=6)

canvas_image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
draw = ImageDraw.Draw(canvas_image)

drawing_canvas.bind("<B1-Motion>", paint)

button_frame = tk.Frame(left_frame, bg=CARD_BG)
button_frame.pack(pady=16)

predict_canvas_button = tk.Button(
    button_frame,
    text="Predict Canvas",
    command=predict_from_canvas,
    font=("Segoe UI", 11, "bold"),
    bg=BLUE,
    fg="white",
    width=16,
    height=2,
    relief="flat",
    cursor="hand2"
)
predict_canvas_button.grid(row=0, column=0, padx=8)

clear_button = tk.Button(
    button_frame,
    text="Clear",
    command=clear_canvas,
    font=("Segoe UI", 11, "bold"),
    bg=RED,
    fg="white",
    width=11,
    height=2,
    relief="flat",
    cursor="hand2"
)
clear_button.grid(row=0, column=1, padx=8)

upload_single_button = tk.Button(
    left_frame,
    text="Upload Single Digit Image",
    command=predict_from_uploaded_image,
    font=("Segoe UI", 11, "bold"),
    bg=GREEN,
    fg="white",
    width=34,
    height=2,
    relief="flat",
    cursor="hand2"
)
upload_single_button.pack(pady=8)

upload_multi_button = tk.Button(
    left_frame,
    text="Upload Multi-Digit Image",
    command=recognize_multi_digit_image,
    font=("Segoe UI", 11, "bold"),
    bg=PURPLE,
    fg="white",
    width=34,
    height=2,
    relief="flat",
    cursor="hand2"
)
upload_multi_button.pack(pady=8)

multi_result_label = tk.Label(
    left_frame,
    text="Multi-Digit Result:",
    font=("Segoe UI", 13, "bold"),
    bg=CARD_BG,
    fg=TEXT_DARK,
    justify="center"
)
multi_result_label.pack(pady=14)


# ==============================
# Middle Card
# ==============================

middle_frame = tk.Frame(
    main_frame,
    bg=CARD_BG,
    width=CARD_WIDTH,
    height=CARD_HEIGHT,
    relief="solid",
    bd=1,
    highlightbackground=CARD_BORDER,
    highlightthickness=1
)
middle_frame.grid(row=0, column=1, padx=CARD_GAP, pady=10, sticky="n")
middle_frame.grid_propagate(False)

preview_title = tk.Label(
    middle_frame,
    text="Image Preview",
    font=("Segoe UI", 18, "bold"),
    bg=CARD_BG,
    fg=TEXT_DARK
)
preview_title.pack(pady=(18, 12))

original_title = tk.Label(
    middle_frame,
    text="Original Image",
    font=("Segoe UI", 12, "bold"),
    bg=CARD_BG,
    fg=TEXT_MID
)
original_title.pack(pady=4)

original_label = tk.Label(
    middle_frame,
    image=placeholder_img,
    bg="#e5e7eb",
    relief="solid",
    bd=1
)
original_label.image = placeholder_img
original_label.pack(pady=8)

processed_title = tk.Label(
    middle_frame,
    text="Processed Preview",
    font=("Segoe UI", 12, "bold"),
    bg=CARD_BG,
    fg=TEXT_MID
)
processed_title.pack(pady=(16, 4))

processed_label = tk.Label(
    middle_frame,
    image=placeholder_img,
    bg="#e5e7eb",
    relief="solid",
    bd=1
)
processed_label.image = placeholder_img
processed_label.pack(pady=8)


# ==============================
# Right Card
# ==============================

right_frame = tk.Frame(
    main_frame,
    bg=CARD_BG,
    width=CARD_WIDTH,
    height=CARD_HEIGHT,
    relief="solid",
    bd=1,
    highlightbackground=CARD_BORDER,
    highlightthickness=1
)
right_frame.grid(row=0, column=2, padx=CARD_GAP, pady=10, sticky="n")
right_frame.grid_propagate(False)

result_title = tk.Label(
    right_frame,
    text="Prediction Result",
    font=("Segoe UI", 18, "bold"),
    bg=CARD_BG,
    fg=TEXT_DARK
)
result_title.pack(pady=(18, 12))

result_label = tk.Label(
    right_frame,
    text="Prediction will appear here",
    font=("Segoe UI", 16, "bold"),
    bg=CARD_BG,
    fg=TEXT_DARK,
    justify="center"
)
result_label.pack(pady=12)

prob_title = tk.Label(
    right_frame,
    text="Confidence / Prediction Details",
    font=("Segoe UI", 12, "bold"),
    bg=CARD_BG,
    fg=TEXT_MID
)
prob_title.pack(pady=(8, 5))

probability_text = tk.Text(
    right_frame,
    height=12,
    width=38,
    font=("Consolas", 11),
    bg="#f8fafc",
    fg=TEXT_DARK,
    relief="solid",
    bd=1
)
probability_text.pack(pady=8)
probability_text.config(state="disabled")

feedback_title = tk.Label(
    right_frame,
    text="Incorrect Prediction Feedback",
    font=("Segoe UI", 12, "bold"),
    bg=CARD_BG,
    fg=TEXT_MID
)
feedback_title.pack(pady=(14, 5))

feedback_note = tk.Label(
    right_frame,
    text="If prediction is wrong, enter the correct digit/number:",
    font=("Segoe UI", 10),
    bg=CARD_BG,
    fg=TEXT_LIGHT,
    wraplength=350,
    justify="center"
)
feedback_note.pack()

actual_digit_entry = tk.Entry(
    right_frame,
    font=("Segoe UI", 13),
    width=26,
    justify="center",
    relief="solid",
    bd=1
)
actual_digit_entry.pack(pady=10)

feedback_button = tk.Button(
    right_frame,
    text="Save Correct Value",
    command=save_feedback,
    font=("Segoe UI", 11, "bold"),
    bg=ORANGE,
    fg="white",
    width=28,
    height=2,
    relief="flat",
    cursor="hand2"
)
feedback_button.pack(pady=8)


# ==============================
# Footer
# ==============================

footer_label = tk.Label(
    root,
    text="Created by Rittik Basak  |  CNN + MNIST + OpenCV + Tkinter",
    font=("Segoe UI", 10, "bold"),
    bg=BG_MAIN,
    fg=TEXT_MID
)
footer_label.grid(row=2, column=0, pady=(3, 12), sticky="s")


# ==============================
# Add Hover Effects
# ==============================

add_hover_effect(predict_canvas_button, BLUE, BLUE_DARK)
add_hover_effect(clear_button, RED, RED_DARK)
add_hover_effect(upload_single_button, GREEN, GREEN_DARK)
add_hover_effect(upload_multi_button, PURPLE, PURPLE_DARK)
add_hover_effect(feedback_button, ORANGE, ORANGE_DARK)


# ==============================
# Run App
# ==============================

root.mainloop()