import cv2
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model(tf.zeros([1, 224, 224, 3]))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.load_weights("drowsiness_weights.weights.h5")
print("✅ Model loaded successfully")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

pred_buffer = deque(maxlen=10)
cap = cv2.VideoCapture(0)

print("Camera started — press ESC to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Could not read from camera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        img = cv2.resize(face, (224, 224))
        img = img.astype("float32")
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img, verbose=0)[0][0]
        pred_buffer.append(pred)
        avg_pred = np.mean(pred_buffer)

        label = "Drowsy" if avg_pred > 0.5 else "Alert"
        color = (0, 0, 255) if avg_pred > 0.5 else (0, 255, 0)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{label} ({avg_pred:.2f})",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

    cv2.imshow("Driver Drowsiness Monitor", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
