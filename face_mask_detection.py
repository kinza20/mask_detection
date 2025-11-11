import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# =========================
# Step 1: Load Dataset
# =========================
data = []
labels = []

with_mask_dir = "dataset/with_mask"
without_mask_dir = "dataset/without_mask"

# Load images with mask
for file in os.listdir(with_mask_dir):
    img = cv2.imread(os.path.join(with_mask_dir, file))
    if img is not None:
        img = cv2.resize(img, (64, 64))
        data.append(img)
        labels.append(1)

# Load images without mask
for file in os.listdir(without_mask_dir):
    img = cv2.imread(os.path.join(without_mask_dir, file))
    if img is not None:
        img = cv2.resize(img, (64, 64))
        data.append(img)
        labels.append(0)

data = np.array(data, dtype="float32") / 255.0  # Normalize
labels = np.array(labels)
labels = to_categorical(labels, num_classes=2)  # One-hot encoding

print("Dataset shape:", data.shape)
print("Labels distribution:", np.sum(labels, axis=0))

# =========================
# Step 2: Train/Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# =========================
# Step 3: Build CNN Model
# =========================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# =========================
# Step 4: Train Model
# =========================
print("Training model...")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)

# =========================
# Step 5: Webcam Detection
# =========================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Cannot open webcam")
    exit()

print("Press 'q' to quit")
while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (64,64))
        face_input = np.expand_dims(face_resized, axis=0)  # shape (1,64,64,3)
        pred = model.predict(face_input, verbose=0)
        label = "Mask" if np.argmax(pred) == 1 else "No Mask"
        color = (0,255,0) if np.argmax(pred) == 1 else (0,0,255)
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Mask Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
