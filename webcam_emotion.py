import os
import cv2
import time
import numpy as np
import tensorflow as tf

# =====================================================
# Configuration
# =====================================================

MODEL_PATH = "EmotionSavedModel"
SAVE_FOLDER = "captures"

os.makedirs(SAVE_FOLDER, exist_ok=True)

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

print("=" * 60)
print(" DeepFER - Real-Time Emotion Detection")
print("=" * 60)

# =====================================================
# Load SavedModel (TensorFlow 2.21 + Keras 3)
# =====================================================

print("Loading EmotionSavedModel...")

try:
    model = tf.keras.layers.TFSMLayer(
        MODEL_PATH,
        call_endpoint="serve"
    )

    print("Model loaded successfully!")

except Exception as e:
    print("\nError loading model")
    print(e)
    exit()

# =====================================================
# Load Face Detector
# =====================================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    print("Could not load Haar Cascade.")
    exit()

# =====================================================
# Image Preprocessing
# =====================================================

def preprocess_face(face):

    face = cv2.resize(face, (48, 48))

    face = face.astype(np.float32)

    face = face / 255.0

    face = np.expand_dims(face, axis=-1)

    face = np.expand_dims(face, axis=0)

    return face


# =====================================================
# Emotion Prediction
# =====================================================

def predict_emotion(face):

    processed = preprocess_face(face)

    prediction = model(processed)

    # TFSMLayer may return dict
    if isinstance(prediction, dict):
        prediction = list(prediction.values())[0]

    prediction = prediction.numpy()[0]

    emotion_index = np.argmax(prediction)

    emotion = emotion_labels[emotion_index]

    confidence = float(np.max(prediction) * 100)

    return emotion, confidence


print("Initialization Complete.")
# =====================================================
# Webcam Initialization
# =====================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Cannot access webcam.")
    exit()

# Webcam resolution
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\nControls")
print("-------------------------------------")
print("SPACE : Capture & Predict Emotion")
print("R     : Resume Live Camera")
print("Q     : Quit")
print("-------------------------------------")


# =====================================================
# Variables
# =====================================================

freeze = False

result_frame = None

captured_frame = None

prediction_done = False

capture_count = 1

last_capture_time = 0

capture_delay = 1.5

fps = 0

previous_time = time.time()

smoothed_confidence = 0

alpha = 0.20
# =====================================================
# Detect Largest Face
# =====================================================

def get_largest_face(gray):

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80,80)
    )

    if len(faces) == 0:
        return None

    largest = max(
        faces,
        key=lambda rect: rect[2] * rect[3]
    )

    return largest
# =====================================================
# FPS Calculation
# =====================================================

def calculate_fps():

    global previous_time

    current_time = time.time()

    fps = 1 / (current_time - previous_time)

    previous_time = current_time

    return int(fps)
# =====================================================
# Main Loop
# =====================================================

while True:

    # Freeze Mode
    if freeze:

        cv2.imshow(
            "DeepFER Emotion Detection",
            result_frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):

            freeze = False
            prediction_done = False

        elif key == ord("q"):
            break

        continue

    ret, frame = camera.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    display_frame = frame.copy()

    fps = calculate_fps()

    cv2.putText(
        display_frame,
        f"FPS : {fps}",
        (15,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,255),
        2
    )
        # =====================================================
    # Detect Face
    # =====================================================

    face = get_largest_face(gray)

    if face is not None:

        x, y, w, h = face

        # Add margin around face
        margin = int(0.20 * max(w, h))

        x1 = max(0, x - margin)
        y1 = max(0, y - margin)

        x2 = min(gray.shape[1], x + w + margin)
        y2 = min(gray.shape[0], y + h + margin)

        face_crop = gray[y1:y2, x1:x2]

        # Draw Face Rectangle
        cv2.rectangle(
            display_frame,
            (x1, y1),
            (x2, y2),
            (0,255,0),
            2
        )

    else:

        face_crop = None

        cv2.putText(
            display_frame,
            "No Face Detected",
            (20,70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,255),
            2
        )

    # =====================================================
    # Show Live Camera
    # =====================================================

    cv2.imshow(
        "DeepFER Emotion Detection",
        display_frame
    )

    key = cv2.waitKey(1) & 0xFF

    # =====================================================
    # Quit
    # =====================================================

    if key == ord("q"):
        break

    # =====================================================
    # Capture (SPACE)
    # =====================================================

    if key == 32:

        current_time = time.time()

        if current_time - last_capture_time < capture_delay:
            continue

        last_capture_time = current_time

        if face_crop is None:

            print("No face detected.")
            continue

        print("\nAnalyzing Emotion...")

        emotion, confidence = predict_emotion(face_crop)

        # Smooth confidence (optional)
        if smoothed_confidence == 0:
            smoothed_confidence = confidence
        else:
            smoothed_confidence = (
                alpha * confidence
                + (1-alpha) * smoothed_confidence
            )

        confidence = smoothed_confidence

        # Freeze current frame
        result_frame = display_frame.copy()

        # Draw prediction
        cv2.rectangle(
            result_frame,
            (x1, y1),
            (x2, y2),
            (0,255,0),
            3
        )

        cv2.putText(
            result_frame,
            f"{emotion} ({confidence:.2f}%)",
            (x1, y1-20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,0),
            2
        )

        # Timestamp
        timestamp = time.strftime("%d-%m-%Y %H:%M:%S")

        cv2.putText(
            result_frame,
            timestamp,
            (15, result_frame.shape[0]-20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )

        filename = os.path.join(
            SAVE_FOLDER,
            f"capture_{capture_count}.jpg"
        )

        cv2.imwrite(
            filename,
            result_frame
        )

        print(f"Saved: {filename}")

        capture_count += 1

        freeze = True
        # =====================================================
# Cleanup
# =====================================================

camera.release()

cv2.destroyAllWindows()

print("\n========================================")
print(" DeepFER Closed Successfully")
print("========================================")
print(f"Images Saved : {capture_count-1}")
print(f"Folder       : {os.path.abspath(SAVE_FOLDER)}")
print("========================================")