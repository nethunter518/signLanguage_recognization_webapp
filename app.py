from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import mediapipe as mp
from gtts import gTTS
from googletrans import Translator
import os
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time
from threading import Lock
import logging

app = Flask(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    static_image_mode=False
)

# Initialize Gesture Recognizer
from gesture import GestureRecognizer
gesture_recognizer = GestureRecognizer()

# Initialize Translator
translator = Translator()

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
TRAINING_DATA_DIR = 'training_data'
MODEL_DIR = 'model'
OUTPUT_DIR = 'output'

os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global variables with thread safety
detection_active = False
current_gesture = "Unknown Gesture"
last_gesture = ""
current_confidence = 0
last_detection_time = 0
speak_cooldown = 0
training_mode = False
current_gesture_label = ""
cap = None
cap_lock = Lock()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train')
def train():
    return render_template('train.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global cap, detection_active, current_gesture, last_gesture, current_confidence, last_detection_time, speak_cooldown
    
    with cap_lock:
        if cap is None:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)
            
            if training_mode:
                cv2.putText(frame, f"Training Mode: {current_gesture_label}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                save_training_frame(frame, current_gesture_label)
            
            if detection_active and not training_mode:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                
                if results and results.multi_hand_landmarks:
                    hand_landmarks_list = []
                    
                    for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        hand_type = "Left" if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x < 0.5 else "Right"
                        hand_color = (0, 255, 0) if hand_type == "Left" else (0, 0, 255)
                        draw_landmarks_with_colors(frame, hand_landmarks, hand_color)
                        hand_landmarks_list.append(hand_landmarks)
                    
                    if hand_landmarks_list:
                        current_gesture, current_confidence = gesture_recognizer.recognize_gesture(hand_landmarks_list)
                        
                        if current_gesture != last_gesture:
                            last_gesture = current_gesture
                            last_detection_time = time.time()
                            if current_gesture != "Unknown Gesture" and time.time() > speak_cooldown:
                                speak_cooldown = time.time() + 2
                                text_to_speech(current_gesture)
                        
                        color = (0, 255, 0) if current_gesture != "Unknown Gesture" else (0, 0, 255)
                        cv2.putText(frame, f"{current_gesture} ({current_confidence}%)", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    hand_count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
                    cv2.putText(frame, f"Hands: {hand_count}", 
                               (frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def save_training_frame(frame, label):
    label_dir = os.path.join(TRAINING_DATA_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{label}_{timestamp}.jpg"
    filepath = os.path.join(label_dir, filename)
    
    cv2.imwrite(filepath, frame)

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global detection_active
    detection_active = True
    return jsonify({"status": "started"})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global detection_active
    detection_active = False
    return jsonify({"status": "stopped"})

@app.route('/get_gesture_text')
def get_gesture_text():
    return jsonify({
        "text": current_gesture,
        "confidence": current_confidence
    })

@app.route('/start_webcam_training', methods=['POST'])
def start_webcam_training():
    global training_mode, current_gesture_label
    data = request.json
    current_gesture_label = data.get('label', '').strip()
    if not current_gesture_label:
        return jsonify({"status": "error", "message": "Label cannot be empty"}), 400
    training_mode = True
    return jsonify({"status": "success"})

@app.route('/stop_training', methods=['POST'])
def stop_training():
    global training_mode
    training_mode = False
    return jsonify({"status": "success", "message": "Training stopped"})

@app.route('/upload_training_files', methods=['POST'])
def upload_training_files():
    if 'files' not in request.files:
        return jsonify({"status": "error", "message": "No files uploaded"}), 400
    
    label = request.form.get('label', '').strip()
    if not label:
        return jsonify({"status": "error", "message": "No label provided"}), 400
    
    files = request.files.getlist('files')
    label_dir = os.path.join(TRAINING_DATA_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    
    saved_count = 0
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(label_dir, filename)
            file.save(filepath)
            saved_count += 1
            
            if filename.lower().endswith(('.mp4', '.avi', '.mov')):
                extract_frames_from_video(filepath, label)
                os.remove(filepath)
    
    return jsonify({"status": "success", "count": saved_count})

def extract_frames_from_video(video_path, label):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    label_dir = os.path.join(TRAINING_DATA_DIR, label)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        if frame_count % 5 == 0:  # Extract every 5th frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{label}_frame_{frame_count}.jpg"
            filepath = os.path.join(label_dir, filename)
            cv2.imwrite(filepath, frame)
    
    cap.release()

@app.route('/get_training_labels')
def get_training_labels():
    labels = []
    if os.path.exists(TRAINING_DATA_DIR):
        labels = sorted([d for d in os.listdir(TRAINING_DATA_DIR) 
                       if os.path.isdir(os.path.join(TRAINING_DATA_DIR, d))])
    return jsonify({"labels": labels})

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        X, y, labels = prepare_dataset()
        
        if len(X) == 0 or len(labels) < 2:
            return jsonify({
                "status": "error", 
                "message": "Need at least 2 gesture classes with sufficient training data"
            }), 400

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        model = Sequential([
            Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(len(labels), activation='softmax')
        ])
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer,
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.2, patience=5)
        ]
        
        history = model.fit(X_train, y_train, 
                          epochs=50, 
                          batch_size=32, 
                          validation_data=(X_test, y_test),
                          callbacks=callbacks,
                          verbose=0)
        
        # Find the next available model number
        model_number = 1
        while os.path.exists(os.path.join(MODEL_DIR, f'model{model_number}.h5')):
            model_number += 1
        
        model_path = os.path.join(MODEL_DIR, f'model{model_number}.h5')
        model.save(model_path)
        
        labels_path = os.path.join(MODEL_DIR, f'labels{model_number}.txt')
        with open(labels_path, 'w') as f:
            f.write('\n'.join(labels))
        
        # Reload all models in the gesture recognizer
        gesture_recognizer.load_all_models()
        
        return jsonify({
            "status": "success",
            "accuracy": float(history.history['accuracy'][-1]),
            "val_accuracy": float(history.history['val_accuracy'][-1]),
            "labels": labels,
            "model_number": model_number,
            "total_models": len(gesture_recognizer.models)
        })

    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


def prepare_dataset():
    try:
        if not os.path.exists(TRAINING_DATA_DIR):
            return np.array([]), np.array([]), []
            
        X = []
        y = []
        labels = sorted([d for d in os.listdir(TRAINING_DATA_DIR) 
                       if os.path.isdir(os.path.join(TRAINING_DATA_DIR, d))])
        
        min_samples_per_class = 10
        class_counts = {}
        
        for label in labels:
            label_dir = os.path.join(TRAINING_DATA_DIR, label)
            sample_count = len([f for f in os.listdir(label_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            class_counts[label] = sample_count
        
        valid_labels = [label for label in labels if class_counts[label] >= min_samples_per_class]
        
        if len(valid_labels) < 2:
            return np.array([]), np.array([]), []
        
        label_to_index = {label: idx for idx, label in enumerate(valid_labels)}
        
        for label in valid_labels:
            label_dir = os.path.join(TRAINING_DATA_DIR, label)
            for filename in os.listdir(label_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(label_dir, filename)
                    landmarks = extract_landmarks_from_image(image_path)
                    if landmarks is not None and len(landmarks) == 126:
                        X.append(landmarks)
                        y.append(label_to_index[label])
        
        if not X:
            return np.array([]), np.array([]), []
        
        X = np.array(X)
        y = to_categorical(y, num_classes=len(valid_labels))
        
        return X, y, valid_labels
    
    except Exception as e:
        logger.error(f"Error in prepare_dataset: {str(e)}")
        return np.array([]), np.array([]), []

def extract_landmarks_from_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return None
        
        landmarks_list = []
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks_list.append(hand_landmarks)
        
        processed = gesture_recognizer.preprocess_landmarks(landmarks_list)
        if len(processed) < 126:
            processed = np.pad(processed, (0, 126 - len(processed)), 'constant')
        
        return processed
    except Exception as e:
        logger.error(f"Error extracting landmarks: {str(e)}")
        return None

def draw_landmarks_with_colors(image, landmarks, hand_color):
    for connection in mp_hands.HAND_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        
        finger_type = start_idx // 4
        color = (255, 0, 0) if finger_type == 0 else (  # Thumb - Blue
                0, 255, 0) if finger_type == 1 else (   # Index - Green
                0, 0, 255) if finger_type == 2 else (   # Middle - Red
                255, 255, 0) if finger_type == 3 else ( # Ring - Yellow
                255, 0, 255)                            # Pinky - Magenta
        
        blended_color = (
            int(color[0] * 0.7 + hand_color[0] * 0.3),
            int(color[1] * 0.7 + hand_color[1] * 0.3),
            int(color[2] * 0.7 + hand_color[2] * 0.3)
        )
        
        start_point = (int(landmarks.landmark[start_idx].x * image.shape[1]), 
                       int(landmarks.landmark[start_idx].y * image.shape[0]))
        end_point = (int(landmarks.landmark[end_idx].x * image.shape[1]), 
                     int(landmarks.landmark[end_idx].y * image.shape[0]))
        
        cv2.line(image, start_point, end_point, blended_color, 2)
    
    for idx, landmark in enumerate(landmarks.landmark):
        finger_type = idx // 4
        color = (255, 0, 0) if finger_type == 0 else (  # Thumb - Blue
                0, 255, 0) if finger_type == 1 else (   # Index - Green
                0, 0, 255) if finger_type == 2 else (   # Middle - Red
                255, 255, 0) if finger_type == 3 else ( # Ring - Yellow
                255, 0, 255)                            # Pinky - Magenta
        blended_color = (
            int(color[0] * 0.7 + hand_color[0] * 0.3),
            int(color[1] * 0.7 + hand_color[1] * 0.3),
            int(color[2] * 0.7 + hand_color[2] * 0.3)
        )
        center = (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
        cv2.circle(image, center, 3, blended_color, -1)

def text_to_speech(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"speech_{timestamp}.mp3"
        filepath = os.path.join(OUTPUT_DIR, filename)
        tts.save(filepath)
        return filename
    except Exception as e:
        logger.error(f"Error in text_to_speech: {str(e)}")
        return None

@app.route('/text_to_speech', methods=['POST'])
def text_to_speech_endpoint():
    data = request.json
    text = data.get("text")
    if text:
        mp3_filename = text_to_speech(text)
        if mp3_filename:
            return jsonify({"status": "success", "file": mp3_filename})
    return jsonify({"status": "error", "message": "No text provided"}), 400

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    text = data.get("text")
    lang = data.get("lang", "hi")
    if text:
        try:
            translation = translator.translate(text, dest=lang)
            return jsonify({
                "status": "success", 
                "translated_text": translation.text
            })
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": str(e)
            }), 500
    return jsonify({"status": "error", "message": "No text provided"}), 400

@app.route('/download_text', methods=['POST'])
def download_text():
    data = request.json
    text = data.get("text")
    if text:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output_{timestamp}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w") as f:
            f.write(text)
        return jsonify({"status": "success", "filename": filename})
    return jsonify({"status": "error", "message": "No text provided"}), 400

@app.route('/download_audio', methods=['POST'])
def download_audio():
    data = request.json
    text = data.get("text")
    if text:
        mp3_filename = text_to_speech(text)
        if mp3_filename:
            return jsonify({"status": "success", "filename": mp3_filename})
    return jsonify({"status": "error", "message": "No text provided"}), 400

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json
    image_data = data.get("image")
    if not image_data:
        return jsonify({"status": "error", "message": "No image data"}), 400
    
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results and results.multi_hand_landmarks:
            hand_landmarks_list = []
            for hand_landmarks in results.multi_hand_landmarks:
                hand_landmarks_list.append(hand_landmarks)
            
            if hand_landmarks_list:
                gesture, confidence = gesture_recognizer.recognize_gesture(hand_landmarks_list)
                return jsonify({
                    "status": "success",
                    "text": gesture,
                    "confidence": confidence
                })
        
        return jsonify({
            "status": "success",
            "text": "Unknown Gesture",
            "confidence": 0
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
    finally:
        with cap_lock:
            if cap is not None:
                cap.release()
