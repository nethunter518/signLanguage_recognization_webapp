import mediapipe as mp
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mp_hands = mp.solutions.hands

class GestureRecognizer:
    def __init__(self):
        self.models = []  # List to store multiple models
        self.labels_list = []  # List of label lists for each model
        self.use_models = False
        self.gesture_buffer = []
        self.buffer_size = 5
        self.confidence_threshold = 70
        
        try:
            self.load_all_models()
            self.initialize_gestures()
        except Exception as e:
            logger.error(f"Failed to initialize gesture recognizer: {str(e)}")
            self.use_models = False

    def load_all_models(self):
        """Load all available models from the model directory"""
        model_files = []
        label_files = []
        
        if os.path.exists("model"):
            for file in os.listdir("model"):
                if file.startswith("model") and file.endswith(".h5"):
                    model_num = file[5:-3]  # Extract number from "model1.h5"
                    model_files.append((int(model_num) if model_num.isdigit() else 0, file))
                elif file.startswith("labels") and file.endswith(".txt"):
                    label_num = file[6:-4]  # Extract number from "labels1.txt"
                    label_files.append((int(label_num) if label_num.isdigit() else 0, file))
        
        # Sort by model number and pair with labels
        model_files.sort()
        label_files.sort()
        
        # Load models in order
        for (model_num, model_file), (label_num, label_file) in zip(model_files, label_files):
            try:
                model_path = os.path.join("model", model_file)
                labels_path = os.path.join("model", label_file)
                
                model = load_model(model_path)
                with open(labels_path, 'r') as f:
                    labels = [line.strip() for line in f.readlines()]
                
                self.models.append(model)
                self.labels_list.append(labels)
                logger.info(f"Loaded model {model_file} with {len(labels)} gestures")
                
            except Exception as e:
                logger.error(f"Error loading model {model_file}: {str(e)}")
        
        self.use_models = len(self.models) > 0
        if self.use_models:
            logger.info(f"Using {len(self.models)} models for ensemble prediction")
        else:
            logger.info("No models found, using rule-based recognition")

    def initialize_gestures(self):
        """Initialize single and double hand gestures with error handling"""
        try:
            # Single hand gestures
            self.single_hand_gestures = {
                "Hello": self.is_hello,
                "Yes": self.is_yes,
                "No": self.is_no,
                "Thank You": self.is_thank_you,
                "Please": self.is_please,
                "Goodbye": self.is_goodbye,
                "Help": self.is_help,
                "Food": self.is_food,
                "Water": self.is_water,
                "Sorry": self.is_sorry,
                "I Love You": self.is_i_love_you,
                "Friend": self.is_friend,
                "Home": self.is_home,
                "School": self.is_school,
                "Work": self.is_work,
                "Name": self.is_name,
                "Age": self.is_age,
                "Where": self.is_where,
                "How": self.is_how,
                "Why": self.is_why
            }
            
            # Double hand gestures
            self.double_hand_gestures = {
                "Time": self.is_time,
                "Day": self.is_day,
                "Night": self.is_night,
                "Week": self.is_week,
                "Month": self.is_month,
                "Year": self.is_year,
                "Now": self.is_now,
                "Today": self.is_today,
                "Tomorrow": self.is_tomorrow,
                "Yesterday": self.is_yesterday,
                "Morning": self.is_morning,
                "Evening": self.is_evening,
                "Afternoon": self.is_afternoon,
                "Happy": self.is_happy,
                "Sad": self.is_sad,
                "Family": self.is_family,
                "Brother": self.is_brother,
                "Sister": self.is_sister,
                "Father": self.is_father,
                "Mother": self.is_mother
            }
        except Exception as e:
            logger.error(f"Failed to initialize gestures: {str(e)}")
            self.single_hand_gestures = {}
            self.double_hand_gestures = {}

    def preprocess_landmarks(self, landmarks_list):
        """Convert landmarks to normalized feature vector"""
        try:
            all_landmarks = []
            for landmarks in landmarks_list:
                hand_landmarks = []
                for landmark in landmarks.landmark:
                    hand_landmarks.extend([landmark.x, landmark.y, landmark.z])
                all_landmarks.extend(hand_landmarks)
            
            # Pad with zeros if only one hand detected
            if len(landmarks_list) < 2:
                all_landmarks.extend([0] * (126 - len(all_landmarks)))
            
            return np.array(all_landmarks)
        except Exception as e:
            logger.error(f"Error preprocessing landmarks: {str(e)}")
            return np.zeros(126)

    def recognize_gesture(self, landmarks_list):
        """Recognize gesture using ensemble of models or rule-based detection"""
        if not landmarks_list:
            return "Unknown Gesture", 0
        
        # Try model prediction first if available
        if self.use_models:
            try:
                processed = self.preprocess_landmarks(landmarks_list)
                predictions = []
                
                # Get predictions from all models
                for model, labels in zip(self.models, self.labels_list):
                    model_pred = model.predict(np.array([processed]), verbose=0)[0]
                    predictions.append((model_pred, labels))
                
                # Combine predictions using weighted voting
                combined_pred = self.combine_predictions(predictions)
                if combined_pred:
                    gesture, confidence = combined_pred
                    if confidence >= self.confidence_threshold:
                        self._update_gesture_buffer(gesture, confidence)
                        return self._get_most_consistent_gesture()
                
            except Exception as e:
                logger.warning(f"Model prediction failed: {str(e)}")
        
        # Fallback to rule-based detection
        return self.rule_based_detection(landmarks_list)

    def combine_predictions(self, predictions):
        """Combine predictions from multiple models using weighted average"""
        try:
            # Create weighted average of predictions
            weighted_preds = defaultdict(float)
            total_weight = 0
            
            for pred, labels in predictions:
                pred_index = np.argmax(pred)
                confidence = float(pred[pred_index])
                gesture = labels[pred_index]
                
                # Weight by confidence and model performance
                weight = confidence
                weighted_preds[gesture] += weight
                total_weight += weight
            
            if not weighted_preds:
                return None
                
            # Normalize and get best prediction
            for gesture in weighted_preds:
                weighted_preds[gesture] /= total_weight
                
            best_gesture = max(weighted_preds, key=weighted_preds.get)
            return best_gesture, weighted_preds[best_gesture] * 100
            
        except Exception as e:
            logger.error(f"Error combining predictions: {str(e)}")
            return None

    def rule_based_detection(self, landmarks_list):
        """Fallback to rule-based gesture detection"""
        if len(landmarks_list) == 1:
            landmarks = landmarks_list[0]
            for gesture_name, gesture_check in self.single_hand_gestures.items():
                if gesture_check(landmarks):
                    self._update_gesture_buffer(gesture_name, 90)
                    return self._get_most_consistent_gesture()
        
        elif len(landmarks_list) == 2:
            try:
                left_landmarks = landmarks_list[0]
                right_landmarks = landmarks_list[1]
                
                # Ensure left hand is actually on the left
                if left_landmarks.landmark[mp_hands.HandLandmark.WRIST].x > \
                   right_landmarks.landmark[mp_hands.HandLandmark.WRIST].x:
                    left_landmarks, right_landmarks = right_landmarks, left_landmarks
                    
                for gesture_name, gesture_check in self.double_hand_gestures.items():
                    if gesture_check(left_landmarks, right_landmarks):
                        self._update_gesture_buffer(gesture_name, 90)
                        return self._get_most_consistent_gesture()
            except Exception as e:
                logger.error(f"Error in double hand detection: {str(e)}")
        
        return "Unknown Gesture", 0

    def _update_gesture_buffer(self, gesture, confidence):
        """Maintain a buffer of recent gestures for temporal smoothing"""
        if gesture != "Unknown Gesture" and confidence >= self.confidence_threshold:
            self.gesture_buffer.append((gesture, confidence))
            if len(self.gesture_buffer) > self.buffer_size:
                self.gesture_buffer.pop(0)

    def _get_most_consistent_gesture(self):
        """Get the most consistent gesture from buffer"""
        if not self.gesture_buffer:
            return "Unknown Gesture", 0
        
        # Count occurrences of each gesture in buffer
        gesture_counts = {}
        for gesture, conf in self.gesture_buffer:
            if gesture not in gesture_counts:
                gesture_counts[gesture] = {'count': 0, 'total_conf': 0}
            gesture_counts[gesture]['count'] += 1
            gesture_counts[gesture]['total_conf'] += conf
        
        # Get gesture with highest count and average confidence
        if gesture_counts:
            best_gesture = max(gesture_counts.keys(), 
                              key=lambda g: (gesture_counts[g]['count'], 
                                           gesture_counts[g]['total_conf']))
            
            avg_confidence = gesture_counts[best_gesture]['total_conf'] / \
                            gesture_counts[best_gesture]['count']
            
            # Only return if gesture appears in majority of buffer
            if gesture_counts[best_gesture]['count'] >= (self.buffer_size // 2 + 1):
                return best_gesture, avg_confidence
        
        return "Unknown Gesture", 0

    def distance(self, point1, point2):
        """Calculate Euclidean distance between two landmarks"""
        return math.sqrt((point1.x - point2.x)**2 + 
                        (point1.y - point2.y)**2 + 
                        (point1.z - point2.z)**2)

    # =============================================
    # Single Hand Gesture Detection Methods
    # =============================================    

    def is_hello(self, landmarks):
        try:
            thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            return (thumb_tip.y < index_tip.y and thumb_tip.x > index_tip.x and 
                    middle_tip.y > index_tip.y)
        except:
            return False

    def is_yes(self, landmarks):
        try:
            index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            return (index_tip.y < middle_tip.y and 
                    abs(index_tip.x - middle_tip.x) < 0.1 and
                    ring_tip.y > middle_tip.y and
                    pinky_tip.y > middle_tip.y)
        except:
            return False

    def is_no(self, landmarks):
        try:
            thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            return (middle_tip.y < ring_tip.y and 
                    abs(middle_tip.x - ring_tip.x) < 0.1 and
                    thumb_tip.x > index_tip.x)
        except:
            return False

    def is_thank_you(self, landmarks):
        try:
            thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            return (thumb_tip.x > pinky_tip.x and 
                    thumb_tip.y < pinky_tip.y and
                    index_tip.y > thumb_tip.y)
        except:
            return False

    def is_please(self, landmarks):
        try:
            palm = landmarks.landmark[mp_hands.HandLandmark.WRIST]
            fingers = [
                landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            ]
            thumb = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            return (all(finger.y < palm.y for finger in fingers) and
                    thumb.y < palm.y)
        except:
            return False

    def is_goodbye(self, landmarks):
        try:
            thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            return (thumb_tip.x < pinky_tip.x and 
                    thumb_tip.y > pinky_tip.y and
                    index_tip.y > thumb_tip.y)
        except:
            return False

    def is_help(self, landmarks):
        try:
            thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            return (thumb_tip.x < index_tip.x and 
                    thumb_tip.y > index_tip.y and
                    middle_tip.y > index_tip.y)
        except:
            return False

    def is_food(self, landmarks):
        try:
            thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            return (thumb_tip.x > index_tip.x and 
                    thumb_tip.x > middle_tip.x and 
                    index_tip.y < middle_tip.y and
                    self.distance(thumb_tip, index_tip) < 0.15)
        except:
            return False

    def is_water(self, landmarks):
        try:
            pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            return (pinky_tip.y < ring_tip.y and 
                    pinky_tip.x < ring_tip.x and
                    middle_tip.y > ring_tip.y)
        except:
            return False

    def is_sorry(self, landmarks):
        try:
            palm = landmarks.landmark[mp_hands.HandLandmark.WRIST]
            fingers = [
                landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            ]
            thumb = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            return (all(finger.y > palm.y for finger in fingers) and
                    thumb.y > palm.y)
        except:
            return False

    def is_i_love_you(self, landmarks):
        try:
            thumb = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            pinky = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            middle = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            return (thumb.x < index.x and 
                    index.y < pinky.y and 
                    pinky.x > thumb.x and
                    middle.y > ring.y and
                    ring.y > pinky.y)
        except:
            return False

    def is_friend(self, landmarks):
        try:
            index = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            return (abs(index.x - middle.x) < 0.05 and 
                    abs(index.y - middle.y) < 0.05 and
                    abs(ring.x - pinky.x) < 0.05 and
                    abs(ring.y - pinky.y) < 0.05)
        except:
            return False

    def is_home(self, landmarks):
        try:
            thumb = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            fingers = [
                landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            ]
            return (all(finger.y < thumb.y for finger in fingers) and
                    thumb.x < fingers[0].x)
        except:
            return False

    def is_school(self, landmarks):
        try:
            thumb = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            return (thumb.y > index.y and 
                    abs(thumb.x - index.x) > 0.1 and
                    middle.y > index.y)
        except:
            return False

    def is_work(self, landmarks):
        try:
            wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
            fingers = [
                landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            ]
            return (all(abs(finger.x - wrist.x) < 0.1 for finger in fingers) and
                    fingers[0].y > fingers[1].y)
        except:
            return False

    def is_name(self, landmarks):
        try:
            index = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            thumb = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            return (index.y < middle.y and 
                    thumb.x > index.x and
                    self.distance(index, thumb) < 0.2)
        except:
            return False

    def is_age(self, landmarks):
        try:
            thumb = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            return (thumb.y > index.y and 
                    index.y > middle.y and
                    thumb.x < index.x)
        except:
            return False

    def is_where(self, landmarks):
        try:
            index = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            return (index.y < middle.y and 
                    middle.y < ring.y and
                    abs(index.x - ring.x) < 0.1)
        except:
            return False

    def is_how(self, landmarks):
        try:
            thumb = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            pinky = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            return (thumb.x < index.x and 
                    index.y < pinky.y and
                    pinky.x > thumb.x)
        except:
            return False

    def is_why(self, landmarks):
        try:
            thumb = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            return (thumb.y > index.y and 
                    index.y > middle.y and
                    thumb.x > index.x)
        except:
            return False
            
            
    # =============================================
    # Double Hand Gesture Detection Methods
    # =============================================
    
    
    # Enhanced double hand gesture detection methods
    def is_time(self, left_landmarks, right_landmarks):
        try:
            left_thumb = left_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            right_index = right_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            left_index = left_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            return (left_thumb.y < right_index.y and 
                    abs(left_thumb.x - right_index.x) < 0.1 and
                    left_index.y > left_thumb.y)
        except:
            return False

    def is_day(self, left_landmarks, right_landmarks):
        try:
            left_index = left_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            right_index = right_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            left_thumb = left_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            return (left_index.y < right_index.y and 
                    abs(left_index.x - right_index.x) < 0.1 and
                    left_thumb.y > left_index.y)
        except:
            return False

    def is_night(self, left_landmarks, right_landmarks):
        try:
            left_middle = left_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            right_middle = right_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            left_thumb = left_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            return (left_middle.y < right_middle.y and 
                    abs(left_middle.x - right_middle.x) < 0.1 and
                    left_thumb.x > left_middle.x)
        except:
            return False

    def is_week(self, left_landmarks, right_landmarks):
        try:
            left_ring = left_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            right_ring = right_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            left_pinky = left_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            return (left_ring.y < right_ring.y and 
                    abs(left_ring.x - right_ring.x) < 0.1 and
                    left_pinky.y > left_ring.y)
        except:
            return False

    def is_month(self, left_landmarks, right_landmarks):
        try:
            left_pinky = left_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            right_pinky = right_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            left_ring = left_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            return (left_pinky.y < right_pinky.y and 
                    abs(left_pinky.x - right_pinky.x) < 0.1 and
                    left_ring.y > left_pinky.y)
        except:
            return False

    def is_year(self, left_landmarks, right_landmarks):
        try:
            left_wrist = left_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            right_wrist = right_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            left_thumb = left_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            return (abs(left_wrist.y - right_wrist.y) < 0.1 and 
                    abs(left_wrist.x - right_wrist.x) < 0.2 and
                    left_thumb.y < left_wrist.y)
        except:
            return False

    def is_now(self, left_landmarks, right_landmarks):
        try:
            left_thumb = left_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            right_thumb = right_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            left_index = left_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            return (abs(left_thumb.y - right_thumb.y) < 0.05 and 
                    abs(left_thumb.x - right_thumb.x) < 0.05 and
                    left_index.y > left_thumb.y)
        except:
            return False

    def is_today(self, left_landmarks, right_landmarks):
        try:
            left_index = left_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            right_index = right_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            left_middle = left_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            return (abs(left_index.y - right_index.y) < 0.05 and 
                    abs(left_index.x - right_index.x) < 0.05 and
                    left_middle.y > left_index.y)
        except:
            return False

    def is_tomorrow(self, left_landmarks, right_landmarks):
        try:
            left_middle = left_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            right_middle = right_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            left_ring = left_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            return (left_middle.y > right_middle.y and 
                    abs(left_middle.x - right_middle.x) < 0.1 and
                    left_ring.y > left_middle.y)
        except:
            return False

    def is_yesterday(self, left_landmarks, right_landmarks):
        try:
            left_ring = left_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            right_ring = right_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            left_pinky = left_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            return (left_ring.y > right_ring.y and 
                    abs(left_ring.x - right_ring.x) < 0.1 and
                    left_pinky.y > left_ring.y)
        except:
            return False

    def is_morning(self, left_landmarks, right_landmarks):
        try:
            left_pinky = left_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            right_pinky = right_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            left_thumb = left_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            return (left_pinky.y < right_pinky.y and 
                    abs(left_pinky.x - right_pinky.x) > 0.2 and
                    left_thumb.x < left_pinky.x)
        except:
            return False

    def is_evening(self, left_landmarks, right_landmarks):
        try:
            left_wrist = left_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            right_wrist = right_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            left_thumb = left_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            return (left_wrist.y > right_wrist.y and 
                    abs(left_wrist.x - right_wrist.x) > 0.2 and
                    left_thumb.y < left_wrist.y)
        except:
            return False

    def is_afternoon(self, left_landmarks, right_landmarks):
        try:
            left_thumb = left_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            right_thumb = right_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            left_index = left_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            return (abs(left_thumb.y - right_thumb.y) < 0.1 and 
                    abs(left_thumb.x - right_thumb.x) > 0.3 and
                    left_index.y > left_thumb.y)
        except:
            return False

    def is_happy(self, left_landmarks, right_landmarks):
        try:
            left_index = left_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            right_index = right_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            left_thumb = left_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            return (left_index.y < right_index.y and 
                    left_index.x > right_index.x and
                    left_thumb.y < left_index.y)
        except:
            return False

    def is_sad(self, left_landmarks, right_landmarks):
        try:
            left_middle = left_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            right_middle = right_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            left_thumb = left_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            return (left_middle.y > right_middle.y and 
                    left_middle.x < right_middle.x and
                    left_thumb.y > left_middle.y)
        except:
            return False

    def is_family(self, left_landmarks, right_landmarks):
        try:
            left_thumb = left_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            right_thumb = right_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            left_index = left_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            return (abs(left_thumb.y - right_thumb.y) < 0.1 and 
                    abs(left_thumb.x - right_thumb.x) < 0.2 and
                    left_index.y < left_thumb.y)
        except:
            return False

    def is_brother(self, left_landmarks, right_landmarks):
        try:
            left_index = left_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            right_index = right_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            left_thumb = left_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            return (left_index.y < right_index.y and 
                    abs(left_index.x - right_index.x) < 0.1 and
                    left_thumb.x > left_index.x)
        except:
            return False

    def is_sister(self, left_landmarks, right_landmarks):
        try:
            left_middle = left_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            right_middle = right_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            left_thumb = left_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            return (left_middle.y < right_middle.y and 
                    abs(left_middle.x - right_middle.x) < 0.1 and
                    left_thumb.x < left_middle.x)
        except:
            return False

    def is_father(self, left_landmarks, right_landmarks):
        try:
            left_ring = left_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            right_ring = right_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            left_thumb = left_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            return (left_ring.y < right_ring.y and 
                    abs(left_ring.x - right_ring.x) < 0.1 and
                    left_thumb.y > left_ring.y)
        except:
            return False

    def is_mother(self, left_landmarks, right_landmarks):
        try:
            left_pinky = left_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            right_pinky = right_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            left_thumb = left_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            return (left_pinky.y < right_pinky.y and 
                    abs(left_pinky.x - right_pinky.x) < 0.1 and
                    left_thumb.y < left_pinky.y)
        except:
            return False
