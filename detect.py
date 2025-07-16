import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

class MoveDetector:
    """Class to handle move detection from camera frames."""
    
    def __init__(self, model_path: str):
        """Initialize the detector with a trained model."""
        self.model = load_model(model_path)
        self.class_names = ['paper', 'rock', 'scissors']  # Same order as training
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a single frame for prediction."""
        # Resize to model input size
        resized = cv2.resize(frame, (128, 128))
        
        # Convert BGR to RGB (OpenCV uses BGR, model expects RGB)
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values to [0, 1]
        normalized = rgb_frame.astype(np.float32) / 255.0
        
        # Add batch dimension
        batch_frame = np.expand_dims(normalized, axis=0)
        
        return batch_frame
    
    def predict(self, frame: np.ndarray) -> tuple:
        """
        Predict the move from a frame.
        
        Args:
            frame: OpenCV frame (BGR format)
            
        Returns:
            tuple: (predicted_class, confidence)
        """
        processed_frame = self.preprocess_frame(frame)
        predictions = self.model.predict(processed_frame, verbose=0)
        
        # Get the class with highest probability
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        predicted_class = self.class_names[predicted_class_idx]
        
        return predicted_class, confidence

def detect_move(model_path: str, frame: np.ndarray) -> str:
    """
    Detect rock, paper, or scissors from a single frame.
    
    Args:
        model_path: Path to the trained Keras model
        frame: OpenCV frame (numpy array in BGR format)
        
    Returns:
        str: Predicted move ("rock", "paper", or "scissors")
    """
    # Load model
    model = load_model(model_path)
    
    # Preprocess frame
    resized = cv2.resize(frame, (128, 128))
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb_frame.astype(np.float32) / 255.0
    batch_frame = np.expand_dims(normalized, axis=0)
    
    # Predict
    predictions = model.predict(batch_frame, verbose=0)
    class_names = ['paper', 'rock', 'scissors']
    predicted_class_idx = np.argmax(predictions[0])
    
    return class_names[predicted_class_idx]

def detect_move_with_confidence(model_path: str, frame: np.ndarray) -> tuple:
    """
    Detect move and return confidence score.
    
    Args:
        model_path: Path to the trained Keras model
        frame: OpenCV frame (numpy array in BGR format)
        
    Returns:
        tuple: (predicted_move, confidence_score)
    """
    model = load_model(model_path)
    
    # Preprocess frame
    resized = cv2.resize(frame, (128, 128))
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb_frame.astype(np.float32) / 255.0
    batch_frame = np.expand_dims(normalized, axis=0)
    
    # Predict
    predictions = model.predict(batch_frame, verbose=0)
    class_names = ['paper', 'rock', 'scissors']
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    return class_names[predicted_class_idx], float(confidence)
