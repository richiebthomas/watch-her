import cv2
import numpy as np
import gradio as gr
from pathlib import Path
import threading
import time
import os
import urllib.request

class GenderClassifier:
    def __init__(self):
        self.face_net = None
        self.gender_net = None
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.gender_list = ['Male', 'Female']
        self.models_loaded = False
        
    def download_models(self):
        """Download required model files if they don't exist"""
        model_files = {
            'opencv_face_detector_uint8.pb': 'https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/opencv_face_detector_uint8.pb',
            'opencv_face_detector.pbtxt': 'https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/opencv_face_detector.pbtxt',
            'gender_deploy.prototxt': 'https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/gender_deploy.prototxt',
            'gender_net.caffemodel': 'https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/gender_net.caffemodel'
        }
        
        for filename, url in model_files.items():
            if not os.path.exists(filename):
                print(f"Downloading {filename}...")
                try:
                    urllib.request.urlretrieve(url, filename)
                    print(f"Downloaded {filename} successfully!")
                except Exception as e:
                    print(f"Error downloading {filename}: {e}")
                    return False
        return True
        
    def load_models(self):
        """Load face detection and gender classification models"""
        try:
            # Download models if they don't exist
            if not self.download_models():
                return False
                
            # Face detection model (DNN-based)
            self.face_net = cv2.dnn.readNetFromTensorflow(
                'opencv_face_detector_uint8.pb',
                'opencv_face_detector.pbtxt'
            )
            
            # Gender classification model
            self.gender_net = cv2.dnn.readNetFromCaffe(
                'gender_deploy.prototxt',
                'gender_net.caffemodel'
            )
            
            self.models_loaded = True
            print("Models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
            
    def detect_faces(self, frame):
        """Detect faces in the frame using DNN"""
        if not self.models_loaded:
            return []
            
        frameHeight, frameWidth = frame.shape[:2]
        
        # Create blob from frame
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter weak detections
            if confidence > 0.7:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                
                faces.append([x1, y1, x2, y2])
                
        return faces
    
    def predict_gender(self, face_img):
        """Predict gender for a face image"""
        if not self.models_loaded:
            return "Unknown", 0.0
            
        # Prepare the face for gender prediction
        blob = cv2.dnn.blobFromImage(
            face_img, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False
        )
        
        # Predict gender
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender_idx = gender_preds[0].argmax()
        gender = self.gender_list[gender_idx]
        confidence = gender_preds[0][gender_idx]
        
        return gender, confidence
    
    def process_frame(self, frame):
        """Process a single frame for gender classification"""
        if not self.models_loaded:
            return frame, []
            
        # Detect faces
        faces = self.detect_faces(frame)
        
        results = []
        for (x1, y1, x2, y2) in faces:
            # Extract face region
            face = frame[max(0, y1):min(frame.shape[0], y2),
                        max(0, x1):min(frame.shape[1], x2)]
            
            if face.size > 0:
                # Predict gender
                gender, confidence = self.predict_gender(face)
                
                # Draw bounding box and label
                color = (0, 255, 0) if gender == 'Male' else (255, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Create label with gender and confidence
                label = f"{gender}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Draw label background
                cv2.rectangle(frame, (x1, y1-label_size[1]-10), 
                            (x1+label_size[0], y1), color, -1)
                
                # Draw label text
                cv2.putText(frame, label, (x1, y1-5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                results.append({
                    'bbox': (x1, y1, x2, y2),
                    'gender': gender,
                    'confidence': confidence
                })
        
        return frame, results

# Initialize the classifier
classifier = GenderClassifier()

def process_video_frame(frame):
    """Process video frame for Gradio interface"""
    print(f"Processing frame: {frame is not None}")  # Debug
    
    if frame is None:
        print("Frame is None, returning None")  # Debug
        return frame
    
    # Check if models are loaded
    if not classifier.models_loaded:
        print("Models not loaded, loading now...")  # Debug
        classifier.load_models()
        if not classifier.models_loaded:
            print("Failed to load models")  # Debug
            return frame
    
    print(f"Frame shape: {frame.shape}")  # Debug
    
    # Convert RGB to BGR (OpenCV format)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Process the frame
    processed_frame, results = classifier.process_frame(frame_bgr)
    print(f"Found {len(results)} faces")  # Debug
    
    # Convert back to RGB for Gradio
    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    
    return processed_frame_rgb

def initialize_models():
    """Initialize models when app starts"""
    print("Initializing models...")  # Debug
    status = classifier.load_models()
    print(f"Model loading status: {status}")  # Debug
    if status:
        return "✅ Models loaded successfully! Ready for gender classification."
    else:
        return "❌ Failed to load models. Please check your internet connection."

def get_model_status():
    """Get current model status"""
    if classifier.models_loaded:
        return "✅ Models loaded and ready"
    else:
        return "❌ Models not loaded"

# Create Gradio interface using Interface for better real-time processing
def create_interface():
    # Initialize models first
    print("Creating interface and loading models...")
    classifier.load_models()
    
    # Create a simple interface for real-time webcam processing
    demo = gr.Interface(
        fn=process_video_frame,
        inputs=gr.Image(sources=["webcam"], streaming=True),
        outputs=gr.Image(streaming=True),
        title="🛡️ KLEOS 3.0 - Team: The boizz",
        description="""
        # Prototype Submission
        ## Women CCTV Safety: Currently only has functionalities for detecting persons and their gender
        
        **🚀 How to Start:**
        1. **Turn on your webcam** and allow camera access when prompted
        2. **Click the "Record" button** to start processing
        3. The system will automatically detect faces and classify gender in real-time
        
        **🎯 Features:**
        - Real-time face detection and gender classification
        - Color-coded bounding boxes (Green for Male, Magenta for Female)
        - Confidence scores displayed with each prediction
        
        **🤖 AI Models Used:**
        - **Face Detection**: OpenCV DNN-based face detector (Tensorflow model)
          - Model: `opencv_face_detector_uint8.pb` with configuration `opencv_face_detector.pbtxt`
          - High accuracy face detection with confidence threshold of 0.7
        
        - **Gender Classification**: Caffe-based deep learning model
          - Model: `gender_net.caffemodel` with `gender_deploy.prototxt`
          - Pre-trained on large datasets for robust gender classification
          - Input: 227x227 pixel face images with mean normalization
        
        **📋 Technical Notes:**
        - First run automatically downloads model files (~100MB)
        - Works best with good lighting conditions
        - Model accuracy may vary across different demographics
        - Processing runs at ~10 FPS for real-time performance
        """,
        live=True,
        allow_flagging="never"
    )
    
    return demo

# Launch the application
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7490,
        share=False,
        debug=True,
        show_error=True
    )