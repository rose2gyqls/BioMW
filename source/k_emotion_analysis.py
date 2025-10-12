import onnxruntime
import os
import numpy as np
import cv2
from deepface import DeepFace

# Path to the ONNX model
onnx_model_path = 'C:/Users/User/Desktop/mw/model/k_emotion_model.onnx'
print(os.path.exists(onnx_model_path))  # True if the model file path is correct
# Load the ONNX Runtime session
onnx_session = onnxruntime.InferenceSession(onnx_model_path)

input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# Face detection and preprocessing function
def preprocess_frame_with_face_detection(frame):
    try:
        # Detect and extract faces
        faces = DeepFace.extract_faces(
            img_path=frame,
            detector_backend='opencv',
            enforce_detection=False
        )

        if len(faces) > 0:
            # Get the first face image
            face_image = faces[0]['face']

            if face_image.dtype != np.uint8:
                face_image = (face_image * 255).astype(np.uint8)

            # Resize to match ONNX model input size
            resized_face = cv2.resize(face_image, (96, 96))

            # Convert to grayscale and normalize
            grayscale_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
            normalized_face = grayscale_face / 255.0  # Normalize to [0, 1]

            # Convert to model input format
            input_array = np.expand_dims(normalized_face, axis=(0, -1)).astype(np.float32)
            return input_array
        else:
            print("No face detected in frame.")
            return None
    except Exception as e:
        print(f"Error in face detection and preprocessing: {e}")
        return None

# Emotion analysis function
def analyze_emotion(frames):
    results = []

    for frame in frames:
        try:
            # Detect face and preprocess
            input_data = preprocess_frame_with_face_detection(frame)
            if input_data is None:
                continue  # Skip to next frame if no face is detected

            # Perform inference with the ONNX model
            onnx_outputs = onnx_session.run([output_name], {input_name: input_data})
            emotion_scores = onnx_outputs[0].squeeze()  # Flatten the output array if necessary

            # Calculate emotion and confidence
            emotion_labels = ['angry', 'embarrassed', 'happy', 'neutral', 'sad']  # Example emotions
            dominant_emotion_idx = np.argmax(emotion_scores)
            dominant_emotion = emotion_labels[dominant_emotion_idx]
            confidence = emotion_scores[dominant_emotion_idx]

            results.append({
                'dominant_emotion': dominant_emotion,
                'confidence': confidence
            })
        except Exception as e:
            print(f"Emotion analysis failed for a frame: {e}")

    if results:
        # Summary of the results
        avg_emotions = {}
        for res in results:
            emotion = res['dominant_emotion']
            confidence = res['confidence']
            if emotion not in avg_emotions:
                avg_emotions[emotion] = []
            avg_emotions[emotion].append(confidence)

        # Calculate the final dominant emotion
        dominant_emotion = max(avg_emotions, key=lambda x: sum(avg_emotions[x]) / len(avg_emotions[x]))
        confidence = sum(avg_emotions[dominant_emotion]) / len(avg_emotions[dominant_emotion])
        return {'emotion': dominant_emotion, 'confidence': confidence}
    else:
        return {'emotion': None, 'confidence': 0}
