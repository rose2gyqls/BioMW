import time
import csv
import asyncio
import mediapipe as mp
import cv2
import numpy as np
import joblib
import pandas as pd
from image_input import get_image_input_from_folder
from get_image import fetch_image
from ear_detection import detect_EAR
from k_emotion_analysis import analyze_emotion
from intervention import process_intervention
from sklearn.preprocessing import MinMaxScaler
from pycaret.classification import load_model
from pycaret.classification import load_model, predict_model

# Joint position extraction with (x, y, z)
def extract_joint_positions(frame):
    try:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        joint_positions = []
        if results.pose_landmarks:
            h, w, _ = frame.shape
            for id, landmark in enumerate(results.pose_landmarks.landmark):
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cz = landmark.z  # Add the z-coordinate (depth)
                joint_positions.append((id, cx, cy, cz))

        return joint_positions
    except Exception as e:
        print(f"[ERROR] extract_joint_positions: {e}")
        return []

# Preprocess joint positions for anomaly detection (x, y only)
def preprocess_joint_positions_2d(joint_positions):
    features = [(x, y) for _, x, y, _ in joint_positions]
    return np.array(features)  # Convert to numpy array

# Preprocess joint positions for pose prediction (x, y, z)
def preprocess_joint_positions_3d(joint_positions):
    try:
        features = [(x, y, z) for _, x, y, z in joint_positions]
        return np.array(features)
    except Exception as e:
        print(f"[ERROR] preprocess_joint_positions_3d: {e}")
        return np.array([])

# Abnormality detection using (x, y)
def detect_abnormality(joint_positions, isolation_forest_model, scaler):
    print("detect_abnormality")
    if not joint_positions:
        return {"total": 0, "abnormal_count": 0, "result": "No Data", "probability": 0.0}

    # Preprocess joint positions (extract only x, y)
    features = preprocess_joint_positions_2d(joint_positions)

    # Normalize features
    scaled_features = scaler.fit_transform(features)  # Scale (x, y) values

    # Perform anomaly detection
    anomaly_scores = isolation_forest_model.decision_function(scaled_features)
    threshold = -0.1  # Threshold for anomaly detection
    outlier_mask = anomaly_scores < threshold  # True for outliers

    # Count total and abnormal points
    total_points = len(features)
    abnormal_count = np.sum(outlier_mask)

    # Result based on abnormal count
    abnormal_ratio = abnormal_count / total_points
    result = "Abnormal" if abnormal_ratio > 0.2 else "Normal"  # Threshold: 20% abnormal points

    abnormal_probability = round(abnormal_ratio, 2)  # Normalize to 0-1 with two decimal places

    # Debug information
    print(f"Total Points: {total_points}, Abnormal Count: {abnormal_count}, Abnormal Ratio: {abnormal_ratio:.2f}")
    print(f"Result: {result}, Abnormality Probability: {abnormal_probability}")

    return {"total": total_points, "abnormal_count": abnormal_count, "result": result, "probability": abnormal_probability}

# Pose prediction using 3D joint positions
def predict_pose_with_z(joint_positions, pose_model):

    if not joint_positions or len(joint_positions) != 33:
        return "Invalid Input: joint_positions must have 33 rows"

    try:
        features = np.array([(x, y, z) for _, x, y, z in joint_positions])


        flattened_features = [coord for tuple_ in features for coord in tuple_]
        input_df = pd.DataFrame([flattened_features], columns=[
            f"{axis}_{i}" for i in range(33) for axis in ["X", "Y", "Z"]
        ])

        prediction_result = predict_model(pose_model, data=input_df)
        print(prediction_result)

        predicted_pose = prediction_result["prediction_label"][0]
        prediction_confidence = prediction_result["prediction_score"][0]
        print(f"[DEBUG] Predicted Pose: {predicted_pose} (Confidence: {prediction_confidence})")

        return predicted_pose

    except Exception as e:
        print(f"[ERROR] Error in predict_pose_with_z: {e}")
        return "Error"


    
# Save results to CSV
def save_to_csv(data, filename="../data/output.csv"):
    print("save_to_csv")
    header = ["Timestamp", "ear", "emotion", "abnormality_probability", "predicted_pose"]
    try:
        with open(filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(header)  # Write header only if the file is empty
            writer.writerow(data)
    except Exception as e:
        print(f"Error saving to CSV: {e}")

# Coroutine to fetch images
async def get_image():
    while True:
        try:
            print(f"[{time.strftime('%H:%M:%S')}] Calling get_image function...")
            fetch_image()  # fetch_image is assumed to be blocking; adjust if necessary
            await asyncio.sleep(10)  # Wait 10 seconds
        except Exception as e:

            break

# Coroutine to detect mind wandering
async def detect_mind_wandering():
    image_folder_path = "../image"
    num_frames = 10
    # Load the model
    isolation_forest_model = joblib.load('../model/isolation_forest_model.pkl')
    pose_model = joblib.load('../model/pose_classification_model_pycaret.pkl')

    # scaler.pkl load
    scaler_path = "C:/Users/User/Desktop/mw/model/scaler.pkl"
    scaler = joblib.load(scaler_path)

    while True:
        try:           
            # Process frames
            frames = get_image_input_from_folder(image_folder_path, num_frames, wait_time=3)
            
            # EAR and emotion detection
            is_ear = detect_EAR(frames)
            ear_status = 1 if is_ear[0] and is_ear[1] > 0 else 0
            emotion_result = analyze_emotion(frames)
            emotion = emotion_result['emotion'] if 'emotion' in emotion_result else "neutral"
            joint_positions = extract_joint_positions(frames[0])
            if not joint_positions or len(joint_positions) != 33:
                continue
            
            # Abnormality detection using (x, y)
            abnormality = detect_abnormality(joint_positions, isolation_forest_model, scaler)

            # Predict pose using 3D joint positions
            predicted_pose = predict_pose_with_z(joint_positions, pose_model)

            # Save to CSV
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            save_to_csv([timestamp, ear_status, emotion, abnormality['probability'], predicted_pose])

            await asyncio.sleep(3)
        except Exception as e:
            print(f"[ERROR] Error in detect_mind_wandering loop: {e}")

# Main coroutine
async def main():
    await asyncio.gather(
        get_image(),
        detect_mind_wandering(),
        process_intervention()
    )

if __name__ == "__main__":
    asyncio.run(main())
