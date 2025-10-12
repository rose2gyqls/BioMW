from deepface import DeepFace

def analyze_emotion(frames):
    results = []

    for frame in frames:
        try:
            # Use DeepFace to analyze emotions in the frame
            analysis = DeepFace.analyze(frame, actions=['emotion'])

            # Check if the result is a list and extract the first element if necessary
            if isinstance(analysis, list):
                analysis = analysis[0]

            # Extract dominant emotion and confidence
            results.append({
                'dominant_emotion': analysis['dominant_emotion'],
                'confidence': analysis['emotion'][analysis['dominant_emotion']]
            })
        except Exception as e:
            print(f"Emotion analysis failed for a frame: {e}")

    if results:
        # Find the emotion with the highest average confidence
        avg_emotions = {}
        for res in results:
            emotion = res['dominant_emotion']
            confidence = res['confidence']
            if emotion not in avg_emotions:
                avg_emotions[emotion] = []
            avg_emotions[emotion].append(confidence)

        # Get dominant emotion
        dominant_emotion = max(avg_emotions, key=lambda x: sum(avg_emotions[x]) / len(avg_emotions[x]))
        confidence = sum(avg_emotions[dominant_emotion]) / len(avg_emotions[dominant_emotion])
        return {'emotion': dominant_emotion, 'confidence': confidence}
    else:
        return {'emotion': None, 'confidence': 0}
