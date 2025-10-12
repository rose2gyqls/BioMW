import os
import openai
import asyncio
from datetime import datetime
import json
import pandas as pd
import numpy as np
from enum import Enum
from typing import Optional, Tuple

class FacialExpression(Enum):
    ANGRY = 'angry'
    DISGUST = 'disgust'
    FEAR = 'fear'
    HAPPY = 'happy'
    SAD = 'sad'
    SURPRISE = 'surprise'
    NEUTRAL = 'neutral'

class InterventionLevel(Enum):
    NONE = 'none'
    WEAK = 'weak'
    STRONG = 'strong'

class MWInterventionGenerator:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key
        
        # Emotion-based intervention score weights
        self.emotion_weights = {
            FacialExpression.ANGRY: 0.8,
            FacialExpression.DISGUST: 0.7,
            FacialExpression.FEAR: 0.6,
            FacialExpression.SAD: 0.7,
            FacialExpression.SURPRISE: 0.4,
            FacialExpression.HAPPY: 0.2,
            FacialExpression.NEUTRAL: 0.3
        }
        
        # List to store intervention history
        self.intervention_history = []

    def calculate_intervention_score(self, ear, facial_expression, posture_change):
        """
        Calculate intervention score based on eye status, facial expression, and posture change.
        """
        try:
            emotion = FacialExpression(facial_expression.lower())
            emotion_score = self.emotion_weights[emotion]
        except (ValueError, KeyError):
            emotion_score = self.emotion_weights[FacialExpression.NEUTRAL]
        
        weights = {
            'ear': 0.3,
            'facial': 0.4,
            'posture': 0.3
        }
        
        intervention_score = (
            weights['ear'] * ear +
            weights['facial'] * emotion_score +
            weights['posture'] * posture_change
        )
        
        return min(max(intervention_score, 0), 1)
    
    def determine_intervention_level(self, intervention_score) -> Tuple[InterventionLevel, Optional[str]]:
        """
        Determine the intervention level based on the intervention score.
        Returns:
        - Tuple[InterventionLevel, Optional[str]]: (Intervention level, prompt)
        """
        if intervention_score >= 0.3:
            return InterventionLevel.STRONG, self._create_strong_intervention_prompt()
        elif intervention_score >= 0.2:
            return InterventionLevel.WEAK, self._create_weak_intervention_prompt()
        else:
            return InterventionLevel.NONE, None

    def _create_system_prompt(self):
        """
        Generate the default system prompt.
        """
        return """
        You are an expert in detecting Mind-Wandering (MW) in users during mindfulness sessions and generating appropriate intervention phrases.
        Consider the user's eye status, emotional state, and posture changes to generate responses that help restore their focus.
        Please select one intervention type and respond like a counselor. Provide only the response without any additional explanation.
        """

    def _create_weak_intervention_prompt(self):
        """
        Generate prompt for weak interventions.
        """
        return f"""
        Intervention type examples:
        1) Attention shift: "It seems like your attention has wandered a bit. How about taking a deep breath?"
        2) Self-awareness: "It looks like your mind is drifting elsewhere. What are you thinking about?"
        3) Acceptance/non-judgment: "It's natural for thoughts to wander during mindfulness. Accept it and let's start again."
        4) Kindness/encouragement: "You've done well so far. How about focusing a bit more?"
        
        Based on these examples, create a new message suited to the current situation.
        Please do not use the example phrases as they are; instead, craft a message tailored to the situation.
        """

    def _create_strong_intervention_prompt(self):
        """
        Generate prompt for strong interventions.
        """
        return f"""
        Intervention type examples:
        1) Posture/physical adjustment: "Your posture seems off. Straighten your back and relax your shoulders."
        2) Specific guidance: "Relax and adjust your posture."
        3) Inquiry: "What are you thinking about right now?"
        
        Based on these examples, create a new message suited to the current situation.
        Please do not use the example phrases as they are; instead, craft a message tailored to the situation.
        The message must include specific instructions or questions.
        """
    
    async def generate_intervention_message(self, ear, facial_expression, posture_change, intervention_prompt):
        """
        Generate intervention message based on the situation.
        """
        emotion = FacialExpression(facial_expression.lower())
        
        system_prompt = self._create_system_prompt()
        
        user_prompt = f"""
        Current learner status:
        - Eye status (1 indicates closed eyes): {ear}
        - Emotional state: {emotion.value}
        - Degree of posture change: {posture_change:.2f}
        
        {intervention_prompt}
        """
        
        try:
            client = openai.AsyncClient(api_key=self.api_key)
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating intervention message: {str(e)}")
            return self._get_default_message(emotion)
    
    def _get_default_message(self, emotion):
        """
        Return default message based on emotional state if API call fails.
        """
        default_messages = {
            FacialExpression.ANGRY: "You seem frustrated. How about taking a moment to breathe deeply and start again?",
            FacialExpression.DISGUST: "Feeling reluctant to study? How about taking a short break?",
            FacialExpression.FEAR: "Is there something challenging? Take your time and review it once more.",
            FacialExpression.SAD: "You seem down. How about listening to some music and clearing your mind?",
            FacialExpression.SURPRISE: "Did you encounter something surprising? Take a moment to organize your thoughts.",
            FacialExpression.HAPPY: "You're studying with joy! Keep this momentum and focus a bit more.",
            FacialExpression.NEUTRAL: "It seems your concentration has dropped a bit. How about taking a short break and starting again?"
        }
        return default_messages.get(emotion, default_messages[FacialExpression.NEUTRAL])
    
    def log_intervention(self, ear, facial_expression, posture_change, intervention_score, intervention_level, message):
        """
        Logging intervention history
        """

        def convert_to_serializable(obj):
            """
            Convert data for JSON serialization.
            """
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            return obj

        intervention_data = {
            'timestamp': datetime.now().isoformat(),
            'ear_status': convert_to_serializable(ear),
            'facial_expression': facial_expression,
            'posture_change': convert_to_serializable(posture_change),
            'intervention_score': convert_to_serializable(intervention_score),
            'intervention_level': intervention_level.value,
            'message': message
        }
        
        # Read existing data and add new data
        try:
            with open('intervention_history.json', 'r', encoding='utf-8') as f:
                self.intervention_history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Initialize as an empty list if file does not exist or has errors
            self.intervention_history = []

        # Add new data
        self.intervention_history.append(intervention_data)

        # Save updated data to JSON file
        with open('intervention_history.json', 'w', encoding='utf-8') as f:
            json.dump(self.intervention_history, f, ensure_ascii=False, indent=2, default=convert_to_serializable)
    
    async def process_state(self, ear, facial_expression, posture_change):
        """
        Process the current state and generate intervention messages if necessary.
        """
        # Calculate intervention score
        intervention_score = self.calculate_intervention_score(
            ear, facial_expression, posture_change
        )
        
         # Determine intervention level
        intervention_level, intervention_prompt = self.determine_intervention_level(intervention_score)
        
        if intervention_level != InterventionLevel.NONE and intervention_prompt:
            # Generate intervention message
            message = await self.generate_intervention_message(
                ear, facial_expression, posture_change, intervention_prompt
            )
            
            # Log intervention history
            self.log_intervention(
                ear, facial_expression, posture_change,
                intervention_score, intervention_level, message
            )
            
            return message, intervention_level
        
        return None, InterventionLevel.NONE

async def download_audio(audio_url, save_path):
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(audio_url) as response:
            if response.status == 200:
                with open(save_path, 'wb') as f:
                    f.write(await response.read())
                print(f"[INFO] Audio downloaded and saved to {save_path}")
            else:
                print(f"[ERROR] Failed to download audio. Status code: {response.status}")

async def process_intervention():
    print("[DEBUG] Starting process_intervention loop.")
    # Set OpenAI API key
    api_key = ""

    intervention_generator = MWInterventionGenerator(api_key)

    audio_dir = "C:/Users/User/Desktop/mw/audio/"
    
    while True:
        try:
            # Read CSV file
            df = pd.read_csv('C:/Users/User/Desktop/mw/data/output.csv')
            last_row = df.iloc[-1]

             # Extract data
            ear = last_row['ear']
            facial_expression = last_row['emotion'] if pd.notna(last_row['emotion']) else 'neutral'
            posture_change = float(last_row['pose'])

            message, level = await intervention_generator.process_state(
                ear, facial_expression, posture_change
            )

            if message:

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                audio_path = os.path.join(audio_dir, f"intervention_{timestamp}.mp3")

                # Generate speech using OpenAI TTS
                async with openai.audio.speech.with_streaming_response.create(
                    model="tts-1-hd",
                    voice="alloy",
                    input=message,
                ) as response:
                    with open(audio_path, "wb") as audio_file:
                        async for chunk in response.audio_chunks:
                            audio_file.write(chunk)
                print(f"[INFO] TTS saved as MP3: {audio_path}")

            else:
                print("[INFO] No intervention required.")

        except Exception as e:
            print(f"[ERROR] Failed to process intervention: {e}")

        await asyncio.sleep(5)