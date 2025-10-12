
# BioMW
![Image](https://github.com/user-attachments/assets/b7db6370-8390-4ba7-ae92-cf70af9f143d)

## Abstract
Real-time mind wandering (MW) detection and intervention can enhance mindfulness by helping individuals maintain focus. This study introduces BioMW, a robot-based, non-invasive model that estimates MW levels using biometric data, including facial expressions, posture, and eye-aspect ratio, and then provides appropriate interventions. We conducted a user study under three different intervention conditions: the proposed robot-assisted, audio-based, and no intervention. Results indicate that robot-based feedback reduces MW, as reflected in lower EEG-based Theta/Alpha Ratio (TAR), improving mindfulness engagement. Time-series variation of the concentration index further demonstrates that real-time interventions help refocus attention during mindfulness. While the effectiveness of interventions varies based on individual mindfulness capacity, as measured by Mindful Attention Awareness Scale (MAAS), robotic feedback generally provides a structured and interactive method to sustain attention. These findings highlight the potential of robotic systems in mindfulness training and suggest the need for personalized intervention strategies tailored to different user experience levels.

## Contribution
- **Vision-based Non-Invasive BioMW Model.** We propose a non-invasive model that calculates MW indices in real-time by integrating multi-biometric indicators—facial expressions, posture, and EAR. This approach presents a low-cost, low-burden solution for monitoring attentional lapses in mindfulness environments.
- **Robot-Based Intervention vs. Non-Robot-Based Intervention.** By quantitatively comparing robot-assisted intervention, audio-based intervention, and a no-intervention control, this study explores the additional value that robot interactions can provide in supporting attentional focus and emotional stability during mindfulness practice.
- **Linking Mindfulness and MW Metrics.** Assuming that lower MW levels correspond to higher mindfulness engagement, we conduct a multifaceted effectiveness analysis by combining MW indices from the BioMW model with pre/post-session surveys and EEG analysis results.

## Usage
### 0. Prerequisites
```bash
# Setting
python -m venv biomw_env
source biomw_env/bin/activate

# Required package
pip install -r requirements.txt
```

### 1. Download
You can download the trained model from the following link: [Google Drive - BioMW Model](https://drive.google.com/drive/folders/1Xb_f64RNsx4d33HyygtEKamSFxZTjFgB?usp=sharing)


### 2. Run
```bash
python main.py
```

## Repository Structure
```
BioMW/
│── data/               		  # Folder where real-time biometric data is stored
│── image/     					      # Folder for storing input images
│── model/     					      # Folder for storing trained models
│── preprocessing/     			  # Preprocessing code
│   ├── eeg.ipynb   			    # EEG data preprocessing code
│── source/            			  # Main source code
│   ├── ear_detection.py		  # EAR detection code
│   ├── emotion_analysis.py	  # Emotion recognition code
│   ├── get_image.py			    # Code to retrieve images from Pepper every 10 seconds
│   ├── image_input.py			  # Code for loading images
│   ├── intervention.py			  # Code for generating intervention messages
│   ├── k_emotion_analysis.py	# Emotion recognition code
│   ├── main.py					      # Main execution file
│── requirements.txt    		  # List of required libraries
│── README.md           		  # Project description file
```
