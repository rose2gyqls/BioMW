import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, welch, iirnotch
import matplotlib.pyplot as plt
import mne
import glob
import os

# Set the output path for saving results
output_path = ''  # Path to save the results
os.makedirs(output_path, exist_ok=True)

# 1. Load data and group classification
data_path = ''  # EEG data path
file_list = sorted(glob.glob(data_path + '*.csv'))  # Sort the CSV files in the path

# Define groups (G1, G2, G3)
G1_group = file_list[0:5]
G2_group = file_list[5:10]
G3_group = file_list[10:15]
groups = {'G1': G1_group, 'G2': G2_group, 'G3': G3_group}

# Sampling frequency setting
fs = 250  # Sampling frequency in Hz

# Define frequency bands
bands = {
    'Theta': (4, 8),
    'SMR': (12, 15),
    'Mid-Beta': (15, 20)
}

# Define drowsiness-related bands
drowsiness_bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12)
}

# Bandpass filter function (0.5-30 Hz)
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

# Notch filter function (removing 50 Hz power line noise)
def notch_filter(data, freq=50.0, fs=250.0, Q=30.0):
    b, a = iirnotch(freq / (fs / 2), Q)
    return lfilter(b, a, data)

# Function to calculate power within a frequency band using Welch's method
def calculate_band_power(data, fs, band):
    f, Pxx = welch(data, fs, nperseg=1024)
    band_power = np.trapz(Pxx[(f >= band[0]) & (f <= band[1])], f[(f >= band[0]) & (f <= band[1])])
    return band_power

# List to store all results
all_results = []

# Process and analyze data by group
for group_name, files in groups.items():
    for idx, file in enumerate(files):
        try:
            # Load CSV file (skip bad lines)
            data = pd.read_csv(file, on_bad_lines='skip')

            # Check if necessary channels exist
            if 'Fp1' in data.columns and 'Fp2' in data.columns:
                eeg_data = data[['Fp1', 'Fp2']]
            else:
                print(f"Required channels are missing, skipping file {file}.")
                continue

            # Apply bandpass filter and notch filter to each channel
            for channel in eeg_data.columns:
                eeg_data[channel] = bandpass_filter(eeg_data[channel], 0.5, 30, fs)  # Apply bandpass filter
                eeg_data[channel] = notch_filter(eeg_data[channel])  # Apply notch filter

            # Include EEG channels in the MNE RawArray
            info = mne.create_info(ch_names=['Fp1', 'Fp2'], sfreq=fs, ch_types=['eeg', 'eeg'])
            raw = mne.io.RawArray(eeg_data.values.T, info)

            # Perform ICA to identify and exclude artifacts
            ica = mne.preprocessing.ICA(n_components=2, random_state=97, max_iter=800)
            ica.fit(raw)

            # Extract EOG-related components automatically using ICA and frontal channels
            eog_indices, _ = ica.find_bads_eog(raw, ch_name=['Fp1', 'Fp2'])  # Use frontal channels for EOG
            ica.exclude = eog_indices  # Exclude detected components
            raw_ica = ica.apply(raw)  # Apply ICA to remove EOG artifacts

            # Convert the filtered data back to a DataFrame
            eeg_data_filtered = pd.DataFrame(raw_ica.get_data().T, columns=eeg_data.columns)

            # Calculate band power for each channel
            channel_metrics = {}
            for channel in eeg_data_filtered.columns:
                channel_data = eeg_data_filtered[channel]
                channel_metrics[channel] = {}
                # Calculate concentration index
                smr_power = calculate_band_power(channel_data, fs, bands['SMR'])
                mid_beta_power = calculate_band_power(channel_data, fs, bands['Mid-Beta'])
                theta_power = calculate_band_power(channel_data, fs, bands['Theta'])
                channel_metrics[channel]['Concentration Index'] = (smr_power + mid_beta_power) / theta_power if theta_power > 0 else np.nan

                # Calculate drowsiness metrics
                delta_power = calculate_band_power(channel_data, fs, drowsiness_bands['Delta'])
                alpha_power = calculate_band_power(channel_data, fs, drowsiness_bands['Alpha'])
                channel_metrics[channel]['DAR'] = delta_power / alpha_power if alpha_power > 0 else np.nan
                channel_metrics[channel]['TAR'] = theta_power / alpha_power if alpha_power > 0 else np.nan
                channel_metrics[channel]['Slow-Wave Activity'] = delta_power + theta_power

            # Convert to DataFrame
            metrics_df = pd.DataFrame(channel_metrics).T

            # Store results
            all_results.append({
                'file': file,
                'group': group_name,
                'Fp1_Concentration': metrics_df.loc['Fp1', 'Concentration Index'],
                'Fp2_Concentration': metrics_df.loc['Fp2', 'Concentration Index'],
                'Fp1_DAR': metrics_df.loc['Fp1', 'DAR'],
                'Fp2_DAR': metrics_df.loc['Fp2', 'DAR'],
                'Fp1_TAR': metrics_df.loc['Fp1', 'TAR'],
                'Fp2_TAR': metrics_df.loc['Fp2', 'TAR'],
                'Fp1_Slow-Wave': metrics_df.loc['Fp1', 'Slow-Wave Activity'],
                'Fp2_Slow-Wave': metrics_df.loc['Fp2', 'Slow-Wave Activity'],
                'Average_Concentration': metrics_df['Concentration Index'].mean(),
                'Average_DAR': metrics_df['DAR'].mean(),
                'Average_TAR': metrics_df['TAR'].mean(),
                'Average_Slow-Wave': metrics_df['Slow-Wave Activity'].mean()
            })
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue

# Convert all results to a DataFrame
all_results_df = pd.DataFrame(all_results)

# Visualize and save results for each group
metrics = ['Average_Concentration', 'Average_DAR', 'Average_TAR', 'Average_Slow-Wave']
for group_name, files in groups.items():
    group_data = all_results_df[all_results_df['group'] == group_name]

    # Visualize and save each metric
    for metric in metrics:
        plt.figure(figsize=(8, 6))
        plt.bar(group_data['file'], group_data[metric], color='blue')
        plt.title(f"{metric.replace('_', ' ')} in Group {group_name}")
        plt.xlabel("Participant")
        plt.ylabel(metric.replace('_', ' '))
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.savefig(f"{output_path}/{group_name}_{metric}.png")
        plt.close()

# Visualize and save group-wise average comparisons
group_means = all_results_df.groupby('group')[metrics].mean().reset_index()
for metric in metrics:
    plt.figure(figsize=(8, 6))
    plt.bar(group_means['group'], group_means[metric], color='green')
    plt.title(f"Group Comparison of {metric.replace('_', ' ')}")
    plt.xlabel("Group")
    plt.ylabel(metric.replace('_', ' '))
    plt.grid(True)
    plt.savefig(f"{output_path}/Group_Comparison_{metric}.png")
    plt.close()