# Peritys-Innovations

#Here's a detailed README file for your project that includes all essential instructions for setting up and running of model:
# Non-Hindi Speech Synthesis Model with VITS

## Project Overview
This project focuses on training a speech synthesis model using the VITS architecture. The objective is to build a model that synthesizes high-quality speech for a non-Hindi language, using datasets from the AI4Bharat corpus. The model is configured to convert WAV audio to FLAC audio and supports streaming via WebSockets.

## Table of Contents
- [Project Setup](#project-setup)
- [Prerequisites](#prerequisites)
- [Dataset Preparation](#dataset-preparation)
- [Model Training](#model-training)
- [Evaluation and Optimization](#evaluation-and-optimization)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## Project Setup

#1. Prerequisites
Ensure you have the following installed:
- Python 3.7+
- [Google Colab]([https://colab.research.google.com/](https://colab.research.google.com/drive/1gi4Nf6M1JH0XF1jVDnNVj-9NdYfYUV1y?usp=sharing)) or local Jupyter Notebook
- `ffmpeg` for audio processing
- Git for repository cloning

Install required Python packages:
```bash
pip install requests pandas tqdm soundfile librosa torch matplotlib TTS pyngrok


#2. Clone VITS Repository
Clone the VITS repository and install dependencies:

git clone https://github.com/jaywalnut310/vits.git
cd vits
pip install -r requirements.txt

#3. Mount Google Drive (optional)
from google.colab import drive
drive.mount('/content/drive')



##Dataset Preparation
Download the dataset from AI4Bharat:
code:
import requests
from tqdm import tqdm
import os

def download_dataset(url: str, save_path: str, chunk_size: int = 8192) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(save_path, 'wb') as file, tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {os.path.basename(save_path)}") as progress_bar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            progress_bar.update(size)

# Example usage
dataset_url = "https://indicvoices.ai4bharat.org/.../v2_Marathi_train.tgz"
save_dir = "/content/drive/MyDrive/Dataset"
download_dataset(dataset_url, os.path.join(save_dir, "v2_Marathi_train.tgz"))



#2. Extract Dataset
Extract the downloaded .tgz file:

python
Copy code
import tarfile

def extract_tgz(tgz_path: str, extract_path: str) -> None:
    with tarfile.open(tgz_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)

extract_path = "/content/drive/MyDrive/Dataset/extracted"
extract_tgz(os.path.join(save_dir, "v2_Marathi_train.tgz"), extract_path)


#Model Training
1. Prepare Metadata
Generate and clean metadata:

import pandas as pd
import os
import json
from tqdm import tqdm

output_dir = '/content/dataset/out'
metadata_file = os.path.join(output_dir, 'metadata.csv')
os.makedirs(output_dir, exist_ok=True)
metadata = []

for json_file in tqdm(os.listdir(json_dir)):
    if json_file.endswith('.json'):
        with open(os.path.join(json_dir, json_file), 'r', encoding='utf-8') as f:
            data = json.load(f)
            for i, segment in enumerate(data['verbatim']):
                metadata.append([f"{os.path.splitext(json_file)[0]}_{i}.wav", segment['text'], data['speaker_id']])

pd.DataFrame(metadata, columns=['id', 'text', 'speaker']).to_csv(metadata_file, sep='|', index=False, header=False)


#2. Normalize Audio
Normalize the audio files to ensure consistent quality:


import librosa
import soundfile as sf

normalized_wavs_dir = os.path.join(output_dir, 'wavs_normalized')
os.makedirs(normalized_wavs_dir, exist_ok=True)

for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
    y, sr = librosa.load(os.path.join(output_dir, 'wavs', row['id']), sr=22050)
    y /= max(abs(y))
    sf.write(os.path.join(normalized_wavs_dir, row['id']), y, sr)


Evaluation and Optimization
Split the metadata into training, validation, and test sets:

from sklearn.model_selection import train_test_split

train_meta, temp_meta = train_test_split(metadata_df, test_size=0.2, random_state=42)
val_meta, test_meta = train_test_split(temp_meta, test_size=0.5, random_state=42)

train_meta.to_csv(os.path.join(output_dir, 'train_metadata.csv'), sep='|', index=False, header=False)
val_meta.to_csv(os.path.join(output_dir, 'val_metadata.csv'), sep='|', index=False, header=False)
test_meta.to_csv(os.path.join(output_dir, 'test_metadata.csv'), sep='|', index=False, header=False)






