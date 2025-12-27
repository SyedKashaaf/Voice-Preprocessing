# Voice Preprocessing and Feature Extraction for Speech Samples
This repository contains a Python implementation of a voice preprocessing workflow and feature extraction for machine learning (ML) applications. The project uses audio samples from the LibriSpeech test-clean dataset and demonstrates key steps in preparing speech data for ML/DL models.
The workflow includes audio loading, preprocessing, visualization, feature extraction (MFCC), and a bonus experiment with noise addition.

# Features
Automatic Dataset Download
Downloads the LibriSpeech test-clean subset (around 330 MB).
Extracts .flac audio files for processing.
Audio Preprocessing Pipeline
Resampling to 16 kHz
Normalization
Silence trimming
Optional low-pass filtering

# Visualization
Plots raw and processed waveforms side-by-side.
Visualizes MFCC feature heatmaps for both audio samples.

# Feature Extraction
Extracts Mel-Frequency Cepstral Coefficients (MFCCs).
Handles variable audio lengths by padding MFCCs for tensor compatibility.
Combines features into a single tensor ready for ML/DL models.
Noise Experiment (Bonus)
Adds Gaussian noise to audio samples.
Demonstrates how preprocessing improves noisy audio.

# Requirements
Python 3.9+
Libraries: librosa, matplotlib, scipy, torch, numpy

You can install all dependencies via pip:

pip install librosa matplotlib scipy torch numpy
