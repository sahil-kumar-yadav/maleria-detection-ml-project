# maleria-detection-ml-project
readme 
unzip marchproject.zip
# Malaria Detection ML Project

This project leverages machine learning and signal processing techniques to detect malaria using spectral analysis and other DSP (Digital Signal Processing) methods. It is built using the Edge Impulse SDK.

## Project Structure



## Key Components

### `src/`
- **`kumar07-project-1_inferencing.h`**: Main header file for the project.
- **`edge-impulse-sdk/`**: Contains the Edge Impulse SDK, including DSP, CMSIS, and TensorFlow Lite Micro components.
- **`model-parameters/`**: Stores parameters for the machine learning model.
- **`tflite-model/`**: Contains the TensorFlow Lite model files.

### `examples/`
- Contains example implementations for various platforms such as ESP32, Nano BLE33 Sense, Nicla Vision, and more.

## Key Features

### Spectral Analysis
The project includes a comprehensive spectral analysis implementation in [`feature.hpp`](kumar07-project-1_inferencing/src/edge-impulse-sdk/dsp/spectral/feature.hpp). Key features include:
- FFT-based spectral analysis.
- RMS, skewness, and kurtosis calculations.
- Support for low-pass and high-pass filtering.
- Spectral power edge calculations.

### Decimation
Efficient signal decimation is implemented to reduce the sampling rate while preserving the signal's characteristics.

### Wavelet Analysis
Wavelet-based feature extraction is supported for advanced signal processing.

## How to Use

1. **Unzip the Project**:
   ```bash
   unzip marchproject.zip


# Malaria Detection ML Project

This project uses machine learning and digital signal processing (DSP) techniques to analyze spectral features for detecting malaria. The implementation is based on the Edge Impulse SDK and includes advanced spectral analysis, filtering, and feature extraction methods.

---

## Overview

The project processes input signals (e.g., from sensors or medical devices) to extract meaningful features that can be used for malaria detection. The processing pipeline includes:

1. **Preprocessing**: Signal filtering and decimation.
2. **Feature Extraction**: Spectral analysis, including FFT, RMS, skewness, kurtosis, and wavelet-based features.
3. **Model Inference**: Using a TensorFlow Lite model to classify the extracted features.

---

## Processing Steps

### 1. Preprocessing

#### **Filtering**
- Low-pass and high-pass filters are applied to remove unwanted frequencies from the signal.
- The filters are implemented using Butterworth filters, which provide a smooth frequency response.

#### **Decimation**
- The signal is downsampled to reduce its sampling rate while preserving its key characteristics.
- Decimation is performed using a second-order section (SOS) filter to avoid aliasing.

#### **Mean Removal**
- The mean of the signal is subtracted to center the data around zero, improving the accuracy of subsequent feature extraction.

---

### 2. Feature Extraction

The core of the project lies in extracting spectral features from the input signal. This is implemented in the `feature.hpp` file.

#### **Spectral Analysis**
- **FFT (Fast Fourier Transform)**: Converts the signal from the time domain to the frequency domain.
- **RMS (Root Mean Square)**: Measures the energy of the signal.
- **Skewness**: Quantifies the asymmetry of the signal's distribution.
- **Kurtosis**: Measures the "tailedness" of the signal's distribution.

#### **Wavelet Analysis**
- Wavelet transforms are used to extract features from non-stationary signals, providing both time and frequency information.

#### **Filtering in the Frequency Domain**
- Specific frequency bins are selected or removed based on the filter configuration (e.g., low-pass or high-pass).

#### **Logarithmic Scaling**
- Logarithmic scaling is applied to the spectral features to compress the dynamic range and enhance feature representation.

---

### 3. Model Inference

- The extracted features are fed into a pre-trained TensorFlow Lite model for classification.
- The model predicts whether the input data indicates the presence of malaria.

---

## How to Use

### Step 1: Unzip the Project
Unzip the project files:
```bash
unzip 