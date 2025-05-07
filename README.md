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




# Project Name

This project utilizes Digital Signal Processing (DSP) techniques for signal processing and feature extraction. The core DSP operations are implemented across several files and sections of code, as detailed below.

## DSP Implementation Details

The DSP operations in this project are primarily implemented in the following files and sections of code:

### 1. `feature.hpp`

*   **Filepath**: `feature.hpp`
*   **Purpose**: This file contains the core DSP logic for spectral analysis and feature extraction.
*   **Key Functions**:
    *   `extract_spec_features`: Performs spectral analysis, including FFT, RMS, skewness, and kurtosis calculations.
    *   `_decimate`: Implements signal decimation using second-order section (SOS) filters.
    *   `extract_spectral_analysis_features_v4`: Combines preprocessing (e.g., filtering, decimation) and feature extraction for spectral analysis.
    *   `get_start_stop_bin`: Calculates the frequency bins for filtering.

### 2. `signal` Namespace

*   **Filepath**: Likely within the `dsp` directory.
*   **Purpose**: Provides utility functions for signal processing.
*   **Key Functions**:
    *   `signal::sosfilt`: Applies second-order section filtering.
    *   `signal::get_decimated_size`: Calculates the size of the signal after decimation.
    *   `signal::decimate_simple`: Performs simple decimation of the signal.

### 3. Wavelet Analysis

*   **Filepath**: Likely within `/workspaces/maleria-detection-ml-project/kumar07-project-1_inferencing/src/edge-impulse-sdk/dsp/wavelet/`.
*   **Purpose**: Implements wavelet-based feature extraction for non-stationary signals.
*   **Key Functions**:
    *   `wavelet::extract_wavelet_features`: Extracts features using wavelet transforms.

### 4. Filtering Logic

*   **Filepath**: Within `feature.hpp` and possibly other files in the `/dsp/` directory.
*   **Purpose**: Implements low-pass and high-pass filtering using Butterworth filters.
*   **Key Code**:
    *   Filtering logic in `extract_spectral_analysis_features_v4` (e.g., `strcmp(config->filter_type, "low")`).

### 5. CMSIS DSP Library

*   **Filepath**: `dsp`
*   **Purpose**: Provides optimized DSP functions for ARM Cortex-M processors.
*   **Key Components**:
    *   FFT, filtering, and matrix operations.

## Summary

The DSP operations are primarily implemented in `feature.hpp` and supported by utility functions in the `signal` namespace, wavelet analysis modules, and CMSIS DSP libraries. These components work together to preprocess signals, extract spectral features, and prepare data for machine learning inference.




# Project Name

This project utilizes a **TensorFlow Lite model** for machine learning inference, specifically designed for deployment on edge devices. The model is configured for **malaria detection** using image data from a camera sensor.

## Machine Learning Algorithm: TensorFlow Lite Model

The core of the machine learning inference in this project is a **TensorFlow Lite (TFLite)** model. TensorFlow Lite is a lightweight framework optimized for running machine learning models on devices with limited computational power and memory, such as microcontrollers and mobile devices.

### Key Details from `model_metadata.h`

The configuration and characteristics of the TensorFlow Lite model are defined in the `model_metadata.h` file.

1.  **Inference Engine**:
    ```cpp
    #define EI_CLASSIFIER_INFERENCING_ENGINE EI_CLASSIFIER_TFLITE
    ```
    -   Specifies that the project uses the TensorFlow Lite inference engine.

2.  **Input Data Type**:
    ```cpp
    #define EI_CLASSIFIER_TFLITE_INPUT_DATATYPE EI_CLASSIFIER_DATATYPE_INT8
    ```
    -   The model expects input data to be in **INT8 quantized format**. Quantization is a technique that reduces the precision of model weights and activations, leading to smaller model size and faster inference, particularly on hardware that supports integer operations efficiently.

3.  **Output Data Type**:
    ```cpp
    #define EI_CLASSIFIER_TFLITE_OUTPUT_DATATYPE EI_CLASSIFIER_DATATYPE_INT8
    ```
    -   The model outputs predictions in **INT8 quantized format**, consistent with the input quantization.

4.  **Input Dimensions**:
    ```cpp
    #define EI_CLASSIFIER_INPUT_WIDTH 96
    #define EI_CLASSIFIER_INPUT_HEIGHT 96
    #define EI_CLASSIFIER_INPUT_FRAMES 1
    ```
    -   The model is configured to accept input data as a **96x96 pixel image** with a single channel (frame). This indicates the model is designed for processing visual input.

5.  **Output Classes**:
    ```cpp
    #define EI_CLASSIFIER_NN_OUTPUT_COUNT 2
    #define EI_CLASSIFIER_LABEL_COUNT 2
    ```
    -   The model is trained to classify input into **two distinct classes**. In the context of malaria detection, these classes likely represent:
        *   Class 0: No malaria detected (or healthy).
        *   Class 1: Malaria detected (or infected).

6.  **Classification Threshold**:
    ```cpp
    #define EI_CLASSIFIER_THRESHOLD 0.6
    ```
    -   A **threshold of 0.6** is used for classification. This means that for a prediction to be assigned to a specific class, the model's confidence score for that class must be greater than or equal to 60%.

7.  **Quantization Enabled**:
    ```cpp
    #define EI_CLASSIFIER_QUANTIZATION_ENABLED 1
    ```
    -   Explicitly confirms that quantization is enabled for this model, contributing to its efficiency on edge devices.

8.  **Sensor Type**:
    ```cpp
    #define EI_CLASSIFIER_SENSOR EI_CLASSIFIER_SENSOR_CAMERA
    ```
    -   Indicates that the model is designed to process data originating from a **camera sensor**, reinforcing that the input is image-based.

### How the Algorithm Works

The TensorFlow Lite model likely employs a **Convolutional Neural Network (CNN)** architecture, which is highly effective for image recognition tasks. The typical workflow for inference with this model would involve:

1.  **Input Preprocessing**: The raw image data from the camera (96x96 pixels) is prepared for the model. This may involve resizing, normalization, and converting the pixel values to the expected INT8 quantized format.
2.  **Feature Extraction**: The preprocessed image is fed into the CNN layers of the model. These layers learn to identify hierarchical features within the image, such as edges, textures, and more complex patterns relevant to detecting malaria parasites in blood cell images.
3.  **Classification**: The features extracted by the CNN layers are then passed to the final layers (typically fully connected layers). These layers use the learned features to predict the probability of the image belonging to each of the two classes.
4.  **Post-Processing**: The model's output (the confidence scores for each class) is compared against the defined classification threshold (0.6). Based on this comparison, the final prediction (e.g., "Malaria Detected") is determined.

### Why TensorFlow Lite?

TensorFlow Lite was chosen for this project due to its suitability for edge deployment:

*   **Optimized for Edge Devices**: Specifically designed to run efficiently on devices with limited processing power, memory, and energy.
*   **Quantization Support**: Enables significant reduction in model size and computational cost, crucial for embedded systems.
*   **Flexibility**: Supports various model architectures and input types, making it versatile for different machine learning tasks on the edge.

### File Location

The TensorFlow Lite model and its associated metadata are located in the following files:

*   **Model File**: `tflite-model`
*   **Metadata File**: `model_metadata.h`

### Summary

The project leverages a **TensorFlow Lite model**, likely a **Convolutional Neural Network (CNN)**, for real-time malaria detection on an edge device. The model is configured to process **96x96 camera images**, utilizes **INT8 quantization** for efficiency, and classifies the input into two categories ("Malaria Detected" or "No Malaria Detected") based on a **0.6 confidence threshold**. This approach is well-suited for resource-constrained environments, enabling on-device inference without relying on cloud connectivity.