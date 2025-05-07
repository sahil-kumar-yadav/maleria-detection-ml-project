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


Based on the provided model_metadata.h file and the context of the project, here is a detailed explanation of the model, its layers, parameters, and the files used for processing:

---

### **Model Overview**
The project uses a **TensorFlow Lite (TFLite)** model for malaria detection. The model is designed to process **96x96 images** from a camera sensor and classify them into two categories: "Malaria Detected" or "No Malaria Detected."
# **Key Model Details**
1. **Input Dimensions**:
   ```cpp
   #define EI_CLASSIFIER_INPUT_WIDTH 96
   #define EI_CLASSIFIER_INPUT_HEIGHT 96
   #define EI_CLASSIFIER_INPUT_FRAMES 1
   ```
   - The model expects input images of size **96x96 pixels** with a single frame.

2. **Output Classes**:
   ```cpp
   #define EI_CLASSIFIER_NN_OUTPUT_COUNT 2
   #define EI_CLASSIFIER_LABEL_COUNT 2
   ```
   - The model has **2 output classes**:
     - Class 0: No malaria detected.
     - Class 1: Malaria detected.

3. **Quantization**:
   ```cpp
   #define EI_CLASSIFIER_TFLITE_INPUT_DATATYPE EI_CLASSIFIER_DATATYPE_INT8
   #define EI_CLASSIFIER_TFLITE_OUTPUT_DATATYPE EI_CLASSIFIER_DATATYPE_INT8
   ```
   - The model uses **INT8 quantization** for both input and output, which reduces the model size and improves inference speed.

4. **Sensor Type**:
   ```cpp
   #define EI_CLASSIFIER_SENSOR EI_CLASSIFIER_SENSOR_CAMERA
   ```
   - The model is configured to process data from a **camera sensor**, confirming that it is image-based.

5. **Classification Threshold**:
   ```cpp
   #define EI_CLASSIFIER_THRESHOLD 0.6
   ```
   - The model uses a confidence threshold of **0.6** for classification.

---

### **Model Architecture**
While the exact architecture of the TensorFlow Lite model is not provided in the model_metadata.h file, the following assumptions can be made based on the context:

1. **Input Layer**:
   - Accepts a **96x96x1** image (grayscale or single-channel input).
   - Preprocessing may include resizing, normalization, and quantization.

2. **Convolutional Layers**:
   - Extract spatial features such as edges, textures, and patterns from the input image.
   - Likely includes ReLU activation functions and max-pooling layers for downsampling.

3. **Fully Connected Layers**:
   - Combines the extracted features to make predictions.
   - Outputs probabilities for the two classes ("Malaria Detected" or "No Malaria Detected").

4. **Output Layer**:
   - A softmax layer provides the final classification probabilities.

---

### **Parameters Being Processed**
1. **Image Data**:
   - Input images are processed as **96x96 pixels**.
   - Preprocessing includes resizing and quantization to INT8 format.

2. **Spectral Features**:
   - The model may use spectral features (e.g., FFT) as part of the preprocessing pipeline, as indicated by:
     ```cpp
     #define EI_CLASSIFIER_HAS_FFT_INFO 1
     ```

3. **Slices Per Window**:
   ```cpp
   #define EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW 4
   ```
   - The input data is divided into **4 slices** for processing.

4. **DSP Parameters**:
   - The project uses DSP (Digital Signal Processing) for preprocessing, as indicated by:
     ```cpp
     #define EI_DSP_PARAMS_GENERATED 1
     ```

---

### **Files Used for Processing**
1. **Model Metadata**:
   - **Filepath**: model_metadata.h
   - **Purpose**: Defines the model's input/output dimensions, quantization, sensor type, and other metadata.

2. **TensorFlow Lite Model**:
   - **Filepath**: tflite-model
   - **Purpose**: Contains the pre-trained TensorFlow Lite model used for inference.

3. **DSP Implementation**:
   - **Filepath**: dsp
   - **Purpose**: Implements DSP preprocessing, including FFT, filtering, and feature extraction.

4. **Feature Extraction**:
   - **Filepath**: feature.hpp
   - **Purpose**: Extracts spectral features from the input data.

5. **Inference Code**:
   - **Filepath**: Likely in src
   - **Purpose**: Runs the TensorFlow Lite model on the preprocessed data and outputs predictions.

---

### **How the Processing Works**
1. **Preprocessing**:
   - Input images are resized to **96x96 pixels**.
   - DSP techniques (e.g., FFT) are applied to extract spectral features if required.

2. **Quantization**:
   - The input data is quantized to **INT8** format for efficient processing.

3. **Inference**:
   - The preprocessed data is passed to the TensorFlow Lite model.
   - The model processes the data through its layers (e.g., convolutional and fully connected layers).

4. **Post-Processing**:
   - The model's output probabilities are compared against the classification threshold (0.6) to determine the final prediction.

---

### **Summary**
The project uses a **TensorFlow Lite model** with a likely **Convolutional Neural Network (CNN)** architecture for malaria detection. The model processes **96x96 images** from a camera sensor, with preprocessing handled by DSP techniques. Key files include model_metadata.h for metadata, the TensorFlow Lite model file for inference, and DSP-related files for preprocessing. The entire pipeline is optimized for edge devices using quantization and efficient processing techniques.



The provided code in ei_run_classifier.h and its related files outlines how the project classifies images using a **TensorFlow Lite (TFLite)** model. Below is a detailed explanation of the image classification process and the files involved.

---

### **How It Classifies Images**

1. **Image Preprocessing**:
   - The raw image data is preprocessed before being passed to the model. This includes:
     - **Scaling**: The image pixel values are scaled to match the model's input requirements.
       ```cpp
       if (block->image_scaling == EI_CLASSIFIER_IMAGE_SCALING_0_255) {
           int scale_res = numpy::scale(fmatrix, 1 / 255.0f);
       }
       ```
     - **Mean Subtraction**: For certain scaling modes, the mean values (e.g., ImageNet mean) are subtracted from the pixel values.
       ```cpp
       else if (block->image_scaling == EI_CLASSIFIER_IMAGE_SCALING_BGR_SUBTRACT_IMAGENET_MEAN) {
           for (size_t ix = 0; ix < fmatrix->rows * fmatrix->cols; ix += 3) {
               float b = fmatrix->buffer[ix + 0];
               fmatrix->buffer[ix + 0] = fmatrix->buffer[ix + 2] + tao_mean[2];
               fmatrix->buffer[ix + 1] += tao_mean[1];
               fmatrix->buffer[ix + 2] = b + tao_mean[0];
           }
       }
       ```

2. **Feature Extraction**:
   - The preprocessed image data is converted into a feature matrix (`fmatrix`) that matches the input format of the TensorFlow Lite model.
   - This feature matrix is passed to the model for inference.

3. **Model Inference**:
   - The TensorFlow Lite model is invoked to classify the image.
     ```cpp
     EI_IMPULSE_ERROR res = block.infer_fn(impulse, fmatrix, ix, (uint32_t*)block.input_block_ids, block.input_block_ids_size, result, block.config, debug);
     ```
   - The `infer_fn` function is responsible for running the model and generating predictions.

4. **Post-Processing**:
   - The raw output from the model is processed to generate human-readable classification results.
   - For example, the probabilities for each class are extracted and displayed:
     ```cpp
     for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
         ei_printf("    %s: ", result->classification[ix].label);
         ei_printf_float(result->classification[ix].value);
         ei_printf("\n");
     }
     ```

5. **Continuous Inference (Optional)**:
   - If continuous inference is enabled, the system processes slices of data and maintains a rolling window of features for classification:
     ```cpp
     EI_IMPULSE_ERROR process_impulse_continuous(ei_impulse_handle_t *handle, signal_t *signal, ei_impulse_result_t *result, bool debug = false);
     ```

---

### **Files Used for Image Classification**

1. **`ei_run_classifier.h`**:
   - **Filepath**: ei_run_classifier.h
   - **Purpose**:
     - Contains the main logic for running the classifier, including preprocessing, inference, and post-processing.
     - Implements functions like `run_classifier`, `run_classifier_image_quantized`, and `process_impulse`.

2. **`tflite_micro.h`**:
   - **Filepath**: tflite_micro.h
   - **Purpose**:
     - Provides the TensorFlow Lite Micro interpreter for running the model on edge devices.
     - Handles model loading, input/output tensor management, and inference execution.

3. **`model_metadata.h`**:
   - **Filepath**: model_metadata.h
   - **Purpose**:
     - Defines the model's input/output dimensions, scaling modes, and other metadata.
     - Specifies the number of classes and their labels.

4. **`image.hpp`**:
   - **Filepath**: image.hpp
   - **Purpose**:
     - Provides image processing utilities, such as resizing and cropping.
     - Used to prepare the raw image data for classification.

5. **`ei_run_dsp.h`**:
   - **Filepath**: ei_run_dsp.h
   - **Purpose**:
     - Handles Digital Signal Processing (DSP) tasks, such as feature extraction and scaling.
     - Works in conjunction with ei_run_classifier.h to preprocess image data.

6. **`tflite-model/`**:
   - **Filepath**: tflite-model
   - **Purpose**:
     - Contains the TensorFlow Lite model file used for inference.
     - The model is loaded and executed by the TensorFlow Lite Micro interpreter.

7. **`ei_logging.h`**:
   - **Filepath**: ei_logging.h
   - **Purpose**:
     - Provides logging utilities for debugging and displaying inference results.

---

### **Summary**
The project classifies images by preprocessing the raw image data, extracting features, and running inference using a TensorFlow Lite model. The key files involved in this process include:

- **`ei_run_classifier.h`**: Main classification logic.
- **`tflite_micro.h`**: TensorFlow Lite Micro interpreter.
- **`model_metadata.h`**: Model metadata and configuration.
- **`image.hpp`**: Image preprocessing utilities.
- **`tflite-model/`**: TensorFlow Lite model file.

These components work together to process the input image, run the model, and generate classification results. Let me know if you'd like further details on any specific part!



Great question! Let’s break down **how image processing and malaria detection happens after the camera scans an image**, **step by step**, in your embedded system using the **Edge Impulse SDK and TensorFlow Lite Micro (TFLM)**.

---

## 🧠 High-Level Workflow

```
[Camera] → [Image Preprocessing (DSP)] → [Feature Extraction] → [TFLite Model Inference] → [Malaria/Uninfected Output]
```

---

## 🔍 Step-by-Step Breakdown

### 🔹 1. **Image Capture from Camera**

* The **camera module** (e.g., on Arduino Nano BLE Sense or Nicla Vision) captures an image of the **blood smear**.
* This raw image is usually **RGB or grayscale**, depending on the sensor.
* The image is passed to the program as a buffer (e.g., `uint8_t image[96][96]` for a 96×96 input).

> ⚙️ Input size is defined by:

```cpp
#define EI_CLASSIFIER_INPUT_WIDTH 96
#define EI_CLASSIFIER_INPUT_HEIGHT 96
```

---

### 🔹 2. **Image Preprocessing**

Before sending the image to the model, the system performs **preprocessing**:

#### ✅ a. **Resizing** (if needed)

* The image is resized to the model's expected input dimensions (usually 96×96).
* This is handled by Edge Impulse's built-in image utility functions.

#### ✅ b. **Scaling / Normalization**

* The pixel values are **scaled**:

  * For **quantized models**: values are scaled to **INT8 \[0–255]** or adjusted using zero-point/scale values.
  * For example:

    ```cpp
    scale = input_tensor->params.scale;
    zero_point = input_tensor->params.zero_point;
    input_tensor->data.int8[i] = (pixel_value / 255.0) / scale + zero_point;
    ```

#### ✅ c. **Mean subtraction (optional)**

* In some models, especially if trained on ImageNet, mean subtraction is used for normalization.
* This shifts the pixel values to be centered around 0.

---

### 🔹 3. **Digital Signal Processing (DSP) Feature Extraction**

* Implemented in files like `feature.hpp` and `ei_run_dsp.h`
* The image is **converted to spectral features**, including:

  * **FFT** – converts image signal (or pixel intensity signal) to frequency domain.
  * **RMS / Skewness / Kurtosis** – used to describe the "shape" of signal distribution.
  * **Wavelet Transform** – captures patterns across different resolutions.

> These features help reduce image complexity and focus on **important diagnostic patterns** (e.g., parasite shapes).

---

### 🔹 4. **TensorFlow Lite Micro Inference**

After feature extraction:

1. Features are passed to the `.tflite` model stored in Flash (or embedded as `model.h`).
2. TFLite interpreter processes the input:

   ```cpp
   interpreter->SetInputTensor(data);
   interpreter->Invoke();
   ```
3. The model returns a prediction (e.g., `Class 0 = Uninfected`, `Class 1 = Infected`).

> Model outputs an array like `[0.3, 0.7]` → meaning 70% confidence for "Infected".

---

### 🔹 5. **Post-processing and Output**

* The model’s output is compared to a threshold (usually 0.6):

  ```cpp
  if (output[1] > 0.6) {
      Serial.println("Malaria Detected ✅");
  } else {
      Serial.println("Uninfected ❌");
  }
  ```

* Results can be:

  * Displayed on a serial monitor.
  * Shown on an OLED or LCD screen.
  * Trigger actions (e.g., turn on LED/buzzer for infected samples).

---

## 📦 Example Visual Flow

```
[CAMERA] 
    ↓ (96x96 image) 
[PREPROCESSING] 
    → Resize → Normalize/Scale → Spectral Transform (FFT/Wavelet) 
    ↓
[FEATURE VECTORS]
    ↓
[INFERENCE: TFLite Micro Model]
    ↓
[OUTPUT: "Infected" or "Uninfected"]
```

---

## 🔁 Continuous Inference (Optional)

If the system is always capturing frames (like a loop), it may:

* Keep a **rolling buffer of images**
* Infer every 1 second or after N frames
* Run in `loop()` or `while(true)` in embedded C++

---

## ✅ Summary

| Stage            | Action                                    |
| ---------------- | ----------------------------------------- |
| 📷 Image Capture | Camera captures a 96x96 image             |
| 🧹 Preprocess    | Resize, normalize, quantize               |
| ⚙️ DSP Features  | FFT, Wavelet, RMS, Skew, etc.             |
| 🧠 Inference     | TFLite Micro model classifies features    |
| 📢 Output        | Classification shown on Serial or Display |

Would you like a **code sample** for how to handle camera → preprocess → inference in Arduino or PlatformIO?
