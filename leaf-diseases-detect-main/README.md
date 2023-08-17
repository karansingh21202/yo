**Leaf Disease Detection using Deep Learning**

This repository contains a Python script for detecting and classifying plant leaf diseases using deep learning. It leverages a pre-trained convolutional neural network (CNN) model to accurately identify various types of leaf diseases in plant images.

## How It Works

1. **Pre-trained Model**: A pre-trained CNN model is used for disease classification, trained on a diverse dataset of plant images with various disease types.

2. **Usage**: Run the `leaf_disease_detection.py` script and provide the path to a test plant leaf image.

3. **Output**: The script predicts the disease class and its probability, aiding in early disease detection and plant health assessment.

## Dependencies

- TensorFlow
- OpenCV
- NumPy

## Getting Started

1. Clone this repository.

2. Install dependencies: `pip install tensorflow opencv-python numpy`

3. Run `leaf_disease_detection.py` and provide a test leaf image path.

## Future Enhancements

- Integration with mobile apps for on-field disease detection.
- Support for multiple plant species and diseases.
- User-friendly web interface for image upload and prediction.
- Continued model improvement through enhanced training and optimization.