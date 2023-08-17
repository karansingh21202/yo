import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Load the pre-trained model
model_filepath = '/content/drive/MyDrive/Plant-Leaf-Disease-Prediction-main/Plant-Leaf-Disease-Prediction-main/model.h5'
model = load_model(model_filepath)

print(model)
print("Model Loaded Successfully")

# Load and preprocess the test image
test_image_path = '/content/drive/MyDrive/Plant-Leaf-Disease-Prediction-main/Plant-Leaf-Disease-Prediction-main/Dataset/test/Tomato Bacterial spot (1).JPG'
test_image = cv2.imread(test_image_path)
test_image = cv2.resize(test_image, (128, 128))
test_image = img_to_array(test_image) / 255.0
test_image = np.expand_dims(test_image, axis=0)

# Predict the disease
result = model.predict(test_image)
pred = np.argmax(result, axis=1)[0]

# Display the prediction
if pred == 0:
    print("Tomato - Bacterial Spot Disease")
elif pred == 1:
    print("Tomato Early Blight Disease")
elif pred == 2:
    print("Tomato - Healthy and Fresh")
elif pred == 3:
    print("Tomato Late Blight Disease")
else:
    print("Unknown Disease")

