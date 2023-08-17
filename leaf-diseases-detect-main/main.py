from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model
leaf_disease_model = load_model('path_to_your_trained_model')  # Update with the actual path

# List of tomato disease labels
tomato_disease_labels = [
    'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight',
    'Tomato Leaf Mold', 'Tomato Septoria leaf spot', 'Tomato Spider mites',
    'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus',
    'Tomato healthy'
]

# Function to predict tomato disease
def predict_tomato_disease(image_path, model):
    img = img_to_array(load_img(image_path, target_size=(150, 150, 3)))
    prediction = model.predict(img.reshape((1,) + img.shape))
    predicted_label = tomato_disease_labels[np.argmax(prediction)]
    confidence_percentage = prediction[0][np.argmax(prediction)] * 100

    return predicted_label, confidence_percentage

# Input image path
image_path = input('Enter the path of the tomato leaf image: ')

# Perform prediction
predicted_disease, confidence = predict_tomato_disease(image_path, leaf_disease_model)

# Display the prediction
print(f'Predicted Tomato Disease: {predicted_disease}')
print(f'Confidence: {confidence:.2f}%')
