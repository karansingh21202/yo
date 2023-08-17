from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the trained model
leaf_disease_model = load_model('path_to_your_trained_mode')  # Update with the actual path

# List of disease labels
label_name = ['Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
              'Tomato Spider mites', 'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy',
              'Other']

@app.route("/", methods=['POST'])
def predict_disease():
    data = request.json
    img = np.array(data['img'])

    # Perform prediction
    pridict_image = leaf_disease_model.predict(img.reshape((1,) + img.shape))

    return jsonify({
        "Label Name": label_name[np.argmax(pridict_image)],
        "Accuracy": pridict_image[0][np.argmax(pridict_image)] * 100
    })

if __name__ == "__main__":
    app.run(debug=True)
