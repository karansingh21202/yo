from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

# Initializing the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=10, activation='softmax'))  # Assuming 10 classes

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Creating data generators
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Loading training and validation data
train_set = train_datagen.flow_from_directory('/content/drive/HyDrive/Plant-Leaf-Disease-Prediction-main/Plant-Leaf-Disease-Prediction-main/Dataset/train',
                                              target_size=(128, 128),
                                              batch_size=6,
                                              class_mode='categorical')

valid_set = test_datagen.flow_from_directory('/content/drive/Drive/Plant-Leaf-Disease-Prediction-main/Plant-Leaf-Disease-Prediction-main/Dataset/val',
                                             target_size=(128, 128),
                                             batch_size=3,
                                             class_mode='categorical')

# Fitting the model
classifier.fit(train_set,
               steps_per_epoch=20,
               epochs=50,
               validation_data=valid_set)

# Save the model architecture to JSON file
classifier_json = classifier.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(classifier_json)

# Save the model weights
classifier.save_weights("my_model_weights.h5")

# Save the entire model
classifier.save("model.h5")
