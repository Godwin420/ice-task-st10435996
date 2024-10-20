import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Replace with the path to your downloaded dataset
dataset_path = r"C:\Users\arend\OneDrive\Programming\ice\50k-celebrity-faces-image-dataset"

# Get a list of all the image file names
image_files = os.listdir(dataset_path)

# Load the images and store a reference image for each label
images = []
labels = []
reference_images = {}  # To store a reference image for each label
face_size = (200, 200)

classifier = cv2.CascadeClassifier(r"C:\Users\arend\OneDrive\Programming\ice\haarcascade_frontalface_default.xml")

for file in image_files:
    # Construct the full image path
    image_path = os.path.join(dataset_path, file)
    
    # Load the image in grayscale (for simplicity)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Use the classifier to detect faces in the image
    faces = classifier.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

    # If a face is detected
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        
        # Crop and resize the face to a fixed size
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, face_size)
        
        # Append the face and label
        images.append(face)
        
        # Extract label from the filename
        label = os.path.splitext(file)[0]
        labels.append(label)

        # Store a reference image for this label
        if label not in reference_images:
            reference_images[label] = face
    else:
        print(f"No face detected in {file}")

# Convert the images and labels into numpy arrays
images = np.array(images).reshape(-1, face_size[0], face_size[1], 1)
labels = np.array(labels)

# Encode the labels into numeric format
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# Normalize the image data
images = images / 255.0

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels_categorical, test_size=0.2, random_state=42)

# Build the CNN model
model = models.Sequential()

# Convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(face_size[0], face_size[1], 1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Fully connected layers
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# Load the image you want to predict
test_image_path = r"C:\Users\arend\OneDrive\Programming\ice\test_image.jpg" # Replace with the path to your test image
test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

if test_image is not None:
    # Resize the test image to the fixed size
    test_image = cv2.resize(test_image, face_size)
    
    # Preprocess the test image
    test_image_processed = test_image.reshape(1, face_size[0], face_size[1], 1) / 255.0
    
    # Predict the label of the test image
    predictions = model.predict(test_image_processed)
    predicted_label_idx = np.argmax(predictions)
    predicted_label = label_encoder.inverse_transform([predicted_label_idx])
    
    print(f"The predicted label of the test image is: {predicted_label[0]}")
    
    # Display the test image and the predicted reference image
    plt.figure(figsize=(10, 5))

    # Display the test image
    plt.subplot(1, 2, 1)
    plt.imshow(test_image, cmap='gray')
    plt.title("Test Image")

    # Display the reference image for the predicted label
    plt.subplot(1, 2, 2)
    plt.imshow(reference_images[predicted_label[0]], cmap='gray')
    plt.title(f"Reference Image for Predicted Label: {predicted_label[0]}")

    plt.show()
else:
    print(f"Failed to load image at {test_image_path}")
