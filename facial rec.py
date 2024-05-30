import os
import cv2
import numpy as np
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Replace with the path to your downloaded dataset
dataset_path = '50k-celebrity-faces-image-dataset'

# Get a list of all the image file names
image_files = os.listdir(dataset_path)

# Load the images
images = []
labels = []
reference_images = {}  # To store a reference image for each label

# Define a fixed size for the faces
face_size = (200, 200)

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

for file in image_files:
    # Construct the full image path
    image_path = os.path.join(dataset_path, file)
    
    # Load the image in grayscale (for simplicity)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Use the classifier to detect faces in the image
    faces = classifier.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

    # If a face is detected
    if len(faces) > 0:
        # For simplicity, we're only considering the first face detected
        (x, y, w, h) = faces[0]
        
        # Crop the face from the image
        face = image[y:y+h, x:x+w]
        
        # Resize the face to the fixed size
        face = cv2.resize(face, face_size)
        
        # Flatten the face image and add it to the list
        images.append(face.flatten())
        
        # Extract the label from the file name and add it to the list
        label = os.path.splitext(file)[0]  # This assumes that the file name is the label
        labels.append(label)

        # Store a reference image for this label if not already stored
        if label not in reference_images:
            reference_images[label] = face
    else:
        print(f"No face detected in {file}")

# Convert the lists to numpy arrays
face_data = np.array(images)
labels = np.array(labels)

# Scale the features
scaler = StandardScaler()
face_data_scaled = scaler.fit_transform(face_data)

# Apply PCA
pca = PCA(n_components=10)
face_data_pca = pca.fit_transform(face_data_scaled)

# Create and train the k-NN model
knn = neighbors.KNeighborsClassifier(n_neighbors=10)
knn.fit(face_data_pca, labels)

# Load the image you want to predict
test_image_path = 'test_image.jpg'  # Replace with the path to your test image
test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

if test_image is not None:
    # Resize the test image to the fixed size
    test_image = cv2.resize(test_image, face_size)
    
    # Scale and apply PCA to the test image
    test_image_scaled = scaler.transform(test_image.flatten().reshape(1, -1))
    test_image_pca = pca.transform(test_image_scaled)
    
    # Predict the label of the test image
    predicted_label = knn.predict(test_image_pca)
    print(f"The predicted label of the test image is: {predicted_label[0]}")
    
    # Display the test image and its predicted label
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(test_image, cmap='gray')
    plt.title("Test Image")

    plt.subplot(1, 2, 2)
    plt.imshow(reference_images[predicted_label[0]], cmap='gray')
    plt.title(f"Reference Image for Predicted Image: {predicted_label[0]}")

    plt.show()
else:
    print(f"Failed to load image at {test_image_path}")
