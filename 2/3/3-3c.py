import cv2
import numpy as np
from skimage.feature import hog
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Load CIFAR-10 dataset (replace this with your actual dataset loading code)
# Example assumes you have a function load_cifar10() to load the dataset
def load_cifar10():
    # Specify the path to the CIFAR-10 dataset files
    cifar10_path = 'C:/Users/Bin/Desktop/Master OBV/Master-WS/BALG/cifar-10-batches-py'  # Replace with your actual path

    # Load the training batch (adjust as needed based on your dataset structure)
    with open(f'{cifar10_path}/data_batch_1', 'rb') as f:
        batch = pickle.load(f, encoding='bytes')

    # Extract images and labels
    images = batch[b'data'].reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = np.array(batch[b'labels'])

    return {'images': images, 'labels': labels}

# Load CIFAR-10 dataset

cifar10_data = load_cifar10()

# Preprocess data (replace this with your actual preprocessing code)
# Example assumes you have a function preprocess_data() to preprocess the data
def preprocess_data(data):
    # Normalize pixel values to the range [0, 1]
    normalized_images = data['images'] / 255.0

    # Add any additional preprocessing steps based on your requirements

    # Return the preprocessed data
    return {'images': normalized_images, 'labels': data['labels']}

# Preprocess CIFAR-10 data
preprocessed_data = preprocess_data(cifar10_data)

# Extract features using HOG and color histogram
def compute_hog_features(image):
    # Convert the image to 8 bits per channel (if it's not already)
    if image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Compute HOG features
    features, _ = hog(gray_image, visualize=True)

    return features
def compute_color_histogram_features(image):
    # Convert the image to 8 bits per channel (if it's not already)
    if image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)

    # Convert the image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Compute color histogram features
    hist_hue = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])

    # Normalize the histogram
    hist_hue /= np.sum(hist_hue)

    return hist_hue.flatten()

def compute_combined_features(image):
    hog_features = compute_hog_features(image)
    color_histogram_features = compute_color_histogram_features(image)
    return np.concatenate((hog_features, color_histogram_features))

# Compute features for each image in the dataset
feature_vectors = []
for image in preprocessed_data['images']:
    features = compute_combined_features(image)
    feature_vectors.append(features)

# Convert feature vectors to NumPy array
X_train = np.array(feature_vectors)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, preprocessed_data['labels'], test_size=0.2, random_state=42)

# Standardize features using StandardScaler
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

softmax_classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
softmax_classifier.fit(X_train_std, y_train)

# Make predictions on the test set
y_pred = softmax_classifier.predict(X_test_std)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')