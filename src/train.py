from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
from tqdm import tqdm

# Initialize the initial learning rate, epochs and batch size
LEARNING_RATE = 1e-3
EPOCHS = 100
BATCH_SIZE = 8

# Defining paths
SRC_DIR = os.path.join(os.path.abspath('.'), 'src')
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'dataset')

# Grab the list of images in out dataset folder
print('[INFO] loading images...')
image_paths = list(paths.list_images(DATA_DIR))
data = []
labels = []

# Loop over the image paths
for image_path in tqdm(image_paths):
    # Extract the class label from filename
    label = image_path.split(os.path.sep)[-2]

    # Load image, swap color channels and resize to 224x224 pixels
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224))

    # Update the data and labels lists
    data.append(image)
    labels.append(label)

# Convert data and labels to numpy array and scale pixels range to [0-1]
data = np.array(data) / 255.0
labels = np.array(labels)

# Perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Split data into training and validation splits using 70-30
(X_train, X_val, y_train, y_val) = train_test_split(data, labels, test_size=.3, stratify=labels, random_state=42)

# Initialize data augmentation
trainAug = ImageDataGenerator(rotation_range=15, fill_mode='nearest')

# Loading the VGG16 network, ensuring the head Fully Connected layer sets are left off
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Construct the head of the model that will be placed on top of the base model
head_model = base_model.output
head_model = AveragePooling2D(pool_size=(4,4))(head_model)
head_model = Flatten(name='flatten')(head_model)
head_model = Dense(64, activation='relu')(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation='softmax')(head_model)

# Place the head Fully Connected model on top of the base model (actual model)
model = Model(base_model.input, head_model)

# Loop over all layers in the base model and freeze them so they won't be updated during the first training process
for layer in base_model.layers:
    layer.trainable = False

# Compiling or model
print('[INFO] compiling model...')
opt = Adam(lr=LEARNING_RATE, decay=LEARNING_RATE / EPOCHS)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Display the summary of network
print('Summary of Convolutional Neural Network:\n', model.summary())
print()

# Train the head of network
print('[INFO] training head...')
H = model.fit(
    trainAug.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    validation_data=(X_val, y_val),
    validation_steps=len(X_val) // BATCH_SIZE,
    epochs=EPOCHS
)

# Making predictions on the validation set
print('[INFO] evaluating network...')
y_pred_val = model.predict(X_val, batch_size=BATCH_SIZE)

# For each image in validation set we need to find the index of the labels with highest predicted prob
y_pred_val = np.argmax(y_pred_val, axis=1)

# Computing confusion matrix and use it to derive the raw accuracy, sensitivity and specificity
cm = confusion_matrix(y_val.argmax(axis=1), y_pred_val)

total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

print(cm)
print('Accuracy: {:.4f}'.format(acc))
print('Sensitivity: {:.4f}'.format(sensitivity))
print('Specificity: {:.4f}'.format(specificity))