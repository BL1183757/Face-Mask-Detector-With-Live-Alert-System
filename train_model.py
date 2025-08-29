'''import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from imutils import paths   

dataset_path="C:/Users/Administrator/Downloads/Face Mask Detection"

categories = ["with_mask", "without_mask"]
image_size=(128, 128)

# Grab the list of images in our dataset directory, then initialize
print("[INFO] Loading images...")
image_paths=list(paths.list_images(dataset_path))

if len(image_paths) == 0:
    print("[ERROR] No images found in the preprocessed_dataset directory. Please run prepare_dataset.py first.")
    exit()

data = []
labels = []

for image_path in image_paths:
    label=image_path.split(os.path.sep)[-2]
    
    img=cv2.imread(image_path)
        
    if img is not None:
        img=cv2.resize(img, image_size)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        # Update the data and labels lists
        data.append(img)
        labels.append(label)

# Convert the data and labels to NumPy arrays            
data=np.array(data,dtype="float32")
labels=np.array(labels)


lb=LabelBinarizer()
labels=lb.fit_transform(labels)
labels=to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42,
    stratify=labels)

X_train=X_train/255.0
X_test=X_test/255.0

model=Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128,activation='relu'),
    Dense(2,activation='softmax') #with mask and without mask
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Data augmentation (for better performance 
datagen=ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Training the model
print("[INFO] Training the model...")
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=20, validation_data=(X_test, y_test))

# Save the model
model.save("face_mask_detector_model.h5")
print("[INFO] Model saved as face_mask_detector_model.h5")'''


import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from imutils import paths
from sklearn.preprocessing import LabelBinarizer

# Change the dataset path to the preprocessed directory
dataset_path = "C:/Users/Administrator/Downloads/preprocesssed_dataset"
categories = ["with_mask", "without_mask"]
image_size = (128, 128)

print("[INFO] Loading images...")
image_paths = list(paths.list_images(dataset_path))

data = []
labels = []

for image_path in image_paths:
    label = image_path.split(os.path.sep)[-2]
    
    img = cv2.imread(image_path)
    
    if img is not None:
        img = cv2.resize(img, image_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Update the data and labels lists
        data.append(img)
        labels.append(label)

# Convert the data and labels to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Partition the data into training and testing splits
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels
)

X_train = X_train / 255.0
X_test = X_test / 255.0

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

print("[INFO] Training the model...")
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=20, validation_data=(X_test, y_test))

model.save("face_mask_detector_model.h5")
print("[INFO] Model saved as face_mask_detector_model.h5")