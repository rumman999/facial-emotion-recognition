import numpy as np
import tensorflow as tf
import cv2
import os
import pandas as pd
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


trainPath = "../data/train"
testPath = "../data/test"

# Get the list of emotion folders and sort them to keep labels consistent
folderList = os.listdir(trainPath)
folderList.sort()
print("Categories found:", folderList)

X_train = []
y_train = []


# load the train data into array

for i, category in enumerate(folderList):
    category_path = os.path.join(trainPath, category)
    files = os.listdir(category_path)

    for file in files:
        img_path = os.path.join(category_path, file)

        img = cv2.imread(img_path, 0)

        X_train.append(img)
        y_train.append(i)

print(f"Total training images loaded: {len(X_train)}")


folderList = os.listdir(testPath)
folderList.sort()
print("Categories found:", folderList)

X_test = []
y_test = []


# load the test data into array

for i, category in enumerate(folderList):
    category_path = os.path.join(testPath, category)
    files = os.listdir(category_path)

    for file in files:
        img_path = os.path.join(category_path, file)

        img = cv2.imread(img_path, 0)

        X_test.append(img)
        y_test.append(i)

print(f"Total test images loaded: {len(X_test)}")

X_train = np.array(X_train, 'float32') / 255.0
X_test = np.array(X_test, 'float32') / 255.0

y_train = np.array(y_train, 'int32')
y_test = np.array(y_test, 'int32')

numOfImages = X_train.shape[0]
X_train = X_train.reshape(numOfImages, 48, 48, 1)

numOfImages = X_test.shape[0]
X_test = X_test.reshape(numOfImages, 48, 48, 1)


y_train = to_categorical(y_train, num_classes=5)
y_test = to_categorical(y_test, num_classes=5)


# Train model

input_shape = X_train.shape[1:]


model = Sequential()
model.add(Conv2D(
    input_shape=input_shape,
    filters=64,
    kernel_size=(3, 3),
    padding="same",
    activation="relu"
))

model.add(Conv2D(
    filters=64,
    kernel_size=(3, 3),
    padding="same",
    activation="relu"
))

model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2)
))


model.add(Conv2D(
    filters=128,
    kernel_size=(3, 3),
    padding="same",
    activation="relu"
))

model.add(Conv2D(
    filters=128,
    kernel_size=(3, 3),
    padding="same",
    activation="relu"
))

model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2)
))


model.add(Conv2D(
    filters=256,
    kernel_size=(3, 3),
    padding="same",
    activation="relu"
))

model.add(Conv2D(
    filters=256,
    kernel_size=(3, 3),
    padding="same",
    activation="relu"
))

model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2)
))


model.add(Conv2D(
    filters=512,
    kernel_size=(3, 3),
    padding="same",
    activation="relu"
))

model.add(Conv2D(
    filters=512,
    kernel_size=(3, 3),
    padding="same",
    activation="relu"
))

model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2)
))


model.add(Conv2D(
    filters=512,
    kernel_size=(3, 3),
    padding="same",
    activation="relu"
))

model.add(Conv2D(
    filters=512,
    kernel_size=(3, 3),
    padding="same",
    activation="relu"
))

model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2)
))

model.add(Flatten())
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(4096, activation="relu"))
model.add(Dense(5, activation="softmax"))

print(model.summary())


model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

batch = 64
epochs = 30


stopEarly = EarlyStopping(monitor='val_accuracy', patience=5)



# train the model

history = model.fit(
    X_train,
    y_train,
    batch_size=batch,
    epochs=epochs,
    verbose=1,
    validation_data=(X_test, y_test),
    shuffle=True,
    callbacks=[stopEarly]
)

# Show Result of Training

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


# show the charts

epochs_range = range(len(acc))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot Accuracy
ax1.plot(epochs_range, acc, 'r-', label="Train")
ax1.plot(epochs_range, val_acc, 'b-', label="Val")
ax1.set_title("Model Accuracy")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True)

# Plot Loss
ax2.plot(epochs_range, loss, 'r-', label="Train")
ax2.plot(epochs_range, val_loss, 'b-', label="Val")
ax2.set_title("Model Loss")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("../reports/performance_metrics.png", dpi=300)
plt.show()

# save the model

modelFileName = "../model/emotion.keras"
os.makedirs("../model", exist_ok=True)
model.save(modelFileName)