# -*- coding: utf-8 -*-

"""
############################################################
LIBRARIES
############################################################
"""
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report
from skimage.transform import resize


"""
############################################################
NEURAL NETWORKS
############################################################
"""
# 1. Load dataset
dataset = keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = dataset.load_data()

# 1.1. Display data
print(f"X_train ({X_train.shape}) ({X_train[0].shape}):\n\n{
      X_train[0]}\n\ny_train ({y_train.shape}):\n\n{y_train[:10]}\n\n")
print(f"X_test ({X_test.shape}) ({X_test[0].shape}):\n\n{X_test[0]}\n\ny_test ({y_test.shape}):\n\n{y_test[:10]}\n\n")

# 2. Create class names that match y_train / y_test output
class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# 2.1. Display data
print(f"Class for y_train[0]: {y_train[0]} is {class_names[y_train[0]]}.\n\n")
plt.imshow(X_train[0], cmap="gray")

# 3. Normalize data to values between 0 and 1
X_train = X_train / 255  # 0 - White, 128 - Gray, 255 - Black

# 4. Split data
X_valid = X_train[:5000]
y_valid = y_train[:5000]

X_train = X_train[5000:]
y_train = y_train[5000:]

# 4.1. Display data
for i in range(0, 5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_valid[i], cmap="gray")
    plt.title(class_names[y_valid[i]])
plt.show()

# 5. Define a model
# Activation function decides if neuron will be activated or not
# The goal is to introduce nonlinearity because otherwise neuron network would be a regression model.
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

# 5.1. Display data
print(f"\n\nModel summary:\n{model.summary()}\n\n")
keras.utils.plot_model(model, show_shapes=True)

hiddenLayer1 = model.layers[1]
hL1_weights, hL1_bias = hiddenLayer1.get_weights()
print(f"\n\nHidden layer 1 weights and bias:\n\n{hL1_weights.shape}\n\nBias:\n\n{hL1_bias}\n\n")

# 6. Compile model
# Define a loss function (e.g. MSE or cross-entropy), an optimization method (e.g. Adam or SGD), parameters to track (e.g. accuracy)
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# 7. Train model
history = model.fit(X_train, y_train, batch_size=32, epochs=20,
                    validation_data=(X_valid, y_valid))  # validation_split=0.2

# 7.1. Visualize history
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True), plt.ylim(0, 1), plt.show()
# model.save("my_model_name")

# 8. Test model
loss, accuracy = model.evaluate(X_test, y_test)
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=-1)
class_map = np.array(class_names)[y_pred]
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

# 8.1. Display data
print(f"\n\nTest loss: {loss}, accuracy {accuracy}.\n\n")
print(f"Predictions:\n\n{predictions[0]}\n\n")
print(f"Retrieve the index of the highest probability class :\n\n{y_pred}\n\n")
print(f"Class map:\n\n{class_map}\n\n")
print(f"Classification report:\n\n{classification_report(y_test, y_pred, target_names=class_names)}\n\n")

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 10))
cm_display.plot(ax=ax)
plt.show()


"""
############################################################
CONVOLUTIONAL NEURAL NETWORKS
############################################################
"""
# 1. Load dataset
dataset = keras.datasets.cifar10
(X_train, y_train), (X_test, y_test) = dataset.load_data()

# 1.1. Display data
print(f"X_train ({X_train.shape}) ({X_train[0].shape}):\n\n{
      X_train[0]}\n\ny_train ({y_train.shape}):\n\n{y_train[:10]}\n\n")
print(f"X_test ({X_test.shape}) ({X_test[0].shape}):\n\n{X_test[0]}\n\ny_test ({y_test.shape}):\n\n{y_test[:10]}\n\n")

# 2. Create class names that match y_train / y_test output
class_names = ['avion', 'automobil', 'ptica', 'mačka', 'jelen', 'pas', 'zaba', 'konj', 'brod', 'kamion']
y_train_ohe = keras.utils.to_categorical(y_train, 10)
y_test_ohe = keras.utils.to_categorical(y_test, 10)

# 2.1. Display data
print(f"Class for y_train_ohe[0]: {y_train_ohe[0]} is {y_train[0]}.\n\n")
plt.imshow(X_train[0], cmap="gray")

# 3. Normalize data to values between 0 and 1
X_train = X_train / 255  # 0 - White, 128 - Gray, 255 - Black
X_test = X_test / 255

# 5. Define a model
# "valid" doesn't have padding and "same" adds as much 0s to make ouput image the same size of input since the image shrinks every time a convolution operation is performed.
# The goal is to reduce reduce map of featured, max pooling changes groupe data with maximum values.
model = keras.models.Sequential()
# Conv Layer (Filter size 3x3, Depth 32)
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
# Conv Layer (Filter size 3x3, Depth 32)
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
# Max Pool Layer (Filter size 2x2)
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
# Dropout Layer (Prob of dropout 0.25)
model.add(keras.layers.Dropout(0.25))
# Conv Layer (Filter size 3x3, Depth 64)
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
# Conv Layer (Filter size 3x3, Depth 64)
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
# Max Pool Layer (Filter size 2x2)
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
# Dropout Layer (Prob of dropout 0.25)
model.add(keras.layers.Dropout(0.25))
# FC Layer (512 neurons)
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation='relu'))
# Dropout Layer (Prob of dropout 0.5)
model.add(keras.layers.Dropout(0.5))
# FC Layer, Softmax (10 neurons)
model.add(keras.layers.Dense(10, activation='softmax'))

# 5.1. Display data
print(f"\n\nModel summary:\n{model.summary()}\n\n")
keras.utils.plot_model(model, show_shapes=True)

# 6. Compile model
# Define a loss function (e.g. MSE or cross-entropy), an optimization method (e.g. Adam or SGD), parameters to track (e.g. accuracy).
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 7. Train model
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(X_train, y_train_ohe, batch_size=32, epochs=20, validation_split=0.2, callbacks=[early_stopping])

# 7.1. Visualize history and save model
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True), plt.ylim(0, 1), plt.show()
# model.save("my_model_name")

# 8. Test model
loss, accuracy = model.evaluate(X_test, y_test_ohe)
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=-1)
class_map = np.array(class_names)[y_pred]
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

# 8.1. Display data
print(f"\n\nTest loss: {loss}, accuracy {accuracy}.\n\n")
print(f"Predictions:\n\n{predictions[0]}\n\n")
print(f"Retrieve the index of the highest probability class :\n\n{y_pred}\n\n")
print(f"Class map:\n\n{class_map}\n\n")
print(f"Classification report:\n\n{classification_report(y_test, y_pred, target_names=class_names)}\n\n")

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 10))
cm_display.plot(ax=ax)
plt.show()

# 9.1. Test on custom images
# !curl -o image.jpg https://www.website.com/wp-content/uploads/some_image.jpg
# image = plt.imread("/content/image.jpg")
# image_resized = resize(image, (32, 32))
# plt.imshow(image_resized)
# probabilities = model.predict(np.array([image_resized,]))
# index = np.argsort(probabilities([0,:]))
# for i in range(9, 5, -1):
#   print(class_map[index[i]], ":", probabilities[0,index[i]])


"""
############################################################
AUTOENCODERS
############################################################
"""
# An encoder compresses the input data (e.g., images, videos) into a latent vector, while a decoder decompresses this latent vector back into the original data.
# Bottleneck - always has fewer neurons in the middle layer than in the input and output layers.
# Typically symmetrical, meaning the size of the input is equal to the size of the output.

# 1. Load dataset
dataset = keras.datasets.mnist
(X_train, _y_train), (X_test, _y_test) = dataset.load_data()

# 2. Define encoder and decoder and combine them for training
encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(30, activation="relu")  # bottleneck
])

decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation="relu", input_shape=[30]),
    keras.layers.Dense(28*28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])

autoencoder = keras.models.Sequential([encoder, decoder])

# 3. Compile a model
autoencoder.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

# 4. Train model
history = autoencoder.fit(X_train, X_train, epochs=10, validation_data=[X_test, X_test])

# 4.1. Visualize history and save model
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True), plt.ylim(0, 1), plt.show()

# 5. Test model (Display results)
plt.figure(figsize=(24, 6))

for i in range(8):
    plt.subplot(2, 8, i + 1)
    plt.imshow(X_test[i], cmap="binary")

    plt.subplot(2, 8, 8 + 1 + i)
    pred = autoencoder.predict(X_test[i].reshape(1, 28, 28))
    plt.imshow(pred.reshape(28, 28), cmap="binary")

plt.show()


# AUTOENCODER WITH NOISE CANCELLATION
# 1. Display images
# 1.1. Image without noise
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(X_test[0], cmap="binary")

# 1.2. Image with noise
plt.subplot(1, 2, 2)
noise = np.random.random((28, 28)) / 4
plt.imshow(X_test[0] + noise, cmap="binary")

# 2. Define encoder and decoder and combine them for training
encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(30, activation="relu")  # Bottleneck
])

decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation="relu", input_shape=[30]),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(28*28, activation="sigmoid"),  # Bottleneck
    keras.layers.Reshape([28, 28])
])

autoencoder = keras.models.Sequential([encoder, decoder])

# 3. Compile a model
autoencoder.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

# 4. Add noise to images
X_train = X_train + ((np.random.random(X_train.shape)) / 4)
X_test = X_test + ((np.random.random(X_test.shape)) / 4)

# 5. Train model
history = autoencoder.fit(X_train, X_train, epochs=10, validation_data=[X_test, X_test])

# 6. Test model (Display results)
plt.figure(figsize=(24, 6))

for i in range(8):
    plt.subplot(2, 8, i + 1)
    plt.imshow(X_test[i], cmap="binary")

    plt.subplot(2, 8, 8 + 1 + i)
    pred = autoencoder.predict(X_test[i].reshape(1, 28, 28))
    plt.imshow(pred.reshape(28, 28), cmap="binary")

plt.show()


"""
############################################################
CONVOLUTIONAL AUTOENCODER
############################################################
"""
# 1. Define encoder and decoder and combine them for training
encoder = keras.models.Sequential([
    keras.layers.Reshape([28, 28, 1], input_shape=[28, 28]),  # Input for 1 image
    keras.layers.Conv2D(16, kernel_size=(3, 3), padding="same", activation="relu"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
    keras.layers.MaxPool2D(pool_size=2)
])

latent_representation = encoder.predict(X_test[0].reshape((1, 28, 28)))
print(f"\n\nLatent representation: {latent_representation}.\n\n")

decoder = keras.models.Sequential([
    keras.layers.Conv2DTranspose(
        32, kernel_size=(3, 3), strides=2, padding="valid",
        activation="relu", input_shape=[3, 3, 64]),
    keras.layers.Conv2DTranspose(
        16, kernel_size=(3, 3), strides=2, padding="same", activation="relu"),
    keras.layers.Conv2DTranspose(
        1, kernel_size=(3, 3), strides=2, padding="same", activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])

autoencoder = keras.models.Sequential([encoder, decoder])

# 2. Compile a model
autoencoder.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

# 3. Train model
history = autoencoder.fit(X_train, X_train, epochs=10, validation_data=[X_test, X_test])

# 4. Test model (Display results)
plt.figure(figsize=(24, 6))

for i in range(8):
    plt.subplot(2, 8, i + 1)
    plt.imshow(X_test[i], cmap="binary")

    plt.subplot(2, 8, 8 + 1 + i)
    pred = autoencoder.predict(X_test[i].reshape(1, 28, 28))
    plt.imshow(pred.reshape(28, 28), cmap="binary")

plt.show()

# 5. Weights encoder learned
plt.figure(figsize=(24, 6))

for i in range(8 * 8):  # √64=8
    plt.subplot(8, 8, i + 1)
    plt.imshow(encoder.layers[-2].weights[0][:, :, 0, i])

plt.show()


"""
############################################################
DEEP NEURAL NETWORKS
############################################################
"""
# 1. Load dataset
dataset = keras.datasets.cifar10
(X_train, y_train), (X_test, y_test) = dataset.load_data()

# 2. Data preprocessing for ResNet50 (input: img 224x224)
# 2.1 Resize images to match input
X_train = tf.image.resize(X_train, (224, 224))
X_test = tf.image.resize(X_test, (224, 224))

# 2.2. Preporcess data
X_train = keras.applications.resnet50.preprocess_input(X_train)
X_test = keras.applications.resnet50.preprocess_input(X_test)

# 2.3. Convert results to one hot encoding
Y_train = keras.utils.to_categorical(y_train, 10)
Y_test = keras.utils.to_categorical(y_test, 10)

# 3. Define a model
# 3.1. Load pretrained model
base_model = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# base_model.trainable = False

# 3.2. Customize model
model = keras.models.Sequential()
model.add(base_model)
model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.Dense(1024, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

# 4. Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Train model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.3)

# 5.1. Visualize history
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True), plt.ylim(0, 1), plt.show()
