import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np


#1
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000,28,28,1)).astype("float32")/255
test_images = test_images.reshape((10000,28,28,1)).astype("float32")/255

train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

model = models.Sequential()

model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Точность на тестовых данных: ", test_acc)
print("Потери на тестовых данных: ", test_loss)

image_index = 0

image = test_images[image_index]

image = np.expand_dims(image, axis=0)

predictions = model.predict(image)
print(predictions)

predicted_class = np.argmax(predictions)
trust_class = np.argmax(test_labels[image_index])

print("Предсказанный класс: ", predicted_class)
print("Настоящий класс: ", trust_class)