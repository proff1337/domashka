import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Шаг 1: Загрузка данных
# 1.1. Загрузка данных CIFAR-10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# 1.2. Подготовка данных
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Шаг 2: Создание сверточной нейронной сети
# 2.1. Импорт библиотек
model = models.Sequential()

# 2.2. Создание сверточной нейронной сети
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Шаг 3: Обучение сверточной нейронной сети
# 3.1. Настройка параметров обучения
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 3.2. Обучение модели
model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.2)

# Шаг 4: Оценка производительности модели
# 4.1. Оценка на тестовых данных
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Точность на тестовых данных:", test_acc)