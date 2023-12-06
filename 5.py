import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

# Шаг 1: Загрузка данных
# Пример: последовательность чисел от 1 до 100
data = np.arange(1, 101, 1)

# Создаем последовательные наборы данных для ввода и вывода RNN
input_sequence = [data[i:i+5] for i in range(len(data)-5)]
output_sequence = [data[i+5] for i in range(len(data)-5)]

# Преобразуем данные в массивы NumPy
X = np.array(input_sequence).reshape(-1, 5, 1)
y = np.array(output_sequence)

# Шаг 2: Создание рекуррентной нейронной сети (RNN)
model = Sequential()
model.add(SimpleRNN(50, activation='relu', input_shape=(5, 1)))
model.add(Dense(1))

# Шаг 3: Обучение RNN
model.compile(optimizer=Adam(), loss='mean_squared_error')
model.fit(X, y, epochs=50, batch_size=1)

# Шаг 4: Оценка производительности модели RNN
# На практике использовать отдельный тестовый набор данных
test_data = np.arange(101, 111, 1)
X_test = np.array([test_data[i:i+5] for i in range(len(test_data)-5)])
y_test = np.array([test_data[i+5] for i in range(len(test_data)-5)])

# Предсказание на тестовых данных
predictions = model.predict(X_test.reshape(-1, 5, 1))

# Оценка производительности
mse = np.mean((predictions.flatten() - y_test) ** 2)
print("Среднеквадратичная ошибка на тестовых данных:", mse)