import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Шаг 1: Загрузка данных
# Замените 'your_dataset.csv' на путь к вашему файлу данных
file_path = 'your_dataset.csv'
data = pd.read_csv(file_path)

# Шаг 2: Очистка данных
# 2.1. Оценка состояния данных
print("Информация о данных до очистки:")
print(data.info())

# 2.2. Идентификация и обработка пропущенных значений
imputer = SimpleImputer(strategy='mean')  # Используем среднее для заполнения пропущенных значений числовых столбцов
data_filled = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Шаг 3: Преобразование данных
# 3.1. Преобразование категориальных признаков в числовые с использованием One-Hot Encoding
# Замените 'categorical_column' на реальное имя вашего категориального столбца
encoder = OneHotEncoder(sparse=False, drop='first')
data_encoded = pd.DataFrame(encoder.fit_transform(data_filled[['categorical_column']]), columns=['encoded_category'])

# 3.2. Масштабирование числовых признаков
# Замените 'numeric_column' на реальное имя вашего числового столбца
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data_filled[['numeric_column']]), columns=['scaled_numeric'])

# Шаг 4: Сохранение предобработанных данных
preprocessed_data = pd.concat([data_encoded, data_scaled, data_filled.drop(['categorical_column', 'numeric_column'], axis=1)], axis=1)
preprocessed_data.to_csv('preprocessed_data.csv', index=False)

print("Предобработанные данные сохранены в preprocessed_data.csv")