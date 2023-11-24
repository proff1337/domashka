import pandas as pd
import matplotlib.pyplot as plt

# Шаг 1: Загрузка данных
# Замените 'your_dataset.csv' на путь к вашему файлу данных
file_path = 'your_dataset.csv'
data = pd.read_csv(file_path)

# Шаг 2: Анализ данных
# 2.1. Общий вид данных
print("Первые несколько строк данных:")
print(data.head())

print("\nИнформация о данных:")
print(data.info())

# 2.2. Простой анализ данных
numeric_summary = data.describe()
print("\nПростой анализ числовых данных:")
print(numeric_summary)

# Шаг 3: Визуализация данных
# 3.1. Гистограмма распределения данных
data['numeric_column'].hist(bins=20)
plt.title('Гистограмма распределения данных')
plt.xlabel('Значения')
plt.ylabel('Частота')
plt.show()

# 3.2. График зависимости между двумя столбцами
plt.scatter(data['column1'], data['column2'])
plt.title('График зависимости между column1 и column2')
plt.xlabel('column1')
plt.ylabel('column2')
plt.show()

# Шаг 4: Отчет
# 4.1. Теперь я знаю как подключать библиотеки в питоне, устанавливать pandas и matplotlib, читать csv файл, извлекать из него значение