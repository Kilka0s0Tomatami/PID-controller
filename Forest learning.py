import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

## Чтение данных
print("Чтение данных...")
file_path = "train.csv"

# Читаем только первые 100000 строк
target_rows = 100000
data = pd.read_csv(file_path, nrows=target_rows)
print(f"Выбрано строк: {len(data)}")

# Вывод минимальных, максимальных и средних значений для каждой колонки
print("\nСтатистика по колонкам:")
for column in data.columns:
    min_val = data[column].min()
    max_val = data[column].max()
    mean_val = data[column].mean()
    print(f"{column}:")
    print(f"  Минимальное значение: {min_val}")
    print(f"  Максимальное значение: {max_val}")
    print(f"  Среднее значение: {mean_val:.2f}")

# Подготовка данных
print("\nПодготовка данных для обучения...")
X = data[['time_step', 'u_in', 'u_out', 'R', 'C']]
y = data['pressure']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
print("\nОбучение модели...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Оценка модели
print("\nОценка модели...")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Среднеквадратичная ошибка: {mse}")

# Сохранение модели
print("\nСохранение модели...")
joblib.dump(model, "lung_model.pkl")
print("Модель сохранена в файл lung_model.pkl")
