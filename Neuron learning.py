import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Загрузка и подготовка данных
print("Загрузка данных...")
data = pd.read_csv('train.csv')
data = data.dropna()
print("Данные загружены и обработаны (удалены NaN).")

# Ограничение данных до 100 000 строк
data = data.iloc[:100000]  # Используем только первые 100 000 строк
print("Ограничение данных до 100,000 строк.")

features = ['R', 'C', 'time_step', 'u_out', 'pressure']
target = 'u_in'

X = data[features].values
y = data[target].values

# Разделение на тренировочные и тестовые данные
print("Разделение данных на тренировочные и тестовые выборки...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Разделение завершено.")

# Стандартизация данных
print("Стандартизация данных...")
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test = scaler_y.transform(y_test.reshape(-1, 1))
print("Стандартизация завершена.")

# 2. Создание PyTorch Dataset
print("Создание PyTorch Dataset...")
class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = RegressionDataset(X_train, y_train)
test_dataset = RegressionDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print("PyTorch Dataset создан.")

# 3. Определение нейронной сети
print("Создание модели...")
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

model = NeuralNetwork(input_size=X.shape[1])
print("Модель создана.")

# 4. Настройка обучения
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. Обучение модели
print("Начало обучения модели...")
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        predictions = model(X_batch).squeeze()
        loss = criterion(predictions, y_batch.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}] завершён, Loss: {loss.item():.4f}")

print("Обучение завершено.")

# Сохранение масштабаторов
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")

# Сохранение модели
torch.save(model.state_dict(), "model.pth")

# Сохранение модели
torch.save(model.state_dict(), 'model.pth')
print("Модель сохранена в файл 'model.pth'.")

# 6. Оценка модели
print("Оценка модели на тестовых данных...")
model.eval()
with torch.no_grad():
    test_losses = []
    for X_batch, y_batch in test_loader:
        predictions = model(X_batch).squeeze()
        loss = criterion(predictions, y_batch.squeeze())
        test_losses.append(loss.item())

    print(f"Test Loss: {sum(test_losses) / len(test_losses):.4f}")
print("Оценка завершена.")

# 7. Прогнозирование для breath.csv
print("Загрузка файла 'breath.csv'...")
breath_data = pd.read_csv('breath.csv')
breath_data = breath_data.dropna()

# Подготовка данных для предсказания
print("Обработка данных из 'breath.csv'...")
X_breath = breath_data[features].values
X_breath = scaler_X.transform(X_breath)  # Применяем сохранённый масштабатор

# Преобразование данных в тензор
X_breath_tensor = torch.tensor(X_breath, dtype=torch.float32)

# Прогнозирование
print("Выполнение предсказаний для 'breath.csv'...")
model.eval()
with torch.no_grad():
    predictions = model(X_breath_tensor).squeeze().numpy()  # Прогнозы

# Обратное масштабирование предсказаний
predictions_rescaled = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()

# Вывод результатов
print("Предсказания завершены. Результаты:")
for i, pred in enumerate(predictions_rescaled):
    print(f"Строка {i + 1}: u_in = {pred:.4f}")
