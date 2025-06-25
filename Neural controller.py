import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import joblib
import torch
import torch.nn as nn

# Определение архитектуры нейронной сети
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

# Загрузка обученной модели PyTorch
neural_model = NeuralNetwork(input_size=5)  # Входные параметры: R, C, time_step, u_out, pressure
neural_model.load_state_dict(torch.load("model.pth"))
neural_model.eval()

scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# PID-регулятор с использованием нейронной сети
class NeuralController:
    def __init__(self):
        self.model = neural_model

    def compute(self, R, C, time_step, u_out, pressure):
        # Масштабирование входных данных
        input_data = scaler_X.transform([[R, C, time_step, u_out, pressure]])
        input_data = torch.tensor(input_data, dtype=torch.float32)

        # Предсказание управляющего воздействия
        with torch.no_grad():
            u_in_scaled = self.model(input_data).item()

        # Обратное масштабирование предсказанного значения
        u_in = scaler_y.inverse_transform([[u_in_scaled]]).item()

        # Ограничение диапазона [0, 100]
        return max(0, min(100, u_in))

# Загрузка данных
print("Загрузка данных...")
data = pd.read_csv("train.csv")
data = data.drop(columns=["pressure"])
data_index = 0

# Загрузка модели для расчета давления
lung_model = joblib.load("lung_model.pkl")

# Инициализация параметров
time_step = 0.0
pressure = 0
u_in = 0
pid = NeuralController()
system_enabled = True  # Переключатель состояния системы

# Подготовка данных для графиков
global_time = 0
previous_time_step = 0
time_data = []
pressure_data = []
control_data = []
setpoint_data = []

# Создание окна tkinter
root = tk.Tk()
root.title("PID Control with tkinter Sliders")

# Создание графиков
fig, ax = plt.subplots(2, 1, figsize=(8, 6))
plt.subplots_adjust(hspace=0.4)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Загрузка изображений переключателя и лампочки
turn_on_image = ImageTk.PhotoImage(Image.open("turn_on.png"))
turn_off_image = ImageTk.PhotoImage(Image.open("turn_off.png"))
light_image = ImageTk.PhotoImage(Image.open("light.png"))
dark_image = ImageTk.PhotoImage(Image.open("dark.png"))

# Добавление нижней рамки для элементов управления
bottom_frame = tk.Frame(root)
bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

# Рамка для ползунков (левая часть)
slider_frame = tk.Frame(bottom_frame)
slider_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

# Рамка для лампочки и переключателя (правая часть)
control_frame = tk.Frame(bottom_frame)
control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

# Лампочка
lamp_frame = tk.Frame(control_frame)
lamp_frame.pack(side=tk.LEFT, padx=5)
lamp_label = tk.Label(lamp_frame, image=light_image)
lamp_label.pack()
tk.Label(lamp_frame, text="Выходной клапан").pack()

# Переключатель
switch_frame = tk.Frame(control_frame)
switch_frame.pack(side=tk.LEFT, padx=5)
switch_label = tk.Label(switch_frame, image=turn_on_image)
switch_label.pack()
tk.Label(switch_frame, text="Переключатель аппаратуры").pack()

# Функция переключения состояния

def toggle_system():
    global system_enabled
    system_enabled = not system_enabled
    if system_enabled:
        switch_label.config(image=turn_on_image)
    else:
        switch_label.config(image=turn_off_image)

# Обработчик клика по переключателю
switch_label.bind("<Button-1>", lambda event: toggle_system())

# Функция обновления графиков

def update_plot():
    global time_step, u_in, pressure, data_index, global_time, previous_time_step

    dt = 0.1  # Интервал времени

    k = 0.5

    # Чтение параметров текущей записи
    current_data = data.iloc[data_index]
    time_step = current_data["time_step"]
    u_out = current_data["u_out"]

    # Обновление состояния лампочки
    if u_out == 1:
        lamp_label.config(image=light_image)
    else:
        lamp_label.config(image=dark_image)

    # Получение значений R и C из ползунков
    R = slider_R.get()
    C = slider_C.get()

    # Обновление глобального времени
    if time_step < previous_time_step:  # Новый цикл вдоха
        global_time += time_step
    else:  # Текущий цикл
        global_time += time_step - previous_time_step
    previous_time_step = time_step

    # Цикличность данных
    data_index = (data_index + 1) % len(data)

    # Управляющее воздействие через нейронную сеть
    u_in = pid.compute(R, C, time_step, u_out, pressure) if system_enabled else 0

    # Прогноз давления через сохранённую модель (lung_model.pkl)
    model_input_data = pd.DataFrame([[time_step, u_in, u_out, R, C]], columns=['time_step', 'u_in', 'u_out', 'R', 'C'])
    if u_in != 0:
        pressure = lung_model.predict(model_input_data)[0]
    else:
        pressure = lung_model.predict(model_input_data)[0] * k
    print(model_input_data)

    # Обновление данных для графиков
    time_data.append(global_time)
    pressure_data.append(pressure)
    control_data.append(u_in)

    # Ограничение числа точек на графике
    max_points = 100
    if len(time_data) > max_points:
        time_data.pop(0)
        pressure_data.pop(0)
        control_data.pop(0)

    # Обновление графиков
    ax[0].clear()
    ax[1].clear()
    ax[0].plot(time_data, pressure_data, label="Давление (Pressure)", color="blue")
    ax[1].plot(time_data, control_data, label="Управляющее воздействие (u_in)", color="orange")
    ax[0].set_title("Давление в легких")
    ax[1].set_title("Управляющее воздействие")
    ax[0].legend()
    ax[1].legend()

    canvas.draw()
    root.after(100, update_plot)

# Ползунки
slider_R = ttk.Scale(slider_frame, from_=10, to=50, value=20, orient=tk.HORIZONTAL, length=200)
slider_C = ttk.Scale(slider_frame, from_=10, to=50, value=20, orient=tk.HORIZONTAL, length=200)

# Метки для ползунков
tk.Label(slider_frame, text="R").grid(row=0, column=0, padx=5, pady=5)
slider_R.grid(row=0, column=1, padx=5, pady=5)

tk.Label(slider_frame, text="C").grid(row=1, column=0, padx=5, pady=5)
slider_C.grid(row=1, column=1, padx=5, pady=5)

# Запуск обновления графика
root.after(100, update_plot)
root.mainloop()
