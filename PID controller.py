import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import joblib

# PID-регулятор
class PIDController:
    def __init__(self, kp, ki, kd, setpoint=25, integral_limit=50):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.integral = 0
        self.prev_error = 0
        self.integral_limit = integral_limit

    def compute(self, current_value, dt, enabled=True):
        if not enabled:  # Если система отключена, управление равно нулю
            return 0
        error = self.setpoint - current_value
        self.integral += error * dt
        self.integral = max(-self.integral_limit, min(self.integral, self.integral_limit))
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

# Загрузка данных
data = pd.read_csv("train.csv")
data = data.drop(columns=["pressure"])
data_index = 0

# Загрузка модели
model = joblib.load("lung_model.pkl")

# Инициализация параметров
time_step = 0.0
pressure = 0
u_in = 0
pid = PIDController(kp=3.0, ki=0.2, kd=1.0, setpoint=25)
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

    k = 0.75

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

    # Обновление PID
    pid.kp = slider_kp.get()
    pid.ki = slider_ki.get()
    pid.kd = slider_kd.get()

    # Управляющее воздействие через PID-регулятор
    if u_out == 1:  # Если выходной клапан открыт
        u_in = 0
    else:
        u_in = pid.compute(pressure, dt, enabled=system_enabled)
        u_in = max(0, min(100, u_in)) if system_enabled else 0  # Ограничение по диапазону

    # Прогноз давления через модель
    input_data = pd.DataFrame([[time_step, u_in, u_out, R, C]], columns=['time_step', 'u_in', 'u_out', 'R', 'C'])
    pressure = model.predict(input_data)[0] * k

    # Обновление данных для графиков
    time_data.append(global_time)
    pressure_data.append(pressure)
    control_data.append(u_in)
    setpoint_data.append(pid.setpoint)

    # Ограничение числа точек на графике
    max_points = 100
    if len(time_data) > max_points:
        time_data.pop(0)
        pressure_data.pop(0)
        control_data.pop(0)
        setpoint_data.pop(0)

    # Обновление графиков
    ax[0].clear()
    ax[1].clear()
    ax[0].plot(time_data, pressure_data, label="Давление (Pressure)", color="blue")
    ax[0].plot(time_data, setpoint_data, label="Целевое давление (Setpoint)", linestyle="--", color="red")
    ax[1].plot(time_data, control_data, label="Управляющее воздействие (u_in)", color="orange")
    ax[0].set_title("Давление в легких")
    ax[1].set_title("Управляющее воздействие")
    ax[0].legend()
    ax[1].legend()

    canvas.draw()
    root.after(100, update_plot)

# Ползунки
slider_kp = ttk.Scale(slider_frame, from_=0.1, to=10.0, value=pid.kp, orient=tk.HORIZONTAL, length=200)
slider_ki = ttk.Scale(slider_frame, from_=0.0, to=2.0, value=pid.ki, orient=tk.HORIZONTAL, length=200)
slider_kd = ttk.Scale(slider_frame, from_=0.0, to=5.0, value=pid.kd, orient=tk.HORIZONTAL, length=200)
slider_R = ttk.Scale(slider_frame, from_=10, to=50, value=20, orient=tk.HORIZONTAL, length=200)
slider_C = ttk.Scale(slider_frame, from_=10, to=50, value=20, orient=tk.HORIZONTAL, length=200)

# Метки для ползунков
tk.Label(slider_frame, text="Kp").grid(row=0, column=0, padx=5, pady=5)
slider_kp.grid(row=0, column=1, padx=5, pady=5)

tk.Label(slider_frame, text="Ki").grid(row=1, column=0, padx=5, pady=5)
slider_ki.grid(row=1, column=1, padx=5, pady=5)

tk.Label(slider_frame, text="Kd").grid(row=2, column=0, padx=5, pady=5)
slider_kd.grid(row=2, column=1, padx=5, pady=5)

tk.Label(slider_frame, text="R").grid(row=3, column=0, padx=5, pady=5)
slider_R.grid(row=3, column=1, padx=5, pady=5)

tk.Label(slider_frame, text="C").grid(row=4, column=0, padx=5, pady=5)
slider_C.grid(row=4, column=1, padx=5, pady=5)

# Запуск обновления графика
root.after(100, update_plot)
root.mainloop()
