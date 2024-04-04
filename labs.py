import time
import random
from scipy.optimize import minimize
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import random

# GUI окно
gui = tk.Tk()
gui.title("Оптимизация")

# Создание трехмерного графика
fig = plt.figure(figsize=(8, 6))
ax = plt.axes(projection='3d')

# Вставка графика в GUI окно
canvas = FigureCanvasTkAgg(fig, master=gui)
canvas.get_tk_widget().pack()

def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

parc=3

def generate_pop(pers):
    population = [[0] * 3 for i in range(pers)]
    popv=-5
    for i in range(pers):
        population[i][1]=popv
        population[i][2]=popv
        population[i][0]=rosenbrock(population[i][0],population[i][1])
        popv+=10/pers
    return population

def selection(population,pers,mut_coef,mut_val):
    population.sort()
    parents=population[0:parc]
    newpop=population
    for i in range(1,pers):
        newpop[i][1]=parents[int(random.uniform(0,parc-1))][1]
        newpop[i][2]=parents[int(random.uniform(0,parc-1))][2]
        if random.randint(0, 1) < mut_coef:
            newpop[i][1]+=mut_val
        if random.randint(1, 50) < mut_coef:
            newpop[i][2]+=mut_val
        newpop[i][0]=rosenbrock(newpop[i][1],newpop[i][2])
    return newpop

# Функция Экли
def ackley_function(x, y):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * ( x**2 + y** 2))) - np.exp(
        0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.exp(1) + 20

# Градиент функции Экли
def gradient_ackley(x, y):
    part1 = 0.04 * x * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) / np.sqrt(0.5 * (x ** 2 + y ** 2))
    part2 = 0.04 * y * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) / np.sqrt(0.5 * (x ** 2 + y ** 2))
    part3 = 2 * np.pi * np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) * np.sin(2 * np.pi * x)
    part4 = 2 * np.pi * np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) * np.sin(2 * np.pi * y)
    dx = part1 - part3
    dy = part2 - part4
    return dx, dy

# Функция Матьяса
def matyas_function(x, y):
    return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y

# Градиент функции Матьяса
def gradient_matyas(x, y):
    dx = 0.52 * x - 0.48 * y
    dy = 0.52 * y - 0.48 * x
    return dx, dy

#Функция для 2 ЛР
def lr2Function(x1,x2):
    return 2 * x1 * x1 + 3 * x2 * x2 + 4 * x1 * x2 - 6 * x1 - 3 * x2

# Функция Букина N6
def bukin_function(x, y):
    return 100 * np.sqrt(np.abs(y - 0.01 * x ** 2)) + 0.01 * np.abs(x + 10)

# Градиент функции Букина N6
def gradient_bukin(x, y):
    dx = 0.01 * (20 - 2 * x) / (2 * np.sqrt(np.abs(y - 0.01 * x ** 2))) + np.sign(x + 10)
    dy = np.float128(50) * (y - 0.01 * x ** 2) / np.sqrt(np.abs(y - 0.01 * x ** 2))
    return dx, dy

# Выбор функции
function_label = tk.Label(gui, text="Тестовые функции:")
function_label.pack(side=tk.TOP)
function_var = tk.StringVar(value="Экли")
function_dropdown = ttk.OptionMenu(gui, function_var, "Экли", "Экли", "lr2Function", "Матьяса", "Букин", "Розенброк")
function_dropdown.pack(side=tk.TOP)

# Выбор метода оптимизации
method_label = tk.Label(gui, text="Выберите метод:")
method_label.pack(side=tk.TOP)

method_var = tk.StringVar(value="Градиентный спуск")
method_dropdown = ttk.OptionMenu(gui, method_var, "Градиентный спуск", "Градиентный спуск", "Симплекс метод", "Генетический алгоритм")
method_dropdown.pack(side=tk.TOP)

points_var = tk.StringVar(value=100)

# Поле ввода для количества точек
points_label = tk.Label(gui, text="Количество точек:")
points_label.pack(side=tk.TOP)

points_entry = ttk.Entry(gui, textvariable=points_var)
points_entry.pack(side=tk.TOP)

# Функция, вызываемая при обновлении значения количества точек
def update_points_entry(*args):
    try:
        value = int(points_entry.get())
        points_var.set(value)
    except ValueError:
        points_entry.delete(0, tk.END)
        points_entry.insert(0, str(points_var.get()))

points_var.trace("w", update_points_entry)

# Функция, вызываемая при нажатии кнопки "Визуализировать"
def visualize():
    # Получение выбранной функции, метода оптимизации и количества точек
    selected_function = function_var.get()
    selected_method = method_var.get()
    # Получение выбранного количества точек
    num_points = points_var.get()

    # Определение функции и градиента
    if selected_function == "Букин":
        function = bukin_function
        gradient = gradient_bukin
    elif selected_function == "Матьяса":
        function = matyas_function
        gradient = gradient_matyas
    elif selected_function == "Экли":
        function = ackley_function
        gradient = gradient_ackley
    elif selected_function == "lr2Function":
        function = lr2Function
    elif selected_function == "Розенброк":
        function = rosenbrock
    else:
        messagebox.showerror("Выберите другую функцию")
        return

    # Создание сетки значений
    x = np.linspace(-5, 5, 500)
    y = np.linspace(-5, 5, 500)
    X, Y = np.meshgrid(x, y)
    Z = function(X, Y)

    # Рисование функции
    ax.cla()
    ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.7)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(selected_function)
    ax.view_init(elev=30, azim=45)

    #Если выбран симплекс метод
    if selected_method=="Генетический алгоритм":
        points_text.delete("1.0", tk.END)
        canvas.draw()
        pers_count=30
        iters= 5 # points_var.get()
        mutc=0.3
        mutv=0.1
        pop=generate_pop(pers_count)
        points=[]
        for t in range(pers_count):
            point=ax.scatter(pop[t][1],pop[t][2],rosenbrock(pop[t][1],pop[t][2]),c='black',alpha=1.0)
            points.append(point)
            canvas.draw()
            gui.update()
            point.remove()
        time.sleep(0.025)
        # for point in points:
        #     point.remove()
        for i in range(4):
            pop=selection(pop,pers_count,mutc,mutv)
            points = []
            for j in range(1,pers_count):
                point = ax.scatter(pop[j][1], pop[j][2], rosenbrock(pop[j][1], pop[j][2]), c='b', alpha=0.1)
                points.append(point)
                points_text.insert(tk.END, f"Точка {j + 1}: ({pop[j][1]:.2f}, {pop[j][2]:.2f})\n")
            point = ax.scatter(pop[0][1], pop[0][2], rosenbrock(pop[0][1], pop[0][2]), c='r', alpha=1.0)
            time.sleep(0.025)
            points.append(point)
            points_text.insert(tk.END, f"{i+1}. (x= {pop[0][1]:.4f}, y= {pop[0][2]:.4f}) f= {pop[0][0]:.4f}\n")
        
    #Если выбран симплекс метод
    if selected_method == "Симплекс метод":
        points_text.delete("1.0", tk.END)

        def lr2func(x_i):
            x1 = x_i[0]
            x2 = x_i[1]
            return 2 * x1 * x1 + 3 * x2 * x2 + 4 * x1 * x2 - 6 * x1 - 3 * x2

        canvas.draw()
        x_start = 5
        y_start = 5

        x0 = np.array([x_start, y_start])
        point = ax.scatter(x0[0], x0[1], function(x0[0], x0[1]), c='r', alpha=1.0)
        point.remove()

        def saver(xk):
            point = ax.scatter(xk[0], xk[1], function(xk[0], xk[1]), c='r', alpha=1.0)
            points_text.insert(tk.END, f"({xk[0]:.4f}, {xk[1]:.4f}) f= {function(xk[0], xk[1]):.4f}\n")
            canvas.draw()
            time.sleep(0.3)
            point.remove()

        b = (0, float("inf"))
        bounds = (b, b)
        min_res = minimize(lr2func, x0, method="SLSQP", bounds=bounds, constraints={'type': 'eq', 'fun': lr2func},
                           callback=saver)
        point = ax.scatter(min_res['x'][0], min_res['x'][1], function(min_res['x'][0], min_res['x'][1]), c='r',
                           alpha=1.0)

    # Если выбран градиентный спуск
    if selected_method == "Градиентный спуск":
        # Очистка текстового поля
        points_text.delete("1.0", tk.END)

        # Начальные значения
        x_start = -5
        y_start = -5
        learning_rate = 0.01
        num_iterations = int(num_points)

        # Если выбран градиентный спуск
        if selected_method == "Градиентный спуск":
            # Очистка текстового поля
            points_text.delete("1.0", tk.END)

            # Начальные значения
            x_start = 3
            y_start = 0
            learning_rate = 0.01
            num_iterations = int(num_points)
                
            # Градиентный спуск
            x_history = [x_start]
            y_history = [y_start]
            point = ax.scatter(x_history, y_history, function(np.array(x_history), np.array(y_history)), c='b', alpha=1.0)
            canvas.draw()
            gui.update()
            for i in range(num_iterations-1):
                
                point.remove()
                grad_x, grad_y = gradient(x_start, y_start)
                x_start = x_start - learning_rate * grad_x
                y_start = y_start - learning_rate * grad_y
                fi=function(x_start, y_start)
                point=ax.scatter(x_start, y_start, fi, c='b', alpha=1.0)
                points_text.insert(tk.END, f"Точка {i + 1}: ({x_start:.2f}, {y_start:.2f})\n")
                canvas.draw()
                gui.update()

    # Обновление графика
    canvas.draw()

visualize_button = tk.Button(gui, text="Построить", command=visualize)
visualize_button.pack()

# Текстовое поле для вывода точек
points_text = tk.Text(gui, height=10, width=40)
points_text.pack()

# Запуск GUI
gui.mainloop()