import numpy as np
from scipy.integrate import solve_ivp as scipy_solve_ivp
import sklearn.manifold as skl_manifold
import matplotlib.pyplot as plt
import time
from math import ceil as m_ceil

from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showerror

import warnings
warnings.filterwarnings('ignore')  # убирает вывод предупреждений Isomap в консоль




## Дифференциальные уравнения ##
def roessler(t, state, a, b, c):
    x, y, z = state

    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)

    return [dx, dy, dz]


def lorenz(t, state, sigma, r, b):
    x, y, z = state

    dx = sigma * (y - x)
    dy = r * x - y - x * z
    dz = -b * z + x * y

    return [dx, dy, dz]


def gin_aa(t, state, m, g):
    x, y, z = state

    f = lambda x: x ** 2 * int(x >= 0)

    dx = m * x + y - x * z
    dy = -x
    dz = -g * z + g * f(x)

    return [dx, dy, dz]
## -------------- ##


## Функция, возвращающая значение координаты по её индексу ##
def coordinate(index):
    def return_coordinate(t, state, *args):
        return state[index]
    
    return return_coordinate
## -------------- ##


## Системы и настройки вычисления их динамики ##
systems = {
    'roessler': {
        'name': 'Система Рёсслера',
        'eng_name': 'roessler',
        'func': roessler,
        'param_names': ['a', 'b', 'c'],
        'variable_names': ['x', 'y', 'z'],
        'params': {
            'doubling': {
                'start_params': [0.1, 0.1, 5],
                'varying_param_index': 2,
                'param_delta': 0.05,
                'variation_limit': 10,
                'direction': -4,
                },
            'intermittency': {
                'start_params': [0.24, 0.2, 4.6],
                'varying_param_index': 2,
                'param_delta': 0.005,
                'variation_limit': 4.7,
                'direction': -4,
                },
            },
        'observed_coord_index': 1,
        'cross_section_coord_index': 0,
        'init_values': [5.8, 1, 0],
        'time': [0, 700, 0.02, 650],
        },
    
    'lorenz': {
        'name': 'Система Лоренца',
        'eng_name': 'lorenz',
        'func': lorenz,
        'param_names': ['sigma', 'r', 'b'],
        'variable_names': ['x', 'y', 'z'],
        'params': {
            'intermittency': {
                'start_params': [10, 166, 8 / 3],
                'varying_param_index': 1,
                'param_delta': 0.025,
                'variation_limit': 167,
                'direction': -4,
                },
            },
        'observed_coord_index': 1,
        'cross_section_coord_index': 0,
        'init_values': [5.8, 1, 0],
        'time': [0, 700, 0.02, 650],
        },

    'gin_aa': {
        'name': 'ГИН Анищенко-Астахова',
        'eng_name': 'gin_aa',
        'func': gin_aa,
        'param_names': ['m', 'g'],
        'variable_names': ['x', 'y', 'z'],
        'params': {
            'doubling_1': {
                'start_params': [0.4, 0.66],
                'varying_param_index': 0,
                'param_delta': 0.005,
                'variation_limit': 1.9,
                'direction': 4,
                },
            'doubling_2': {
                'start_params': [0.966, 0.2],
                'varying_param_index': 0,
                'param_delta': 0.003,
                'variation_limit': 1.356,
                'direction': 4,
                },
            'something': {
                'start_params': [1.5, 0.2],
                'varying_param_index': 0,
                'param_delta': 0.001,
                'variation_limit': 1.531,
                'direction': -4,
                }
            },
        'observed_coord_index': 1,
        'cross_section_coord_index': 0,
        'init_values': [5.8, 1, 0],
        'time': [0, 700, 0.02, 625],
        }
    }
## -------------- ##


## Методы понижения размерности ##
drm_methods = {
    'tSNE': skl_manifold.TSNE(random_state=4),
    'Isomap': skl_manifold.Isomap()
    }
## -------------- ##


## Функция, сообщающая, установлены ли все настройки comboboxов ##
def all_settings_chosen():
    return all([system_type_chosen in system_type_names,
                params_set_chosen in params_set_names,
                drm_method_chosen in drm_method_names])
## -------------- ##


## Функция для system_type_combobox. Выбирает систему и настройки из systems, ##
## если на данный момент система не выбрана; иначе говорит её сбросить ##
def choose_system(event):
    global system_type_chosen, curr_system
    global params_set_chosen, params, params_set_names
    global initial_values, observed_coord, return_coordinate
    global t_start, t_end, dt, t_cut

    if len(axs_crs_bfd.collections) == 0:
        system_type_chosen = system_type_name.get()
        
        curr_system = systems[system_type_chosen]

        params = curr_system['params']
        params_set_names = list(params.keys())
        params_set_combobox['values'] = params_set_names
        initial_values = curr_system['init_values']
        observed_coord = curr_system['observed_coord_index']
        return_coordinate = coordinate(curr_system['cross_section_coord_index'])
        t_start, t_end, dt, t_cut = curr_system['time']

        params_set_chosen = None
        params_set_name.set('')

        if all_settings_chosen():
            start_calc_button.configure(state=tk.NORMAL)
        else:
            start_calc_button.configure(state=tk.DISABLED)
        
    else:
        system_type_combobox.current(system_type_names.index(system_type_chosen))
        message = ('Сначала необходимо сбросить текущие вычисления и настройки')
        showerror(title='Сбросьте текущие вычисления и настройки', message=message)
## -------------- ##
        

## Функция для params_set_Combobox. Выбирает параметры из списка параметров ##
## для системы, если они не выбраны при выбранной системе, иначе говорит, ##
## если система не выбрана, выбрать систему, или всё сбросить ##
def choose_params_set(event):
    global params_set_chosen, curr_params_set, curr_params, varying_param_index, param_delta, param_limit
    
    if len(axs_crs_bfd.collections) == 0:
        if system_type_chosen is not None:
            params_set_chosen = params_set_name.get()
            
            curr_params_set = params[params_set_chosen]

            curr_params = curr_params_set['start_params']
            varying_param_index = curr_params_set['varying_param_index']
            param_delta = curr_params_set['param_delta']
            param_limit = curr_params_set['variation_limit']
            return_coordinate.direction = curr_params_set['direction']

            axs_crs_bfd.set_xlabel(curr_system['param_names'][varying_param_index])
            axs_crs_bfd.set_ylabel(curr_system['variable_names'][observed_coord])
            axs_drm_bfd.set_xlabel(curr_system['param_names'][varying_param_index])

            if all_settings_chosen():
                start_calc_button.configure(state=tk.NORMAL)
                
        else:
            params_set_name.set('')
            message = ('Сначала необходимо выбрать систему')
            showerror(title='Выберите систему', message=message)
            
    else:
        params_set_combobox.current(params_set_names.index(params_set_chosen))
        message = ('Сначала необходимо сбросить текущие вычисления и настройки')
        showerror(title='Сбросьте текущие вычисления и настройки', message=message)
## -------------- ##


## Функция для drm_method_combobox. Выбирает метод понижения размерности из ##
## drm_methods, если на данный момент он не выбран; иначе говорит его сбросить ##
def choose_drm_method(event):
    global drm_method_chosen, drm_method
    
    if len(axs_crs_bfd.collections) == 0:
        drm_method_chosen = drm_method_name.get()
        
        drm_method = drm_methods[drm_method_chosen]

        if all_settings_chosen():
            start_calc_button.configure(state=tk.NORMAL)

    else:
        drm_method_combobox.current(drm_method_names.index(drm_method_chosen))
        message = ('Сначала необходимо сбросить текущие вычисления и настройки')
        showerror(title='Сбросьте текущие вычисления и настройки', message=message)
## -------------- ##
        

## Функция для SpanSelector. Осуществляет запоминание выбранного диапазона значений параметра ##
def get_param_limits(span_min, span_max):
    global param_min, param_max

    if len(axs_crs_bfd.collections):
        param_min, param_max = span_min, span_max
        show_phase_space_button.configure(state=tk.NORMAL)
## -------------- ##


## Функция, удаляющая данные (результаты .plot() и .scatter()) с графика ##
def clear_ax(ax, mode):
    if mode == 'lines':
        for i in range(len(ax.lines)):
            ax.lines[0].remove()
            
    elif mode == 'dots':
        for i in range(len(ax.collections)):
            ax.collections[0].remove()
## -------------- ##


## Функция для кнопки clear_button. Сбрасывает выбранную систему, параметры и ##
## метод понижения размерности, очищает графики ##
def clear_calcs():
    global system_type_chosen, curr_system
    global params_set_chosen, params, params_set_names
    global curr_params_set, curr_params, varying_param_index, param_delta, param_limit
    global initial_values, observed_coord, return_coordinate
    global t_start, t_end, dt, t_cut
    global drm_method_chosen, drm_method

    system_type_chosen, curr_system = None, None
    params_set_chosen, params = None, None
    params_set_names = []
    params_set_combobox['values'] = params_set_names
    curr_params_set, curr_params, varying_param_index, param_delta, param_limit = None, None, None, None, None
    initial_values, observed_coord, return_coordinate = None, None, None
    t_start, t_end, dt, t_cut = None, None, None, None
    param_min, param_max = None, None
    drm_method_chosen, drm_method = None, None

    system_type_name.set('')
    params_set_name.set('')
    drm_method_name.set('')

    start_calc_button.configure(state=tk.DISABLED)
    show_phase_space_button.configure(state=tk.DISABLED)

    axs_crs_bfd.set_xlabel('')
    axs_crs_bfd.set_ylabel('')
    axs_drm_bfd.set_xlabel('')

    axs_drm_bfd.set_title('Бифуркационная диаграмма (пониж. размерность)')
    axs_drm_time.set_title('Время применения метода пониж. размерности, с')

    clear_ax(axs_crs_bfd, 'dots')
    clear_ax(axs_drm_bfd, 'dots')
    clear_ax(axs_calc_time, 'lines')
    clear_ax(axs_drm_time, 'lines')
    fig_phase_trajectories.clear()
    fig_drm.clear()

    parameter_span.active = False

    canvas_crs_bfd.draw()
    toolbar_crs_bfd.update()
    canvas_drm_bfd.draw()
    toolbar_drm_bfd.update()
    canvas_calc_time.draw()
    canvas_drm_time.draw()
    canvas_phase_trajectories.draw()
    canvas_drm.draw()
## -------------- ##


## Функция для расчёта высот Figure, в которых будут изображаться фазовые ##
## портреты. Каждый изображается в ячейке "таблицы" из постоянного ##
## количества столбцов plots_in_column и меняющегося количества строк nrows ##
def adjust_portraits(nrows):
    phase_trajectories_height_pix = side_horiz_pix * 2 + plot_side_pix * nrows + title_pix * (nrows - 1)
    phase_trajectories_bottom = side_horiz_pix / phase_trajectories_height_pix
    phase_trajectories_top = 1 - phase_trajectories_bottom

    drm_height_pix = suptitle_pix + plot_side_pix * nrows + title_pix * (nrows - 1) + side_horiz_pix
    drm_bottom = side_horiz_pix / drm_height_pix
    drm_top = 1 - suptitle_pix / drm_height_pix

    fig_phase_trajectories.set_size_inches(width / dpi, phase_trajectories_height_pix / dpi)
    fig_phase_trajectories.subplots_adjust(top=phase_trajectories_top,
                                       bottom=phase_trajectories_bottom)

    fig_drm.set_size_inches(width / dpi, drm_height_pix / dpi)
    fig_drm.subplots_adjust(top=drm_top,
                            bottom=drm_bottom)
## -------------- ##
    



## Функция для кнопки start_calc_button. Строит бифуркационные диаграммы ##
def do_calculations():
    curr_params = curr_params_set['start_params'].copy()
    t_array = np.arange(t_cut, t_end, dt)
    
    step_num = 0
    step_num_max = m_ceil((param_limit - curr_params[varying_param_index]) / param_delta + 1)
    steps = []
    calc_time = []
    drm_time = []

    drm_method.set_params(n_components=1)

    print(f"Начато построение бифуркационных диаграмм для {curr_system['name']}, {params_set_chosen}, {drm_method_chosen}")

    while curr_params[varying_param_index] - param_limit < param_delta:
        t_calc_start = time.time_ns()
        solution = scipy_solve_ivp(curr_system['func'], (t_start, t_cut), initial_values, args=curr_params, t_eval=[t_cut])
        solution = scipy_solve_ivp(curr_system['func'], (t_cut, t_end), solution.y[:, -1], args=curr_params, t_eval=t_array, events=return_coordinate)
        t_calc = (time.time_ns() - t_calc_start) / 10 ** 9
        calc_time.append(t_calc)

        if len(solution.y_events[0]):
            variation = [curr_params[varying_param_index]] * len(solution.y_events[0])
            axs_crs_bfd.scatter(variation, solution.y_events[0][:, observed_coord], color=(0.14, 0, 0.5, 1), s=0.2)
        
            t_drm_start = time.time_ns()
            after_dim_red = drm_method.fit_transform(solution.y[:, -100:].T)
            t_drm = (time.time_ns() - t_drm_start) / 10 ** 9
            drm_time.append(t_drm)
            
            if len(after_dim_red):
                variation = [curr_params[varying_param_index]] * len(after_dim_red)
                axs_drm_bfd.scatter(variation, after_dim_red, color=(0.14, 0, 0.5, 1), s=0.2)
        else:
            drm_time.append(0)
            print(f'    На шаге {step_num} в сечении нет точек')
        
        steps.append(step_num)
        step_num += 1
        if step_num % step_info == 0:
            print(f"    Шагов выполнено: {step_num} из ~{step_num_max}")
       
        curr_params[varying_param_index] += param_delta

    print(f"Закончено построение бифуркационных диаграмм для {curr_system['name']}, {params_set_chosen}, {drm_method_chosen}\n")

    axs_drm_bfd.set_title(f'Бифуркационная диаграмма (пониж. размерность, {drm_method_chosen})')
    axs_drm_time.set_title(f'Время применения метода пониж. размерности\n({drm_method_chosen}), с')

    axs_calc_time.plot(steps, calc_time, color=(0.14, 0, 0.5, 1))
    axs_drm_time.plot(steps, drm_time, color=(0.14, 0, 0.5, 1))

    axs_drm_bfd.autoscale()
    axs_drm_time.autoscale()
    axs_calc_time.autoscale()
    axs_drm_time.autoscale()

    start_calc_button.configure(state=tk.DISABLED)
    parameter_span.active = True

    canvas_crs_bfd.draw()
    toolbar_crs_bfd.update()
    canvas_drm_bfd.draw()
    toolbar_drm_bfd.update()
    canvas_calc_time.draw()
    canvas_drm_time.draw()
## -------------- ##


## Функция для кнопки show_phase_space_button. Изображает первые три измерения фазового пространства ##
## и результат применения метода понижения размерности к фазовому пространству при значениях ##
## параметра в выбранном SpanSelector диапазоне ##
def phase_trajectories_calc():
    curr_params = curr_params_set['start_params'].copy()
    curr_params[varying_param_index] = param_min
    param_limit = param_max

    step_num_max = m_ceil((param_max - param_min) / param_delta + 1)

    if step_num_max > max_plots_total:
        message = 'Слишком большой диапазон изменения параметра'
        showerror(title='Необходимо уменьшить диапазон', message=message)
        return

    fig_phase_trajectories.clear()
    fig_drm.clear()

    ncols = plots_in_row
    nrows = m_ceil(step_num_max / ncols)
    adjust_portraits(nrows)

    fig_drm.suptitle(drm_method_chosen)
    axs_phase_trajectories = fig_phase_trajectories.subplots(nrows=nrows, ncols=ncols, squeeze=False, subplot_kw={'projection': '3d', 'xticks': [], 'yticks': [], 'zticks': []})
    axs_drm = fig_drm.subplots(nrows=nrows, ncols=ncols, squeeze=False, subplot_kw={'xticks': [], 'yticks': []})
    
    canvas_phase_trajectories.draw()
    canvas_drm.draw()

    t_array = np.arange(t_cut, t_end, dt)
    step_num = 0

    drm_method.set_params(n_components=2)

    print(f"Начато изображение фазовых траекторий для {curr_system['name']}, {params_set_chosen}, {drm_method_chosen}")
    
    while curr_params[varying_param_index] - param_limit < param_delta:
        solution = scipy_solve_ivp(curr_system['func'], (t_start, t_end), initial_values, args=curr_params, t_eval=t_array)

        if solution.status != 0:
            print(f'    Проблема на шаге {step_num}: {solution.success=}')
        else:
            axs_phase_trajectories[step_num // ncols][step_num % ncols].scatter(solution.y[0], solution.y[1], solution.y[2], color=(0.14, 0, 0.5, 1), s=0.2)
            axs_phase_trajectories[step_num // ncols][step_num % ncols].set_title(f"{curr_system['param_names'][varying_param_index]} = {curr_params[varying_param_index]:.4f}")
        
            after_dim_red = drm_method.fit_transform(solution.y.T)
            axs_drm[step_num // ncols][step_num % ncols].scatter(after_dim_red[:, 0], after_dim_red[:, 1], color=(0.14, 0, 0.5, 1), s=0.2)
            axs_drm[step_num // ncols][step_num % ncols].set_title(f"{curr_system['param_names'][varying_param_index]} = {curr_params[varying_param_index]:.4f}")

        step_num += 1
        if step_num % step_info == 0:
            print(f"    Шагов выполнено: {step_num} из ~{step_num_max}")
        
        curr_params[varying_param_index] += param_delta

    print(f"Закончено изображение фазовых траекторий для {curr_system['name']}, {params_set_chosen}, {drm_method_chosen}\n")

    canvas_phase_trajectories.draw()
    toolbar_phase_trajectories.update()
    canvas_drm.draw()
    toolbar_drm.update()
## -------------- ##




## Создание пространств для графиков и объявление переменных с настройками ##
## для построения этих графиков, а также констант ##
fig_crs_bfd, axs_crs_bfd = plt.subplots()
fig_drm_bfd, axs_drm_bfd = plt.subplots()
fig_calc_time, axs_calc_time = plt.subplots()
fig_drm_time, axs_drm_time = plt.subplots()

axs_crs_bfd.set_title('Бифуркационная диаграмма')
axs_drm_bfd.set_title('Бифуркационная диаграмма (пониж. размерность)')
axs_calc_time.set_title('Время расчёта динамики, с')
axs_drm_time.set_title('Время применения метода пониж. размерности, с')

axs_calc_time.set_xlabel('Шаг')
axs_calc_time.set_ylabel('Время')
axs_drm_time.set_xlabel('Шаг')
axs_drm_time.set_ylabel('Время')

fig_phase_trajectories = Figure()
fig_drm = Figure()

epsilon = 10 ** (-2)
step_info = 2  # каждые step_info шагов в консоль выводится кол-во совершённых шагов

system_type_chosen, curr_system = None, None
params_set_chosen, params = None, None
curr_params_set, curr_params, varying_param_index, param_delta, param_limit = None, None, None, None, None
initial_values, observed_coord, return_coordinate = None, None, None
t_start, t_end, dt, t_cut = None, None, None, None
drm_method_chosen, drm_method = None, None
param_min, param_max = None, None

system_type_names = list(systems.keys())
params_set_names = []
drm_method_names = list(drm_methods.keys())
## -------------- ##


## Создание окна ##
root = tk.Tk()

width, height = 1200, 1000
root.geometry(f'{width}x{height}')
root.resizable(False, False)
## -------------- ##


## Расчёт и изменение размеров графиков ##
dpi = 100


toolbar_height = 50

bfd_width_inch = (width / 2) / dpi
bfd_height_inch = (height / 2 - toolbar_height) / dpi

tabs_height = 15

settings_weight, time_weight = 2, 4
total_weight = settings_weight + 2 * time_weight
time_width_inch = width / total_weight * time_weight / dpi
time_height_inch = height / 2 / dpi - tabs_height / dpi


plots_in_row = 10
max_plots_total = 100
plots_in_column = m_ceil(max_plots_total / plots_in_row)

side_horiz_share = 0.02
side_horiz_pix = width * side_horiz_share

plot_side_inner_share = 0.85
plot_side_share = (1 - 2 * side_horiz_share) / plots_in_row * plot_side_inner_share
plot_side_pix = width * plot_side_share

space_horiz_share = (1 - 2 * side_horiz_share) / plots_in_row * (1 - plot_side_inner_share) * plots_in_row / (plots_in_row - 1) / plot_side_share

suptitle_pix = 80
title_pix = 40

space_vert_share = title_pix / plot_side_pix

phase_trajectories_height_pix = side_horiz_pix * 2 + plot_side_pix * plots_in_column + title_pix * (plots_in_column - 1)
phase_trajectories_bottom = side_horiz_pix / phase_trajectories_height_pix
phase_trajectories_top = 1 - phase_trajectories_bottom

drm_height_pix = suptitle_pix + plot_side_pix * plots_in_column + title_pix * (plots_in_column - 1) + side_horiz_pix
drm_bottom = side_horiz_pix / drm_height_pix
drm_top = 1 - suptitle_pix / drm_height_pix


fig_crs_bfd.set_size_inches(bfd_width_inch, bfd_height_inch)
fig_drm_bfd.set_size_inches(bfd_width_inch, bfd_height_inch)

fig_calc_time.set_size_inches(time_width_inch, time_height_inch)
fig_drm_time.set_size_inches(time_width_inch, time_height_inch)

fig_phase_trajectories.set_size_inches(width / dpi, 1)
fig_phase_trajectories.subplots_adjust(left=side_horiz_share,
                                   right=1 - side_horiz_share,
                                   top=phase_trajectories_top,
                                   bottom=phase_trajectories_bottom,
                                   wspace=space_horiz_share,
                                   hspace=space_vert_share)

fig_drm.set_size_inches(width / dpi, 1)
fig_drm.subplots_adjust(left=side_horiz_share,
                        right=1 - side_horiz_share,
                        top=drm_top,
                        bottom=drm_bottom,
                        wspace=space_horiz_share,
                        hspace=space_vert_share)
## -------------- ##


## Создание содержимого окна ##

## Бифуркационная диаграма и SpanSelector для выбора значений параметра ##
root.columnconfigure(index=0, weight=1)
root.columnconfigure(index=1, weight=1)

frame_crs_bfd = ttk.Frame(borderwidth=1, relief=tk.SOLID)
canvas_crs_bfd = FigureCanvasTkAgg(fig_crs_bfd, master=frame_crs_bfd)
canvas_crs_bfd.draw()
toolbar_crs_bfd = NavigationToolbar2Tk(canvas_crs_bfd, frame_crs_bfd, pack_toolbar=False)
toolbar_crs_bfd.update()

frame_crs_bfd.grid(row=0, column=0)
toolbar_crs_bfd.pack(side=tk.TOP, fill=tk.X)
canvas_crs_bfd.get_tk_widget().pack(expand=True, fill=tk.BOTH)

parameter_span = SpanSelector(canvas_crs_bfd.figure.axes[0],
                              get_param_limits,
                              'horizontal',
                              useblit=True,
                              props={'facecolor': (0.5, 0.5, 1), 'alpha': 0.4},
                              interactive=True,
                              drag_from_anywhere=True)

parameter_span.active = False

## Бифуркационная диаграмма, построенная с помощью метода понижения размерности ##
frame_drm_bfd = ttk.Frame(borderwidth=1, relief=tk.SOLID)
canvas_drm_bfd = FigureCanvasTkAgg(fig_drm_bfd, master=frame_drm_bfd)
canvas_drm_bfd.draw()
toolbar_drm_bfd = NavigationToolbar2Tk(canvas_drm_bfd, frame_drm_bfd, pack_toolbar=False)
toolbar_drm_bfd.update()

frame_drm_bfd.grid(row=0, column=1)
toolbar_drm_bfd.pack(side=tk.TOP, fill=tk.X)
canvas_drm_bfd.get_tk_widget().pack(expand=True, fill=tk.BOTH)

## Frame для вкладок и сами вкладки ##
frame_tabs = ttk.Frame(borderwidth=1, relief=tk.SOLID)
frame_tabs.grid(row=1, column=0, columnspan=2)

notebook = ttk.Notebook(frame_tabs)
notebook.pack(expand=True, fill=tk.BOTH)

## Вкладка с настройками и графиками времени вычислений ##
tab_controls = ttk.Frame(notebook)
tab_controls.columnconfigure(index=0, weight=settings_weight)
tab_controls.columnconfigure(index=1, weight=time_weight)
tab_controls.columnconfigure(index=2, weight=time_weight)
tab_controls.pack(expand=True, fill=tk.BOTH)
notebook.add(tab_controls, text='Настройки')

subspace_controls = ttk.Frame(tab_controls)
subspace_controls.grid(row=0, column=0)

program_label = ttk.Label(subspace_controls, text='Программа!')
system_label = ttk.Label(subspace_controls, text='Система:')
params_label = ttk.Label(subspace_controls, text='Набор параметров:')
drm_label = ttk.Label(subspace_controls, text='Метод понижения размерности:')

system_type_name = tk.StringVar()
params_set_name = tk.StringVar()
drm_method_name = tk.StringVar()
system_type_combobox = ttk.Combobox(subspace_controls, textvariable=system_type_name, values=system_type_names, state='readonly')
params_set_combobox = ttk.Combobox(subspace_controls, textvariable=params_set_name, values=params_set_names, state='readonly')
drm_method_combobox = ttk.Combobox(subspace_controls, textvariable=drm_method_name, values=drm_method_names, state='readonly')

start_calc_button = ttk.Button(subspace_controls, text='Изобразить бифуркационные диаграммы', command=do_calculations, state='disabled')
show_phase_space_button = ttk.Button(subspace_controls, text='Изобразить фазовые траектории', command=phase_trajectories_calc, state='disabled')
clear_button = ttk.Button(subspace_controls, text='Сбросить всё', command=clear_calcs)

program_label.pack(anchor='nw')
system_label.pack(anchor='w')
system_type_combobox.pack(anchor='e')
system_type_combobox.bind('<<ComboboxSelected>>', choose_system)
params_label.pack(anchor='w')
params_set_combobox.pack(anchor='e')
params_set_combobox.bind('<<ComboboxSelected>>', choose_params_set)
drm_label.pack(anchor='w')
drm_method_combobox.pack(anchor='e')
drm_method_combobox.bind('<<ComboboxSelected>>', choose_drm_method)
start_calc_button.pack()
show_phase_space_button.pack()
clear_button.pack()

## Графики с временем вычислений для построения бифуркационных диаграмм ##
subspace_calc_time = ttk.Frame(tab_controls)
canvas_calc_time = FigureCanvasTkAgg(fig_calc_time, master=subspace_calc_time)
canvas_calc_time.draw()
canvas_calc_time.get_tk_widget().pack(expand=True, fill=tk.BOTH)
subspace_calc_time.grid(row=0, column=1)

subspace_drm_time = ttk.Frame(tab_controls)
canvas_drm_time = FigureCanvasTkAgg(fig_drm_time, master=subspace_drm_time)
canvas_drm_time.draw()
canvas_drm_time.get_tk_widget().pack(expand=True, fill=tk.BOTH)
subspace_drm_time.grid(row=0, column=2)

## Вкладка с фазовыми портретами системы при разных значениях параметра ##
## из диапазона, выбранного SpanSelector ##
tab_phase_trajectories = ttk.Frame(notebook)
tab_phase_trajectories.pack(expand=True, fill=tk.BOTH)
notebook.add(tab_phase_trajectories, text='Фазовые траектории')

canvas_phase_trajectories = FigureCanvasTkAgg(fig_phase_trajectories, master=tab_phase_trajectories)
canvas_phase_trajectories.draw()
toolbar_phase_trajectories = NavigationToolbar2Tk(canvas_phase_trajectories, tab_phase_trajectories, pack_toolbar=False)
toolbar_phase_trajectories.update()

toolbar_phase_trajectories.pack(side=tk.TOP, fill=tk.X)
canvas_phase_trajectories.get_tk_widget().pack(expand=True, fill=tk.BOTH)

## Вкладка с фазовыми портретами системы после понижения размерности ##
## при разных значениях параметра из диапазона, выбранного SpanSelector ##
tab_drm = ttk.Frame(notebook)
tab_drm.pack(expand=True, fill=tk.BOTH)
notebook.add(tab_drm, text='Фазовые траектории (пониж. размерность)')

canvas_drm = FigureCanvasTkAgg(fig_drm, master=tab_drm)
canvas_drm.draw()
toolbar_drm = NavigationToolbar2Tk(canvas_drm, tab_drm, pack_toolbar=False)
toolbar_drm.update()

toolbar_drm.pack(side=tk.TOP, fill=tk.X)
canvas_drm.get_tk_widget().pack(expand=True, fill=tk.BOTH)


## Начало работы приложения ##
root.mainloop()
## -------------- ##
