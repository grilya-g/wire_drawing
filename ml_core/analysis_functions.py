import logging
import os
# np.set_printoptions(threshold=1000)
##########################
# SAVE-LOAD using pickle #
##########################
import pickle
from csv import writer
from datetime import datetime
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np

# from numba import njit
# from numba import jit
import optuna
import pandas as pd

# import xlrd
import sklearn
import torch
from bayes_opt import BayesianOptimization
from kan import KAN
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import ShuffleSplit, train_test_split

# from numba import prange
# import plotly
# plotly.offline.init_notebook_mode(connected=True)
# import plotly.graph_objects as go
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import ignore_warnings
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
from skopt.utils import use_named_args

RANDOM_STATE = 50


# save
def saver(model, model_name, path_import):
    with open(path_import + "/" + model_name + ".pkl", "wb") as f:
        pickle.dump(model, f)
        print("saved in " + str(f.name))


# load
def opener(model_name, path_import):
    name = str(model_name)
    name = path_import + "/" + name + ".pkl"
    print(name)
    with open(name, "rb") as f:
        model = pickle.load(f)
    return model


def average_in_list(list):
    """
    Функция подсчета среднего значения в списке
    param: list -- список для поиска среднего

    return summ / len(list) -- среднее значение в списке
    """
    summ = 0
    for i in list:
        summ += i
    return summ / len(list)


def read_data_txt_np(name_of_file, example_dir, is_nonisothermal=False):
    """
    Функция для считывания данных из отчетных файлов ПО Abaqus
    params:
    name_of_file -- имя файла
    example_dir -- директория файла

    return:
    data -- считанные данные
    """
    content = os.listdir(example_dir)
    if os.path.isfile(os.path.join(example_dir, name_of_file)):
        if name_of_file in content:
            file = os.path.join(example_dir, name_of_file)
            if file.endswith(".txt"):
                if is_nonisothermal:
                    data = np.loadtxt(file, dtype="double", skiprows=56)
                else:
                    data = np.loadtxt(file, dtype="double", skiprows=44)
            else:
                print(f"{file} doesnt ends with .txt")
        else:
            print(f"No file {name_of_file} in such dir")
    else:
        print(f"No such file {name_of_file}")
        return -1
    return data


def files_list(list_one, dir_):
    """
    Функция для считывания данных для переданного списка
    params:
    list_one -- список файлов
    dir_ -- директория файлов

    return:
    list_files -- список считанных данных
    """
    list_files = []
    for i in range(0, len(list_one)):
        file = read_data_txt_np(list_one[i], dir_)
        list_files.append(file)

    return list_files


def median_np(x):
    """Построение медианы по переданному списку"""
    return np.median(x)


def preprocessing_res_np(file):
    """
    Создание np.array по репорту
    params:
    file -- считааные из `read_data_txt_np` данные

    return:
    x_1_    -- массив х координат узлов решетки
    y_1_    -- массив у координат узлов решетки
    Stress  -- массив массивов компонент тензора напряжений и интенсивностей напряжений
    Strain  -- массив массивов компонент тензора деформаций и интенсивностей деформаций
    """
    x_1 = (file[::, 1]) * 10**3
    y_1 = (file[::, 2]) * 10**3

    eps_11_name = file[::, 11]
    eps_22_name = file[::, 12]
    eps_33_name = file[::, 13]
    eps_12_name = file[::, 14]

    stress_11_name = (file[::, 3]) / 10**6
    stress_22_name = (file[::, 4]) / 10**6
    stress_33_name = (file[::, 5]) / 10**6
    stress_12_name = (file[::, 6]) / 10**6

    # nt11_name = (file[::,15])
    # temp_name = (file[::,16])

    index = 0

    x_1_ = np.zeros((1, 1))
    y_1_ = np.zeros((1, 1))
    Strain = np.zeros((1, 5))
    Stress = np.zeros((1, 5))
    # NT11 = np.zeros((1, 1))
    # TEMP = np.zeros((1, 1))
    # ind_ = np.zeros((1,1))
    for j in range(0, np.shape(y_1)[0]):
        if y_1[j] >= 0.25 * (np.max(y_1) - np.min(y_1)) and y_1[j] <= 0.75 * (
            np.max(y_1) - np.min(y_1)
        ):
            x_1_[index] = x_1[j]
            y_1_[index] = y_1[j]
            # ind_[index] = j

            Stress[index, 0] = stress_11_name[j]
            Stress[index, 1] = stress_22_name[j]
            Stress[index, 2] = stress_33_name[j]
            Stress[index, 3] = stress_12_name[j]
            Stress[index, 4] = (1 / (np.sqrt(2))) * np.sqrt(
                (stress_11_name[j] - stress_22_name[j]) ** 2
                + (stress_22_name[j] - stress_33_name[j]) ** 2
                + (stress_33_name[j] - stress_11_name[j]) ** 2
                + 6 * (stress_12_name[j] ** 2)
            )

            Strain[index, 0] = eps_11_name[j]
            Strain[index, 1] = eps_22_name[j]
            Strain[index, 2] = eps_33_name[j]
            Strain[index, 3] = eps_12_name[j]
            Strain[index, 4] = (1 / (np.sqrt(2))) * np.sqrt(
                (eps_11_name[j] - eps_22_name[j]) ** 2
                + (eps_22_name[j] - eps_33_name[j]) ** 2
                + (eps_33_name[j] - eps_11_name[j]) ** 2
                + 6 * (eps_12_name[j] ** 2)
            )

            # NT11[index]=nt11_name[j]
            # TEMP[index]=temp_name[j]

            x_1_ = np.concatenate((x_1_, np.zeros((1, 1))))
            y_1_ = np.concatenate((y_1_, np.zeros((1, 1))))
            Stress = np.concatenate((Stress, np.zeros((1, 5))))
            Strain = np.concatenate((Strain, np.zeros((1, 5))))
            # NT11 = np.concatenate((NT11,np.zeros((1,1))))
            # TEMP = np.concatenate((TEMP,np.zeros((1,1))))
            index = index + 1

    x_1_ = np.delete(x_1_, -1)
    y_1_ = np.delete(y_1_, -1)
    Stress = np.delete(Stress, -1, axis=0)
    Strain = np.delete(Strain, -1, axis=0)

    x_1_, characteristics = sorter(x_1_, [y_1_, Stress, Strain])
    y_1_, Stress, Strain = characteristics[0], characteristics[1], characteristics[2]

    return x_1_, y_1_, Stress, Strain


def average_val_np(array_1, nodes=13):
    """Построение осредненных распределений"""
    b = np.zeros((nodes))
    for i in range(0, nodes):
        b[i] = np.median(np.array_split(array_1, nodes)[i])

    return b


def get_param(cur_job_name, vel20List, all20Arrays, char_1=2, char_2=4):
    """
    Функция для извлечения зависимых и независимых переменных для 1 расчета
    params:
    cur_job_name -- название расчетного файла до точки
    vel20List -- список всех имен расчетных файов
    all20Arrays -- список предобработанных данных
    char_1 -- индекс характеристики (2-напряжения, 3-деформации)
    char_2 -- индекс компоненты тензора, 4 - интенсивность

    return:
    [red, cal, ha, vel, fric, char] -- список требуемых характеристик
    (обжатие, коэффициент калибровочного участка, полуугол волоки, скорость, трение и искомая характеристика соотвественно)
    """
    int_res_stress_idx = vel20List.index(cur_job_name)

    char = average_val_np(all20Arrays[int_res_stress_idx][char_1][::, char_2], 20)

    red = r(cur_job_name)
    cal = c(cur_job_name)
    ha = h(cur_job_name)
    vel = v(cur_job_name)
    fric = f(cur_job_name)

    return [red, cal, ha, vel, fric, char]


def sorter(xx, heap_of_stresses):
    """
    Сортирует по возрастанию узлы по значению расстояния от оси проволоки и
        перегруппировывает значения в остальных массивах в соответствии с новыми
        значениями индексов, но больше не удаляет точку (0,0)
    :param xx: значения координат х
    :param heap_of_stresses: список характеристик
    :return: x_sorted, heap_of_stresses_sorted
    """

    heap_of_stresses_sorted = heap_of_stresses[:]
    ai = np.argsort(xx, axis=0)
    x_sorted = np.take_along_axis(xx, ai, axis=0)
    ai = ai.flatten()

    for idx, i in enumerate(heap_of_stresses):
        ai_i = np.copy(ai)
        if i.ndim > 1:  # alternatively use np.tile instead
            for j in range(i.shape[1] - 1):
                ai_i = np.vstack((ai_i, ai))
            ai_i = ai_i.T
        heap_of_stresses_sorted[idx] = np.take_along_axis(i, ai_i, axis=0)
        # heap_of_stresses_sorted[idx] = np.delete(heap_of_stresses_sorted[idx], delete_idx)

    return x_sorted, heap_of_stresses_sorted


def r(job_name):
    """
    Ищет обжатие по переданному имени файла
    """
    splitted_job = job_name.split("_")
    for i in range(len(splitted_job)):
        if splitted_job[i] == "red":
            found = splitted_job[i + 1]
    reduct = float(found) / 10000
    return reduct


def c(job_name):
    """
    Ищет коэффициент калибровочного участка по переданному имени файла
    """
    splitted_job = job_name.split("_")
    for i in range(len(splitted_job)):
        if splitted_job[i] == "cal":
            found = splitted_job[i + 1]
    cali = float(found) / 100
    return cali


def f(job_name):
    """
    Ищет к-т трения по переданному имени файла
    """
    splitted_job = job_name.split("_")
    for i in range(len(splitted_job)):
        if splitted_job[i] == "fric":
            found = splitted_job[i + 1][1:]
    fr = float(found) / 1000
    return fr


def v(job_name):
    """
    Ищет скорость по переданному имени файла
    """
    splitted_job = job_name.split("_")
    for i in range(len(splitted_job)):
        if splitted_job[i] == "vel":
            found = splitted_job[i + 1]
    velo = int(found)
    return velo


def h(job_name):
    """
    Ищет полуугол по переданному имени файла
    """
    splitted_job = job_name.split("_")
    for i in range(len(splitted_job)):
        if splitted_job[i] == "2a":
            found = splitted_job[i + 1]
    ha = int(found) / 2
    return ha


# @title Функция для подсчета максимальной ошибки для графиков
def max_er(pred, ground_true):
    maxerror = 0
    for i in range(0, len(pred)):
        cmape = (ground_true[i] - pred[i]) / ground_true[i] * 100
        maxerror = cmape if cmape > maxerror else maxerror

    return maxerror


# @title Функция для подсчета максимальной средней ошибки для графиков
def max_graph_er(pred, ground_true):
    maxerror = 0
    for i in range(0, len(pred) // 20):
        cmape = (
            sklearn.metrics.mean_absolute_percentage_error(
                ground_true[i * 20 : (i + 1) * 20], pred[i * 20 : (i + 1) * 20]
            )
            * 100
        )
        maxerror = cmape if cmape > maxerror else maxerror

    return maxerror


# @title Функция для подсчета медианной средней ошибки для графиков
def med_graph_er(pred, ground_true):
    mapes = np.zeros((len(pred) // 20))
    for i in range(0, len(pred) // 20):
        cmape = (
            sklearn.metrics.mean_absolute_percentage_error(
                ground_true[i * 20 : (i + 1) * 20], pred[i * 20 : (i + 1) * 20]
            )
            * 100
        )
        mapes[i] = cmape
    med = min(mapes, key=lambda x: abs(x - np.median(mapes)))
    return med


# @title Функция для подсчета минимальной средней ошибки для графиков
def min_graph_er(pred, ground_true):
    minerror = 1e10
    for i in range(0, len(pred) // 20):
        cmape = (
            sklearn.metrics.mean_absolute_percentage_error(
                ground_true[i * 20 : (i + 1) * 20], pred[i * 20 : (i + 1) * 20]
            )
            * 100
        )
        minerror = cmape if cmape < minerror else minerror

    return minerror


def do_rep_list(path_import):
    rep_list = [
        fil[:-4]
        for fil in listdir(path_import)
        if isfile(join(path_import, fil)) and fil.endswith(".txt")
    ]
    return rep_list


def do_preprocessing(list_names, path_import):
    vel20ListFiles = []
    for fi in list_names:
        vel20ListFiles.append(fi + ".txt")
    la = files_list(vel20ListFiles, path_import)
    all20Arrays = []
    for fl in la:
        all20Arrays.append(preprocessing_res_np(fl))
        print(".", end="")
    print("")
    return all20Arrays


def data_preparer(train_list, train_arrays):
    """
    Преобразует массивы из preprocessing_np
    в массивы для разделения на трейн и тест.

    Выделяет компоненты тензора в таргет и параметры процесса в инпут.

    """
    X_stress_components = []
    X_strain_components = []
    X_components = [X_stress_components, X_strain_components]

    y_stress_components = []
    y_strain_components = []
    y_components = [y_stress_components, y_strain_components]

    for char in range(2):
        for comp in range(3):
            X = np.zeros((len(train_list), 5))
            y = np.zeros((len(train_list), 20))

            for i, job_name in enumerate(train_list):
                red, cal, ha, vel, fric, value = get_param(
                    job_name, train_list, train_arrays, char_1=2 + char, char_2=comp
                )

                X[i] = np.array([red, cal, ha, fric, vel])
                y[i] = value
            X_components[char].append(X)
            y_components[char].append(y)

    X_stress_components = np.array(X_stress_components)
    X_strain_components = np.array(X_strain_components)

    y_stress_components = np.array(y_stress_components)
    y_strain_components = np.array(y_strain_components)

    return (
        X_stress_components,
        X_strain_components,
        y_stress_components,
        y_strain_components,
    )


def plot_result_stress_avr_v(list_files, labels, name, **kwargs):
    """Строит графики для минимальных, максимальных и средних деформаций для переданных списков в вертикальной ориентации"""
    plt.figure(figsize=(18, 10))
    global colours_
    global markers_
    # f = plt.figure(figsize=(9,6))

    # preped_arrays = [preprocessing_res_np(item) for item in list_files]
    preped_arrays = list_files

    for i in range(0, len(list_files)):
        x_1 = average_val_np(preped_arrays[i][0])
        R1 = x_1.max()

        for j in range(1, 4):
            plt.subplot(1, 3, j)
            # plt.subplots_adjust(left=0.0, right=0.5, top=0.7, bottom=0.0)

            avr_strain = average_val_np(preped_arrays[i][2][::, j - 1])
            #             min_strain = min_val_np(preprocessing_res_np(list_files[i])[3][::,j-1])
            #             max_strain =  max_val_np(preprocessing_res_np(list_files[i])[3][::,j-1])

            plt.plot(x_1 / R1, avr_strain, markersize=9, label=labels[i])
            #             plt.plot(x_1/R1, min_strain, '-*', markersize=9, color = colours_[i])
            #             plt.plot(x_1/R1, max_strain, '-.', markersize=9, color = colours_[i])

            if j == 1:
                plt.ylabel(
                    "$ \sigma_{r}$, -",
                    fontsize=20,
                )

            elif j == 2:
                plt.ylabel(
                    "$ \sigma_{z}$, -",
                    fontsize=20,
                )

            elif j == 3:
                plt.ylabel(
                    "$ \sigma_{\phi}$, -",
                    fontsize=20,
                )

            # elif(i==4):
            #   plt.ylabel('$\sigma_{rz}$, MPa ', fontsize=22)

            plt.xlim(0, 1.01)
            plt.tick_params(axis="x", length=5, labelsize=10, zorder=15)
            plt.tick_params(axis="y", length=5, labelsize=10, zorder=15)
            plt.xticks(np.arange(0, 1.02, 0.2))

            plt.xlabel(
                " r, -",
                fontsize=20,
            )
            # plt.grid(True)

    if kwargs.get("to_plot_legend", False):
        plt.legend(loc="best")
    to_save = kwargs.get("to_save", False)
    to_show = kwargs.get("to_show", True)
    plt.subplots_adjust(wspace=0.333, hspace=0)
    dpi = kwargs.get("dpi", 600)
    if to_save:
        plt.savefig(name + "avr_stress_v_" + ".png", dpi=dpi)
    if to_show:
        plt.show()


def flatten_r(X_train_fair, y_train_fair):
    # ic(y_train_fair.shape)
    X_1 = np.zeros((len(X_train_fair) * 20, X_train_fair.shape[1] + 1))
    y_1 = np.zeros((len(X_train_fair) * 20,))
    for i in range(len(X_train_fair)):
        for j in range(20):
            X_1[i * 20 + j] = np.hstack((X_train_fair[i], j / 19))
            y_1[i * 20 + j] = y_train_fair[i][j]
    return X_1, y_1


def split_transform(X, y):
    splitted_X, splitted_y = [], []
    for comp in range(3):
        cur_X, cur_y = X[comp], y[comp]
        cur_X_train, cur_X_test, cur_y_train, cur_y_test = train_test_split(
            cur_X, cur_y, train_size=0.7, random_state=RANDOM_STATE
        )
        cur_X_train, cur_y_train = flatten_r(cur_X_train, cur_y_train)
        cur_X_val, cur_X_test, cur_y_val, cur_y_test = train_test_split(
            cur_X_test, cur_y_test, train_size=0.5, random_state=RANDOM_STATE
        )
        cur_X_test, cur_y_test = flatten_r(cur_X_test, cur_y_test)
        cur_X_val, cur_y_val = flatten_r(cur_X_val, cur_y_val)
        splitted_X.append([cur_X_train, cur_X_val, cur_X_test])
        splitted_y.append([cur_y_train, cur_y_val, cur_y_test])
    return splitted_X, splitted_y


def split_transform_one_comp(X, y):
    splitted_X, splitted_y = [], []
    cur_X, cur_y = X, y
    cur_X_train, cur_X_test, cur_y_train, cur_y_test = train_test_split(
        cur_X, cur_y, train_size=0.7, random_state=RANDOM_STATE
    )
    cur_X_train, cur_y_train = flatten_r(cur_X_train, cur_y_train)
    cur_X_val, cur_X_test, cur_y_val, cur_y_test = train_test_split(
        cur_X_test, cur_y_test, train_size=0.5, random_state=RANDOM_STATE
    )
    cur_X_test, cur_y_test = flatten_r(cur_X_test, cur_y_test)
    cur_X_val, cur_y_val = flatten_r(cur_X_val, cur_y_val)
    splitted_X.append([cur_X_train, cur_X_val, cur_X_test])
    splitted_y.append([cur_y_train, cur_y_val, cur_y_test])
    return splitted_X[0], splitted_y[0]


def split_transform_one_comp_train_test(X, y):
    cur_X_train, cur_X_test, cur_y_train, cur_y_test = train_test_split(
        X, y, train_size=0.7, random_state=RANDOM_STATE
    )
    cur_X_train, cur_y_train = flatten_r(cur_X_train, cur_y_train)
    cur_X_test, cur_y_test = flatten_r(cur_X_test, cur_y_test)

    return cur_X_train, cur_X_test, cur_y_train, cur_y_test


def scorer(y_true, y_pred, pipeline, X_train):
    evs = explained_variance_score(y_true, y_pred)  # 1-- BEST , 0 -- WORST
    medae = median_absolute_error(y_true, y_pred)  # 0 -- BEST, \INF -- WORST
    mse = mean_squared_error(y_true, y_pred)  # 0 -- BEST
    mae = mean_absolute_error(y_true, y_pred)  # 0 -- BEST
    r2 = r2_score(y_true, y_pred)  # 1 --BEST
    me = max_error(y_true, y_pred)  # 0-- BEST
    rmse = root_mean_squared_error(y_true, y_pred)  # 0 -- BEST

    if isinstance(
        pipeline,
        (
            MLPRegressor,
            sklearn.tree.DecisionTreeRegressor,
            sklearn.ensemble.GradientBoostingRegressor,
        ),
    ):
        model = pipeline
    else:
        model = (
            pipeline.named_steps.get("mlpregressor")
            or pipeline.named_steps.get("decisiontree")
            or pipeline.named_steps.get("gradientboosting")
        )

    if isinstance(model, MLPRegressor):
        n_params = model.coefs_[0].size + model.coefs_[1].size
    elif isinstance(model, sklearn.tree.DecisionTreeRegressor):
        n_params = model.tree_.node_count
    elif isinstance(model, sklearn.ensemble.GradientBoostingRegressor):
        n_params = sum(
            estimator.tree_.node_count for estimator in model.estimators_.ravel()
        )
    else:
        raise ValueError("Unsupported model type")

    n = len(X_train)  # number of samples
    aic = n * np.log(mse) + 2 * n_params  # 0 -- BEST
    bic = n * np.log(mse) + n_params * np.log(n)  # 0 -- BEST

    errors = np.array([evs, medae, mse, mae, r2, me, aic, bic, rmse])

    return errors


def choose_worst(val_metrics):
    worst = np.zeros(9)
    worst[0] = val_metrics[:, 0].min()
    worst[1] = val_metrics[:, 1].max()
    worst[2] = val_metrics[:, 2].max()
    worst[3] = val_metrics[:, 3].max()
    worst[4] = val_metrics[:, 4].min()
    worst[5] = val_metrics[:, 5].max()
    worst[6] = val_metrics[:, 6].max()
    worst[7] = val_metrics[:, 7].max()
    worst[8] = val_metrics[:, 8].max()

    return worst


def split_transform_one_comp_cv(X, y, n_splits=5):
    val_set_X, val_set_y, train_set_X, train_set_y = [], [], [], []
    # Split to get test set
    cur_X_train, cur_X_test, cur_y_train, cur_y_test = train_test_split(
        X, y, train_size=0.85, random_state=RANDOM_STATE
    )
    # Format val set
    cur_X_test, cur_y_test = flatten_r(cur_X_test, cur_y_test)
    # Shuffle to get test and val sets
    ss = ShuffleSplit(
        n_splits=n_splits, test_size=0.15 / (0.7 + 0.15), random_state=RANDOM_STATE
    )
    for i, (train_index, test_index) in enumerate(ss.split(cur_X_train)):
        # Get train and val sets on iteration
        cur_X_val_splitted = cur_X_train[test_index]
        cur_X_train_splitted = cur_X_train[train_index]
        cur_y_val_splitted = cur_y_train[test_index]
        cur_y_train_splitted = cur_y_train[train_index]
        # Formatting
        cur_X_val_formatted, cur_y_val_formatted = flatten_r(
            cur_X_val_splitted, cur_y_val_splitted
        )
        cur_X_train_formatted, cur_y_train_formatted = flatten_r(
            cur_X_train_splitted, cur_y_train_splitted
        )
        # Saving results
        val_set_X.append(cur_X_val_formatted)
        val_set_y.append(cur_y_val_formatted)
        train_set_X.append(cur_X_train_formatted)
        train_set_y.append(cur_y_train_formatted)

    return cur_X_test, cur_y_test, val_set_X, val_set_y, train_set_X, train_set_y


def get_train_test(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.85, random_state=RANDOM_STATE
    )
    x_train, y_train = flatten_r(x_train, y_train)
    x_test, y_test = flatten_r(x_test, y_test)
    return x_test, y_test, x_train, y_train


def append_list_as_row(file_name, list_of_elem, path=""):
    file_name_path = path + file_name
    # Open file in append mode
    with open(file_name_path, "a+", newline="") as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


def taught_checker(path_import, fname, i, j):
    return (
        pd.read_csv(
            path_import + fname,
            names=[
                # Constructional
                "n_layers",
                "n_neurons",
                "solver",
                "max_iter",
                "learning_rate_init",
                "learning_rate",
                "early_stopping",
                "activation",
                "n_splits",
                "alpha",
                # Resultant _val
                "explained_variance_score_val",
                "median_absolute_error_val",
                "mean_squared_error_val",
                "mean_absolute_error_val",
                "r2_score_val",
                "max_error_val",
                "AIC_val",
                "BIC_val",
                # Resultant _test
                "explained_variance_score_test",
                "median_absolute_error_test",
                "mean_squared_error_test",
                "mean_absolute_error_test",
                "r2_score_test",
                "max_error_test",
                "AIC_test",
                "BIC_test",
            ],
        )[["n_layers", "n_neurons"]]
        .query(f"(n_layers == {i}) & (n_neurons == {j})")
        .empty
    )


# @title ANN
@ignore_warnings(category=ConvergenceWarning)
def special_ann_stress_strains_val(
    fname,
    range_layers,
    range_neurons,
    X,
    y,
    alpha=0.0001,
    solver="adam",
    learning_rate="constant",
    learning_rate_init=0.001,
    activation="relu",
    early_stopping=True,
    max_iter=200,
    n_splits=5,
    path_import="",
):
    # I've got an idea to perform a Cross-Validation
    # so i want to build k train-val-test sets once here
    # and use them during the fitting later

    today = datetime.now()

    # Preparing datasets
    cur_X_test, cur_y_test, val_list_X, val_list_y, train_list_X, train_list_y = (
        split_transform_one_comp_cv(X, y)
    )

    # Start of a outer cycle
    for j in range_neurons:
        # Start of an inner cycle
        for i in range_layers:
            # check if already was taught
            if not taught_checker(path_import, fname, i, j):
                print(f"{i} neurons and {j} layers was in df")
                return None

            # hidden layers construction
            hls_tuple = ()
            for _ in range(1, i + 1):
                hls_tuple = hls_tuple + (j,)

            #############################################################
            #                                                           #
            #                 LEARNING AND VALIDATION                   #
            #                                                           #
            #############################################################

            # Making a model sceleton
            regr = make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=hls_tuple,
                    learning_rate_init=learning_rate_init,
                    learning_rate=learning_rate,
                    activation=activation,
                    random_state=RANDOM_STATE,
                    #  alpha=alpha,
                    early_stopping=True,
                    solver=solver,
                    # verbose=True,
                    max_iter=max_iter,
                ),
            )

            # Fitting and scoring `n_split` times
            errors = np.zeros((n_splits, 8))

            for split_idx in range(n_splits):
                cur_X_train = train_list_X[split_idx]
                cur_y_train = train_list_y[split_idx]

                cur_X_val = val_list_X[split_idx]
                cur_y_val = val_list_y[split_idx]

                regr.fit(cur_X_train, cur_y_train)

                #######  Validation ########
                #  Prediction
                cur_prediction = regr.predict(cur_X_val)
                # Scoring
                errors[split_idx] = scorer(cur_y_val, cur_prediction, regr, cur_X_train)

            # Collect validation result
            val_metrics = choose_worst(errors)

            ########################################################
            #                                                      #
            #                        TESTING                       #
            #                                                      #
            ########################################################

            # Scoring on the test set
            test_prediction = regr.predict(cur_X_test)
            test_metrics = scorer(cur_y_test, test_prediction, regr, train_list_X[0])

            ########################################################
            #                                                      #
            #              SHOWING AND SAVING RESULTS              #
            #                                                      #
            ########################################################

            new_row = (
                [
                    i,
                    j,
                    solver,
                    max_iter,
                    learning_rate_init,
                    learning_rate,
                    early_stopping,
                    activation,
                    n_splits,
                    alpha,
                ]
                + list(val_metrics)
                + list(test_metrics)
            )
            append_list_as_row(fname, new_row)

            print(
                f"{solver} for {i:>3} hidden layers of size {j:>3}:"
                f"\ttime:{datetime.now()}, taken:{datetime.now() - today}\n"
                "explained_variance_score, median_absolute_error,"
                "mean_squared_error, mean_absolute_error,"
                "r2_score, max_error, AIC, BIC\n"
                "val metrics are"
                f"\n {list(val_metrics)}\n"
                f"test metrics are\n {list(test_metrics)}\n"
            )

    return regr, cur_X_test, cur_y_test


def clean_input_array(x, y):
    mask = ~(
        np.isin(x, [np.nan, np.inf, -np.inf]).any(axis=1)
        | np.isin(y, [np.nan, np.inf, -np.inf])
    )
    x = x[mask].round(15)
    y = y[mask].round(15)
    return x, y


def do_optuna(X, y, n_trials=100, **kwargs):
    n_splits = kwargs.get("n_splits", 3)
    n_layers = kwargs.get("n_layers", 30)
    n_neurons = kwargs.get("n_neurons", 100)

    # Preparing datasets
    cur_X_test, cur_y_test, val_list_X, val_list_y, train_list_X, train_list_y = (
        split_transform_one_comp_cv(X, y, n_splits=n_splits)
    )

    def optuna_ann_stress_strains_val(trial):
        k_layers = trial.suggest_int("n_layers", 1, n_layers)
        layers = []
        for i in range(k_layers):
            layers.append(trial.suggest_int(f"n_units_{i}", 1, n_neurons))

        params = {
            "hidden_layer_sizes": tuple(layers),
            "learning_rate_init": trial.suggest_float(
                "learning_rate_init", 1e-6, 0.1, log=True
            ),
            "random_state": 100,
            "early_stopping": trial.suggest_categorical(
                "early_stopping", [True, False]
            ),
            "max_iter": trial.suggest_int("max_iter", 10, 15000),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", ["constant", "invscaling", "adaptive"]
            ),
            "alpha": trial.suggest_categorical(
                "alpha", [0.3, 0.1, 0.01, 0.001, 0.0001]
            ),
            "activation": trial.suggest_categorical(
                "activation", ["logistic", "relu", "tanh"]
            ),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "adam", "sgd"]),
        }

        # Fitting and scoring `n_split` times
        errors = np.zeros((n_splits, 9))

        for split_idx in range(n_splits):
            regr = make_pipeline(
                StandardScaler(),
                MLPRegressor(**params),
            )

            cur_X_train = train_list_X[split_idx]
            cur_y_train = train_list_y[split_idx]

            cur_X_val = val_list_X[split_idx]
            cur_y_val = val_list_y[split_idx]

            cur_X_train, cur_y_train = clean_input_array(cur_X_train, cur_y_train)
            cur_X_val, cur_y_val = clean_input_array(cur_X_val, cur_y_val)

            regr.fit(cur_X_train, cur_y_train)

            #######  Validation  ########
            #  Prediction
            cur_prediction = regr.predict(cur_X_val)
            # Scoring
            errors[split_idx] = scorer(cur_y_val, cur_prediction, regr, cur_X_train)

        # Collect validation result
        val_metrics = choose_worst(errors)
        # return_value = val_metrics[0] if pd.notnull(val_metrics[0]) else -1e6 # для evs
        return_value = (
            val_metrics[-1] if pd.notnull(val_metrics[-1]) else +1e6
        )  # для rmse
        return return_value

    # Create a study object to optimize the objective
    # study = optuna.create_study(direction='maximize') # evs
    study = optuna.create_study(direction="minimize")  # rmse
    study.optimize(optuna_ann_stress_strains_val, n_trials=n_trials, n_jobs=-1)

    # Print the best hyperparameters found by Optuna
    best_params = study.best_params
    best_value = study.best_value
    print("Best Hyperparameters:", best_params)

    return best_params, cur_X_test, cur_y_test, best_value


@ignore_warnings(category=ConvergenceWarning)
def do_skopt(X, y, n_calls=100, **kwargs):
    n_splits = kwargs.get("n_splits", 3)
    n_layers = kwargs.get("n_layers", 30)
    n_neurons = kwargs.get("n_neurons", 100)

    # Preparing datasets
    cur_X_test, cur_y_test, val_list_X, val_list_y, train_list_X, train_list_y = (
        split_transform_one_comp_cv(X, y, n_splits=n_splits)
    )

    # Define the search space
    space = [
        Integer(1, n_layers, name="n_layers"),
        Integer(1, n_neurons, name="n_units_0"),
        Real(1e-6, 0.1, "log-uniform", name="learning_rate_init"),
        Categorical([True, False], name="early_stopping"),
        Integer(10, 15000, name="max_iter"),
        Categorical(["constant", "invscaling", "adaptive"], name="learning_rate"),
        Categorical([0.3, 0.1, 0.01, 0.001, 0.0001], name="alpha"),
        Categorical(["logistic", "relu", "tanh"], name="activation"),
        Categorical(["lbfgs", "adam", "sgd"], name="solver"),
    ]

    @use_named_args(space)
    def objective(**params):
        k_layers = params.pop("n_layers")
        layers = [params.pop(f"n_units_{i}", 1) for i in range(k_layers)]

        params["hidden_layer_sizes"] = tuple(layers)
        params["random_state"] = 100

        # Fitting and scoring `n_split` times
        errors = np.zeros((n_splits, 9))

        for split_idx in range(n_splits):
            regr = make_pipeline(
                StandardScaler(),
                MLPRegressor(**params),
            )

            cur_X_train = train_list_X[split_idx]
            cur_y_train = train_list_y[split_idx]

            cur_X_val = val_list_X[split_idx]
            cur_y_val = val_list_y[split_idx]

            cur_X_train, cur_y_train = clean_input_array(cur_X_train, cur_y_train)
            cur_X_val, cur_y_val = clean_input_array(cur_X_val, cur_y_val)

            regr.fit(cur_X_train, cur_y_train)

            #######  Validation  ########
            #  Prediction
            cur_prediction = regr.predict(cur_X_val)
            # Scoring
            errors[split_idx] = scorer(cur_y_val, cur_prediction, regr, cur_X_train)

        # Collect validation result
        val_metrics = choose_worst(errors)
        # return_value = val_metrics[0] if pd.notnull(val_metrics[0]) else -1e6 # для evs
        return_value = (
            val_metrics[-1] if pd.notnull(val_metrics[-1]) else +1e6
        )  # для rmse
        return return_value

    # Perform the optimization
    res = gp_minimize(objective, space, n_calls=n_calls, n_jobs=-1)

    # Extract the best parameters
    best_params = {dim.name: val for dim, val in zip(space, res.x)}
    best_value = res.fun
    print("Best Hyperparameters:", best_params)

    return best_params, cur_X_test, cur_y_test, best_value


@ignore_warnings(category=ConvergenceWarning)
def do_bayes_opt(X, y, n_iter=100, **kwargs):
    n_splits = kwargs.get("n_splits", 3)
    n_layers = kwargs.get("n_layers", 30)
    n_neurons = kwargs.get("n_neurons", 100)

    # Preparing datasets
    cur_X_test, cur_y_test, val_list_X, val_list_y, train_list_X, train_list_y = (
        split_transform_one_comp_cv(X, y, n_splits=n_splits)
    )

    # Define the search space
    pbounds = {
        "n_layers": (1, n_layers),
        "n_units_0": (1, n_neurons),
        "learning_rate_init": (1e-6, 0.1),
        "early_stopping": (0, 1),
        "max_iter": (10, 15000),
        "learning_rate": (0, 2),
        "alpha": (0, 4),
        "activation": (0, 2),
        "solver": (0, 2),
    }

    def objective(
        n_layers,
        n_units_0,
        learning_rate_init,
        early_stopping,
        max_iter,
        learning_rate,
        alpha,
        activation,
        solver,
    ):
        k_layers = int(n_layers)
        layers = [int(n_units_0) for _ in range(k_layers)]

        params = {
            "hidden_layer_sizes": tuple(layers),
            "learning_rate_init": learning_rate_init,
            "random_state": 100,
            "early_stopping": bool(early_stopping),
            "max_iter": int(max_iter),
            "learning_rate": ["constant", "invscaling", "adaptive"][int(learning_rate)],
            "alpha": [0.3, 0.1, 0.01, 0.001, 0.0001][int(alpha)],
            "activation": ["logistic", "relu", "tanh"][int(activation)],
            "solver": ["lbfgs", "adam", "sgd"][int(solver)],
        }

        # Fitting and scoring `n_split` times
        errors = np.zeros((n_splits, 9))

        for split_idx in range(n_splits):
            regr = make_pipeline(
                StandardScaler(),
                MLPRegressor(**params),
            )

            cur_X_train = train_list_X[split_idx]
            cur_y_train = train_list_y[split_idx]

            cur_X_val = val_list_X[split_idx]
            cur_y_val = val_list_y[split_idx]

            cur_X_train, cur_y_train = clean_input_array(cur_X_train, cur_y_train)
            cur_X_val, cur_y_val = clean_input_array(cur_X_val, cur_y_val)

            regr.fit(cur_X_train, cur_y_train)

            #######  Validation  ########
            #  Prediction
            cur_prediction = regr.predict(cur_X_val)
            # Scoring
            errors[split_idx] = scorer(cur_y_val, cur_prediction, regr, cur_X_train)

        # Collect validation result
        val_metrics = choose_worst(errors)
        # return_value = val_metrics[0] if pd.notnull(val_metrics[0]) else -1e6 # для evs
        return_value = (
            val_metrics[-1] if pd.notnull(val_metrics[-1]) else +1e6
        )  # для rmse
        return -return_value  # Invert the value for minimization

    # Perform the optimization
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=RANDOM_STATE,
    )
    optimizer.maximize(init_points=10, n_iter=n_iter)

    # Extract the best parameters
    best_params = optimizer.max["params"]
    best_params["n_layers"] = int(best_params["n_layers"])
    best_params["n_units_0"] = int(best_params["n_units_0"])
    best_params["early_stopping"] = bool(best_params["early_stopping"])
    best_params["max_iter"] = int(best_params["max_iter"])
    best_params["learning_rate"] = ["constant", "invscaling", "adaptive"][
        int(best_params["learning_rate"])
    ]
    best_params["alpha"] = [0.3, 0.1, 0.01, 0.001, 0.0001][int(best_params["alpha"])]
    best_params["activation"] = ["logistic", "relu", "tanh"][
        int(best_params["activation"])
    ]
    best_params["solver"] = ["lbfgs", "adam", "sgd"][int(best_params["solver"])]

    best_value = -optimizer.max["target"]
    print("Best Hyperparameters:", best_params)

    return best_params, cur_X_test, cur_y_test, best_value


def rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)


def test_after_opt(best_params, x, y, model_name, path_import, metric="rmse"):
    n_layers = best_params.get("n_layers")
    layers = []
    for i in range(n_layers):
        layers.append(best_params.get(f"n_units_{i}"))

    params = best_params.copy()
    [params.pop(k) for k in best_params.keys() if "n_" in k]
    params["hidden_layer_sizes"] = tuple(layers)

    best_regr = make_pipeline(
        StandardScaler(),
        MLPRegressor(**params),
    )

    cur_X_test, cur_y_test, cur_X_train, cur_y_train = get_train_test(x, y)
    best_regr.fit(cur_X_train, cur_y_train)

    #######  testing  ########
    cur_prediction = best_regr.predict(cur_X_test)
    cur_prediction_train = best_regr.predict(cur_X_train)

    model_to_save = (best_regr, cur_prediction, cur_y_test, cur_X_test)
    saver(model_to_save, model_name=model_name, path_import=path_import)

    metrics_dict = {
        "rmse": rmse,
        "root_mean_squared_error": root_mean_squared_error,
        "max_error": max_error,
        "mean_absolute_error": mean_absolute_error,
        "mae": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "mse": mean_squared_error,
        "explained_variance_score": explained_variance_score,
        "evs": explained_variance_score,
        "mean_absolute_percentage_error": mean_absolute_percentage_error,
        "mape": mean_absolute_percentage_error,
    }
    _metric = metrics_dict.get(metric)
    test_error = _metric(cur_y_test, cur_prediction)
    train_error = _metric(cur_y_train, cur_prediction_train)
    print(f"test {metric} = {test_error}")
    return best_regr, cur_prediction, cur_y_test, cur_X_test, test_error, train_error


# Настройка логгера
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class KANModelTrainTest:
    """Model for create, train and test KAN model"""

    def __init__(self, use_gpu=True):
        self.model = None
        self.train_input = None
        self.train_label = None
        self.test_input = None
        self.test_label = None
        self.best_params = None
        self.best_value = None
        self.seed = RANDOM_STATE
        # Set device to GPU if available and use_gpu is True
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        logger.info(f"Using device: {self.device}")

    def create_train_val_test(self, X, y, n_splits=5):
        """Create train and test sets"""
        cur_X_test, cur_y_test, val_set_X, val_set_y, train_set_X, train_set_y = (
            split_transform_one_comp_cv(X, y, n_splits=n_splits)
        )
        self.train_set_X = train_set_X
        self.train_set_y = [s.reshape(-1, 1) for s in train_set_y]
        self.val_set_X = val_set_X
        self.val_set_y = [s.reshape(-1, 1) for s in val_set_y]
        self.cur_X_test = cur_X_test
        self.cur_y_test = cur_y_test.reshape(-1, 1)
        self.input_layer = self.train_set_X[0].shape[1]
        self.output_layer = (
            1 if len(self.train_set_y[0].shape) == 1 else self.train_set_y[0].shape[1]
        )

    def create_dataset(self, x_train, y_train, x_test, y_test):
        tensor_x = torch.Tensor(x_train).to(self.device)  # transform to torch tensor and move to device
        tensor_y = torch.Tensor(y_train).to(self.device)

        test_tensor_x = torch.Tensor(x_test).to(self.device)
        test_tensor_y = torch.Tensor(y_test).to(self.device)

        my_dataset = {
            "train_input": tensor_x,
            "test_input": test_tensor_x,
            "train_label": tensor_y,
            "test_label": test_tensor_y,
        }
        
        return my_dataset

    def train_model(self, width, opt, steps, grid, k):
        """Train KAN model"""
        list_val_rmse = []
        for x_train, y_train, x_val, y_val in zip(
            self.train_set_X, self.train_set_y, self.val_set_X, self.val_set_y
        ):
            kan_model = KAN(width=width, seed=self.seed, grid=grid, k=k, device=self.device)
            my_dataset = self.create_dataset(x_train, y_train, x_val, y_val)
            def train_rmse():
                try:
                    with torch.no_grad():
                        predictions = kan_model(my_dataset["train_input"])
                        # First check if predictions contain NaN values and replace them
                        predictions_clean = torch.where(
                            torch.isnan(predictions),
                            my_dataset["train_label"],  # If NaN, use the actual label (will give MSE=0 for those points)
                            predictions
                        )
                        # Calculate element-wise squared error to have more control
                        squared_errors = torch.pow(predictions_clean - my_dataset["train_label"], 2)
                        # Average the errors
                        mse = torch.mean(squared_errors)
                        # Ensure MSE is positive and not too small
                        mse_safe = torch.clamp(mse, min=1e-6)
                        rmse = torch.sqrt(mse_safe)
                        # Final check for NaN
                        if torch.isnan(rmse).item():
                            logger.warning("NaN detected in train_rmse, returning safe value")
                            return torch.tensor(10.0, device=self.device).detach()
                        return rmse.detach()
                except Exception as e:
                    logger.error(f"Error in train_rmse: {str(e)}")
                    return torch.tensor(10.0, device=self.device).detach()

            def test_rmse():
                try:
                    with torch.no_grad():
                        predictions = kan_model(my_dataset["test_input"])
                        # First check if predictions contain NaN values and replace them
                        predictions_clean = torch.where(
                            torch.isnan(predictions),
                            my_dataset["test_label"],  # If NaN, use the actual label
                            predictions
                        )
                        # Calculate element-wise squared error
                        squared_errors = torch.pow(predictions_clean - my_dataset["test_label"], 2)
                        # Average the errors
                        mse = torch.mean(squared_errors)
                        # Ensure MSE is positive and not too small
                        mse_safe = torch.clamp(mse, min=1e-6)
                        rmse = torch.sqrt(mse_safe)
                        # Final check for NaN
                        if torch.isnan(rmse).item():
                            logger.warning("NaN detected in test_rmse, returning safe value")
                            return torch.tensor(10.0, device=self.device).detach()
                        return rmse.detach()
                except Exception as e:
                    logger.error(f"Error in test_rmse: {str(e)}")
                    return torch.tensor(10.0, device=self.device).detach()
            result: dict = kan_model.fit(
                my_dataset,
                opt=opt,
                steps=steps,
                metrics=(train_rmse, test_rmse),
                loss_fn=torch.nn.MSELoss(),
            )
            val_rmse = result["test_loss"][-1]
            logger.info(f"val_rmse = {val_rmse}")
            list_val_rmse.append(val_rmse)
        return list_val_rmse

    def calc_validation_metric(self, list_val_rmse) -> float:
        return np.array(list_val_rmse).max()

    def optimize_hyperparams(
        self,
        n_trials=100,
        max_n_layers=3,
        max_steps=15,
        max_n_units=5,
        max_grid=15,
        max_k=10,
        n_jobs=1,
    ):
        """Optimize hyperparameters of KAN model using Optuna"""

        def objective(trial: optuna.trial.Trial):
            """
            Objective function for Optuna to optimize the hyperparameters of the KAN model.

            Args:
            trial (optuna.trial.Trial): A single trial object to suggest hyperparameters.

            Returns:
            float: The validation metric (e.g., RMSE) to minimize.
            """
            n_layers = trial.suggest_int("n_layers", 1, max_n_layers)
            opt = trial.suggest_categorical("opt", ["LBFGS", "Adam"])
            steps = trial.suggest_int("steps", 1, max_steps)
            grid = trial.suggest_int("grid", 1, max_grid)
            k = trial.suggest_int("k", 1, max_k)
            layers = []
            for i in range(n_layers):
                layers.append(trial.suggest_int(f"n_units_{i}", 1, max_n_units))

            width = [self.input_layer] + layers + [self.output_layer]
            logger.info(f"width = {width}")
            try:
                list_val_rmse = self.train_model(
                    width=width, opt=opt, steps=steps, grid=grid, k=k
                )
                return self.calc_validation_metric(list_val_rmse)
            except Exception as e:
                logger.error(f"Error during training: {e}")
                return 1e6

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, gc_after_trial=True, n_jobs=n_jobs)
        self.best_params = study.best_params
        return self.best_params

    def calc_test_metric(self):
        """Calculate metric on test dataset"""
        k_layers = self.best_params.get("n_layers")
        layers = []
        for i in range(k_layers):
            layers.append(self.best_params.get(f"n_units_{i}"))

        width = [self.input_layer] + layers + [self.output_layer]
        opt = self.best_params.get("opt")
        steps = self.best_params.get("steps")
        k = self.best_params.get("k")
        grid = self.best_params.get("grid")

        test_model = KAN(
            width=width,
            seed=self.seed,
            grid=grid,
            k=k,
        )
        my_dataset = self.create_dataset(
            self.train_set_X[0],
            self.train_set_y[0],
            self.cur_X_test,
            self.cur_y_test,
        )
        def train_rmse():
            try:
                with torch.no_grad():
                    predictions = test_model(my_dataset["train_input"])
                    # First check if predictions contain NaN values and replace them
                    predictions_clean = torch.where(
                        torch.isnan(predictions),
                        my_dataset["train_label"],  # If NaN, use the actual label
                        predictions
                    )
                    # Calculate element-wise squared error to have more control
                    squared_errors = torch.pow(predictions_clean - my_dataset["train_label"], 2)
                    # Average the errors
                    mse = torch.mean(squared_errors)
                    # Ensure MSE is positive and not too small
                    mse_safe = torch.clamp(mse, min=1e-6)
                    rmse = torch.sqrt(mse_safe)
                    # Final check for NaN
                    if torch.isnan(rmse).item():
                        logger.warning("NaN detected in test calc train_rmse, returning safe value")
                        return torch.tensor(10.0, device=self.device).detach()
                    return rmse.detach()
            except Exception as e:
                logger.error(f"Error in test calc train_rmse: {str(e)}")
                return torch.tensor(10.0, device=self.device).detach()
                
        def test_rmse():
            try:
                with torch.no_grad():
                    predictions = test_model(my_dataset["test_input"])
                    # First check if predictions contain NaN values and replace them
                    predictions_clean = torch.where(
                        torch.isnan(predictions),
                        my_dataset["test_label"],  # If NaN, use the actual label
                        predictions
                    )
                    # Calculate element-wise squared error
                    squared_errors = torch.pow(predictions_clean - my_dataset["test_label"], 2)
                    # Average the errors
                    mse = torch.mean(squared_errors)
                    # Ensure MSE is positive and not too small
                    mse_safe = torch.clamp(mse, min=1e-6)
                    rmse = torch.sqrt(mse_safe)
                    # Final check for NaN
                    if torch.isnan(rmse).item():
                        logger.warning("NaN detected in test calc test_rmse, returning safe value") 
                        return torch.tensor(10.0, device=self.device).detach()
                    return rmse.detach()
            except Exception as e:
                logger.error(f"Error in test calc test_rmse: {str(e)}")
                return torch.tensor(10.0, device=self.device).detach()
        # Train the model
        result: dict = test_model.fit(
            my_dataset,
            opt=opt,
            steps=steps,
            metrics=(train_rmse, test_rmse),
            loss_fn=torch.nn.MSELoss(),
        )
        self.test_rmse = result["test_loss"][-1]
        self.test_pred = test_model(torch.Tensor(self.cur_X_test))
        self.final_model = test_model
        logger.info(f"test_rmse = {self.test_rmse}")
        return self.test_rmse

    def get_train_dataset(self):
        return self.train_set_X, self.train_set_y

    def get_test_dataset(self):
        return self.cur_X_test, self.cur_y_test

    def get_val_dataset(self):
        return self.val_set_X, self.val_set_y
