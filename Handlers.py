from functools import wraps
from datetime import timedelta
from time import time

import logging
import os

import numpy as np
import pandas as pd
from scipy import stats  # библиотека для расчетов
from scipy.stats import norm
from scipy.stats import t
from sklearn import metrics, model_selection
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import IsolationForest

from statsmodels.tsa.stattools import adfuller
from scipy.stats import normaltest, shapiro

import matplotlib.pyplot as plt
import seaborn as sns

import winsound


# plt.style.use("ggplot")
# sns.set_theme("notebook")


def format_timedelta(td):
    """
    Format a timedelta object into a string, breaking it down into days, hours, minutes, and seconds.

    Parameters:
    - td (timedelta): The timedelta object to format.

    Returns:
    - str: The formatted string representation of the timedelta.
    """

    # Calculate total seconds from the timedelta object
    total_seconds = td.total_seconds()

    # Less than 1 minute
    if total_seconds < 60:
        return f"{total_seconds:02.02f} sec."

    # Less than 1 hour
    elif total_seconds < 3600:
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{int(minutes):02d} min. {int(seconds):02d} sec."

    # Less than 1 day
    elif total_seconds < 86400:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{int(hours):02d} hours {int(minutes):02d} min. {int(seconds):02d} sec."

    # 1 day or more
    else:
        days = td.days
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60
        return f"{days:02d} days {int(hours):02d} hours {int(minutes):02d} min."


def show_duration_decorator(func):
    """
    A decorator that prints the duration taken for the decorated function to execute.

    Parameters:
    - func (callable): The function to decorate.

    Returns:
    - callable: The decorated function.
    """

    @wraps(func)
    def decorated_func(*args, **kwargs):
        # Record start time before executing the function
        start_time = time()

        # Execute the original function
        result = func(*args, **kwargs)

        # Record end time after function execution
        end_time = time()

        # Calculate the duration in seconds
        duration_sec = end_time - start_time

        # Convert the duration to a timedelta object
        td = timedelta(seconds=duration_sec)

        # Format the timedelta to a string
        format_td = format_timedelta(td)

        # Construct and print the output message
        text = f"Function '{func.__name__}' has completed in {format_td}"
        print(text)
        print()

        # Return the result of the original function
        return result

    return decorated_func


def get_columns_null_info_df(df):
    """
    Функция для получения информации по количеству и % нулевых значнений и типа данных в каждом из столбцов

    :param df: Датафрейм для анализа
    :return: Итоговый датафрейм с разультатамиЫ анализа
    """

    # создаём пустой список
    unique_list = []

    # Прохожу по именам столбцов в таблице
    for col in df.columns:
        count_null = df[col].isnull().sum()
        if count_null:
            item = (
                col,
                count_null,
                round(df[col].isnull().mean() * 100, 2),
                df[col].dtype,
            )
            # добавляем кортеж (который будет являться строкой датафрейма) в список
            unique_list.append(item)

    # создаю датафрейм который будет возвращаться
    unique_counts = pd.DataFrame(
        unique_list, columns=["Column Name", "Count Null", "% Null", "Type"]
    )
    # .sort_values(by='Num Unique', ignore_index=True)

    # Возвращаю датафрейм если он не пустой
    if not unique_counts.empty:
        return unique_counts
    else:
        return "No one null values!"


def get_top_unique_values(df, level=0):
    """
    Функция для получения инфомации по уникальным значениям в признаках

    :param df: Датафрейм для анализа
    :param level: Уровень уникальности в %, признаки ниже этого уровня не выводятся
    :return: Возвращает датафрейм с именем признака,
    количестве уникальных значений,
    сколько % от всей выборки занимают уникальные значения,
    наиболее часто повторяющимся уникальным значением,
    количество повторов,
    сколько % от выборки это значение занимает

    Когда % Unique' > 30, то это повод задуматься об уменьшении числа категорий
    """

    cols = list(df.columns)

    df_len = df.shape[0]

    # создаём пустой список
    unique_list = []

    for col in cols:
        col_lev = round(df[col].value_counts(normalize=True).values[0] * 100, 2)

        if col_lev > level:
            item = (
                col,
                df[col].nunique(),
                round(df[col].nunique() / df_len * 100, 2),
                df[col].value_counts(normalize=True).index[0],
                df[col].value_counts().values[0],
                col_lev,
            )
            # добавляем кортеж в список
            unique_list.append(item)

    unique_values = pd.DataFrame(
        unique_list,
        columns=[
            "Column Name",
            "Count Unique",
            "% Unique",
            "Top Value",
            "Top Value Count",
            "Top Value %",
        ],
    )

    return unique_values


def get_columns_isnull_info_df(df, in_percent=True):
    """
    Функия для получения количества пропущенных элементов в % по каждому из столбцов

    :param df: Датафрейм для анализа
    :return: Итоговый датафрейм с разультатами анализа
    """

    if in_percent:
        cols_null = df.isnull().mean() * 100
        col_name = "% of null"
    else:
        cols_null = df.isnull().sum()
        col_name = "Count null"

    cols_with_null = cols_null[cols_null > 0].sort_values(ascending=False)

    if len(cols_with_null) > 0:
        cols_with_null = cols_with_null.to_frame().reset_index()
        cols_with_null.columns = ["feature", col_name]

        return cols_with_null
    else:
        return "No one null values!"


def outliers_iqr(df, feature, log_scale=False, left=1.5, right=1.5):
    """
    Функция для определения выбросов по методу Тьюки.

    :param df: Исходный датафрейм
    :param feature: Фитча датафрейма для определения выбросов
    :param log_scale: Нужно ли логарифмировать рассмативаемый признак
    :param left: Множитель для определения левой границы выброса, по умолчанию 1.5
    :param right: Множитель для определения правой границы выброса, по умолчанию 1.5
    :return: Функция возвращает датафрейм с выбросами и отчищенный от выбросов датафрейм
    """

    x = df[feature]

    if log_scale:
        x = np.log(x)

    quartile_1, quartile_3 = (
        x.quantile(0.25),
        x.quantile(0.75),
    )
    iqr = quartile_3 - quartile_1

    lower_bound = quartile_1 - (iqr * left)
    upper_bound = quartile_3 + (iqr * right)

    outliers_mask = (x < lower_bound) | (x > upper_bound)
    outliers = df[outliers_mask]
    cleaned = df[~outliers_mask]

    info = f"Выбросы: {len(outliers)} строк ({len(outliers) / len(df) * 100:.2f}%)."

    return info, outliers, cleaned


def plot_outliers_z_score(df, feature, log_scale=False, left=3, right=3):
    """
    Функция для построения распределения исходного признака
    и приведение признака к нормальному, через логарифмирование

    :param df: Исходный датафрейм
    :param feature: Фитча датафрейма
    :param log_scale: Нужно ли логарифмировать рассмативаемый признак
    :param left: Множитель для определения левой границы выброса, по умолчанию 3
    :param right: Множитель для определения правой границы выброса, по умолчанию 3
    :return: Функция выводит график
    """

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))

    # Строим гистограмму признака в исходном масштабе
    sns.histplot(data=df, x=feature, ax=axes[0])

    x = df[feature]

    if log_scale:
        if 0 in x:
            x = np.log(x + 1)  # Если в массиве есть 0, то добавляю 1.
        else:
            x = np.log(x)

    mu = x.mean()
    sigma = x.std()

    left_bound = mu - left * sigma
    right_bound = mu + right * sigma

    # Строим гистограмму в логарифмическом масштабе
    sns.histplot(data=x, ax=axes[1])

    # Добавляем вертикальные линии для среднего и 3ех стандартных отклонений влево и вправо от среднего
    axes[1].axvline(mu, color="k", lw=2)
    axes[1].axvline(left_bound, color="k", ls="--", lw=2)
    axes[1].axvline(right_bound, color="k", ls="--", lw=2)

    plt.tight_layout()
    plt.show()


def outliers_z_score(df, feature, log_scale=False, left=3, right=3):
    """
    Функция для определения выбросов по методу 3х сигм

    :param df: Исходный датафрейм
    :param feature: Фитча датафрейма для определения выбросов
    :param log_scale: Нужно ли логарифмировать рассмативаемый признак
    :return: Функция возвращает датафрейм с выбросами и отчищенный от выбросов датафрейм
    """

    x = df[feature]

    if log_scale:
        # Если в серии минимальное значение 0,
        # то небходимо добавить 1 во всю серии, т.к. логарифм от 0 невозможен
        if 0 in x:
            x = np.log(x + 1)
        else:
            x = np.log(x)

    mu = x.mean()
    sigma = x.std()

    lower_bound = mu - left * sigma
    upper_bound = mu + right * sigma

    outliers_mask = (x < lower_bound) | (x > upper_bound)
    outliers = df[outliers_mask]
    cleaned = df[~outliers_mask]

    info = f"Выбросы: {len(outliers)} строк ({len(outliers) / len(df) * 100:.2f}%)."

    return info, outliers, cleaned


def get_low_inform_features_list(df, level=0.95):
    """
    Метод для получения списка признаков с низкой информативностью

    :param df: Датафрейм для анализа признаков
    :param level: Уровень попадания в низкоинформативные, по умолчанию 0.95
    :return: Возвращает список признаков датафрейма, которые имеют низкую информативность
    """

    # список неинформативных признаков
    low_inform_features = []

    # цикл по всем столбцам
    for col in df.columns:
        # наибольшая относительная частота в признаке
        top_freq = df[col].value_counts(normalize=True).max()
        # доля уникальных значений от размера признака
        nunique_ratio = df[col].nunique() / df[col].count()

        # сравниваем наибольшую частоту с порогом
        if top_freq > level:
            low_inform_features.append(col)
            print(f"{col}: {round(top_freq * 100, 2)}% одинаковых значений")
        # сравниваем долю уникальных значений с порогом
        elif nunique_ratio > level:
            low_inform_features.append(col)
            print(f"{col}: {round(nunique_ratio * 100, 2)}% уникальных значений")

    if low_inform_features:
        return low_inform_features

    return "Нет малоинформативных признаков!"


def QQ_Plots(df, column_name):
    """
    Метод для получения графиков Q-Q Plots для проверки нормальности распределения фитчи

    :param df: Датафрейм для анализа признаков
    :param column_name: Имя столбца по которому построить графики
    """

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)  # задаем сетку рисунка количество строк и столбцов
    stats.probplot(df[column_name], plot=plt)  # qq plot

    plt.subplot(1, 2, 2)  # располагаем второй рисунок рядом
    plt.hist(df[column_name])  # гистограмма распределения признака
    plt.title(column_name)

    plt.tight_layout()  # чтобы графики не наезжали другу на друга, используем tight_layout

    plt.show()  # просмотр графика


def box_and_hist_plots(df, feature):
    """
    Generate a combined plot with a box plot at the top
    and a histogram at the bottom for a given feature.

    Parameters:
    -----------
    df : DataFrame
        Input dataframe containing the data.
    feature : str
        The name of the column to visualize.

    Returns:
    --------
    None
        Displays the combined plot.
    """

    fig, ax = plt.subplots(
        2, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": (0.50, 0.85)}
    )

    sns.boxplot(x=df[feature], ax=ax[0])
    sns.histplot(data=df, x=feature, kde=True, ax=ax[1])

    ax[0].set(xlabel="")
    ax[0].set_title(f"Box-Plot and Distribution for '{feature}'", fontsize=12)

    plt.ylabel("Count", fontsize=10)
    plt.xlabel("Class", fontsize=10)

    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)

    plt.tight_layout()
    plt.show()


def null_heatmap_plot(df):
    """
    Функция для построения тепловой карты пропуска в столбцах датафрейма.
    Выводятся только столбцы в которых есть пропуски!

    :param df:
    :return:
    """

    cols_null_perc = df.isnull().mean() * 100
    cols_null_perc = cols_null_perc[cols_null_perc > 0]
    cols_null_perc = cols_null_perc.index.to_list()

    colors = ["blue", "yellow"]
    fig = plt.figure(figsize=(5, 15))
    ax = sns.heatmap(df[cols_null_perc].isnull(), cmap=colors, cbar=False)
    ax.set_title("Null Heatmap")


def confidence_interval_v0(n, x_mean, sigma, gamma=0.95):
    """
    Функци для расчета доверительного интервала, когда известно истинное стандартное отклонение на всей совокупности

    :param n: Количество элементов в выборке
    :param x_mean: Выборочное среднее
    :param sigma: Истинное стандартное отклонение
    :param gamma: Уровень надежности
    :return: Возвращает кортеж границ доверительного интервала
    """

    alpha = 1 - gamma
    z_crit = -norm.ppf(alpha / 2)
    print(z_crit)
    eps = z_crit * sigma / (n**0.5)  # погрешность
    lower_bound = x_mean - eps  # левая (нижняя) граница
    upper_bound = x_mean + eps  # правая (верхняя) граница
    return round(lower_bound, 2), round(upper_bound, 2)


def confidence_interval(n, x_mean, x_std, gamma=0.95):
    """
    Функци для расчета доверительного интервала, когда известно выборочное стандартное отклонение

    :param n: Количество элементов в выборке
    :param x_mean: Выборочное среднее
    :param x_std: Выборочное стандартное отклонение
    :param gamma: Уровень надежности
    :return: Возвращает кортеж границ доверительного интервала
    """

    alpha = 1 - gamma
    k = n - 1
    t_crit = -t.ppf(alpha / 2, k)  # t-критическое
    eps = t_crit * x_std / (n**0.5)  # погрешность
    lower_bound = x_mean - eps  # левая (нижняя) граница
    upper_bound = x_mean + eps  # правая (верхняя) граница

    return round(lower_bound, 2), round(upper_bound, 2)


def proportions_confidence_interval(n, x_p, gamma=0.95):
    """
    Функция расчета доверительного интервала для конверсий.
    Конверсия - доля пользователей совершивших целевое действие.

    :param n: Размер выборки
    :param x_p: Выборочная пропорция или конверсия x/n
    :param gamma: Уровень надежности
    :return: Возвращает кортеж из границ доверительного интервала
    """

    alpha = 1 - gamma  # уровень значимости
    z_crit = -norm.ppf(alpha / 2)  # z критическое
    eps = z_crit * (x_p * (1 - x_p) / n) ** 0.5  # погрешность
    lower_bound = x_p - eps  # левая (нижняя) граница
    upper_bound = x_p + eps  # правая (верхняя) граница
    # возвращаем кортеж из округлённых границ интервала
    return round(lower_bound * 100, 2), round(upper_bound * 100, 2)


def diff_proportions_confidence_interval(n, xp, gamma=0.95):
    """
    Функция для расчета доверительного интервала разницы конверсий

    :param n: Список размеров выборки для варианта A и B
    :param xp: Список выборочных пропорция или конверсий для варианта A и B
    :param gamma: Уровень надежности
    :return: Возвращает кортеж из границ доверительного интервала
    """

    alpha = 1 - gamma  # уровень значимости
    diff = xp[1] - xp[0]  # выборочная разница конверсий групп B и A
    z_crit = -norm.ppf(alpha / 2)  # z критическое
    eps = (
        z_crit * (xp[0] * (1 - xp[0]) / n[0] + xp[1] * (1 - xp[1]) / n[1]) ** 0.5
    )  # погрешность
    lower_bound = diff - eps  # левая (нижняя) граница
    upper_bound = diff + eps  # правая (верхняя) граница
    # возвращаем кортеж из округлённых границ интервала
    return round(lower_bound * 100, 2), round(upper_bound * 100, 2)


# def get_logger(path, file):
#     """
#     Функция для создания лог-файла и записи в него информации
# 
    # :param path: путь к директории
    # :param file: имя файла
    # :return: Возвращает объект логгера
    # """

    # # проверяем, существует ли файл
    # log_file = os.path.join(path, file)

    # # если  файла нет, создаем его
    # if not os.path.isfile(log_file):
    #     open(log_file, "w+").close()
    # # формат логирования
    # file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"
    # # формат даты
    # date_format = "%Y-%m-%d %H:%M:%S"
    # # конфигурируем лог-файл
    # logging.basicConfig(
    #     format=file_logging_format, datefmt=date_format, encoding="utf-8"
    # )
    # logger = logging.getLogger()
    # logger.setLevel(logging.DEBUG)
    # # создадим хэнлдер для записи лога в файл
    # handler = logging.FileHandler(log_file, encoding="utf-8")
    # # установим уровень логирования
    # handler.setLevel(logging.DEBUG)
    # # создадим формат логирования, используя file_logging_format
    # formatter = logging.Formatter(file_logging_format, date_format)
    # handler.setFormatter(formatter)
    # # добавим хэндлер лог-файлу
    # logger.addHandler(handler)
    # return logger


def get_logger(path, file):
    """
    Create a logger to write logs to a specified file.

    Args:
    path (str): Path to the directory where the log file will be saved.
    file (str): Name of the log file.

    Returns:
    logging.Logger: Configured logger object.
    """
    
    # Construct full log file path
    log_file = os.path.join(path, file)

    # Create log file if it doesn't exist
    if not os.path.isfile(log_file):
        open(log_file, "w+").close()

    # Define logging format and date format
    file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Create a logger object
    logger = logging.getLogger()

    # Set the log level
    logger.setLevel(logging.DEBUG)

    # Create a file handler for writing logs
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setLevel(logging.DEBUG)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter(file_logging_format, date_format)
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger


def get_X_y_dataset(train_data, target_feature):
    """
    Функция для получения из полного датасета двух, тренировочного и датасета, который содержит лишь целевой признак

    :param train_data: Исходный, полный датасет для тренировки
    :param target_feature: Имя целевого признака
    :return: Возвращает кортеж датасетов (X, y)
    """

    X = train_data.drop([target_feature], axis=1)
    y = train_data[target_feature]

    return X, y


def merge_train_and_test_to_full(train_data, test_data, target_feature):
    """
    Функция для объединения тернировочного и тестового датасета

    :param train_data: Тренировочный датасет с целевым признаком
    :param test_data: Тестовый датасет, столбца целового признака нет
    :param target_feature: Имя целевого признака
    :return: Возвращает объединенный датасет
    """

    # Реформатирую порядок столбцов
    train_data = reformat_columns(train_data, target_feature)

    train_data["dataset"] = "train"  # Помечаю, что это тренировочный датафрейм

    test_data[
        target_feature
    ] = 0  # На тестовом датафрейме целевой признак заполняю нулями
    test_data["dataset"] = "test"  # Помечаю, что это тестовый датафрейм

    full_data = train_data.append(test_data).reset_index(drop=True)
    full_data["dataset"] = full_data["dataset"].astype("category")

    full_data = full_data.convert_dtypes()

    return full_data


def split_full_to_train_and_test(full_data, target_feature):
    """
    Функция разбивает полный датасет на train и test части

    :param full_data: Полный датасет, содержит признак dataset со значениями train, test
    :param target_feature: Имя целевого признака
    :return: Возвращает кортеж из train и tes датасета

    """
    # Реформатирую порядок столбцов
    full_data = reformat_columns(full_data, target_feature)

    # Обработанный датасет делим назад на треин и тест части
    train_data = full_data.query('dataset == "train"').drop(["dataset"], axis=1)
    test_data = full_data.query('dataset == "test"').drop(
        ["dataset", target_feature], axis=1
    )

    return train_data, test_data


def reformat_columns(df, target_feature):
    """
    Reorder the columns of a DataFrame so that the target feature is the last column.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing all features, including the target feature.
    target_feature : str
        The name of the target feature column.

    Returns:
    --------
    reordered_df : pandas.DataFrame
        The DataFrame with columns reordered so that the target feature is the last column.
    """

    # Create a list of column names excluding the target feature
    cols = [col for col in df.columns if col != target_feature]

    # Append the target feature to the end of the list
    cols.append(target_feature)

    # Reorder the DataFrame columns
    reordered_df = df[cols]

    return reordered_df


def calculate_target_feature_per_category(
    data, feature: str, TARGET_FEATURE: str, func="mean"
):
    """
    Функция для расчета среднего (мединаы, СКО) значения для целеового по каждой категории выбранного признака

    :param data: Полный датасет
    :param feature: Имя категориального признака в разрезе которого нужно посчитать целевой
    :param TARGET_FEATURE: Имя целевого признака
    :param func: Функция по которой проводить аггрегацию. mean, median, std
    :return:
    """

    # Из датасета получение датафрейма категорий с их долей в % ко всему датасету
    df = pd.DataFrame(data[feature].value_counts(True) / len(data) * 100).reset_index()
    df.columns = [feature, "%"]  # переименовываю столбцы

    # Проверяю, что передан полный датасет, в котором есть признак отвечающий за разделение выборок на треин и тест
    if "dataset" in data.columns:
        # Получаю треин датасет в котором есть значения целевого признака
        train_data = data.query('dataset == "train"').drop(["dataset"], axis=1)
    else:
        train_data = data

    # Группирую по категориям получаю агрегированное по указанной функции значения целевого признака
    group_category = (
        train_data.groupby([feature])[TARGET_FEATURE].agg([func]).reset_index()
    )

    return df.merge(group_category, on=feature, how="left")


def plot_corr_heatmap(df, decimal_places=2):
    """
    Plot a correlation heatmap for the given DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame whose correlation matrix needs to be visualized.
    decimal_places : int, optional
        The number of decimal places to show in the annotations. Default is 2.
    """

    # Compute correlation matrix
    corr_df = df.corr()

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_df, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 9))

    # Plotting the heatmap with a mask, using a diverging color map
    ax = sns.heatmap(
        corr_df,
        annot=True,
        annot_kws={"size": 8},
        vmin=-1,
        vmax=1,
        square=True,
        fmt=f".{decimal_places}f",
        cmap="coolwarm",
        mask=mask,
        cbar=True,
    )

    # Set the fontsize for the colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)

    # Setting title and font size for x and y tick labels
    ax.set_title("Correlation Heatmap", fontsize=12)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    # Setting font size for x and y axis labels
    ax.set_xlabel(ax.get_xlabel(), fontsize=10)
    ax.set_ylabel(ax.get_ylabel(), fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_pairplot(df):
    """
    Plot pair-wise relationships in a dataset using Seaborn's pairplot.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame to be visualized using pairplot.
    """

    # Create the pairplot
    g = sns.pairplot(data=df, corner=True, plot_kws={"s": 10})

    # Adjusting the fontsize for ticks and labels
    for ax in g.axes.flat:
        if ax:  # Check if the axis is not None
            ax.tick_params(axis="both", labelsize=8)  # Adjust tick fontsize
            ax.set_xlabel(ax.get_xlabel(), fontsize=10)  # Adjust x-axis label fontsize
            ax.set_ylabel(ax.get_ylabel(), fontsize=10)  # Adjust y-axis label fontsize

    plt.tight_layout()
    plt.show()


def plot_categories(data, feature, level=5):
    """
    Visualize the distribution of a categorical feature as percentages.
    Categories exceeding a certain level are highlighted in green.

    Parameters:
    -----------
    full_data : DataFrame
        Input dataframe containing the data.
    category_feature : str
        The name of the categorical column to visualize.
    level : int or float, default=5
        Value at which to draw a horizontal benchmark line and also used
        as a threshold to color bars in green if they exceed this value.

    Returns:
    --------
    None
        Displays the bar chart.
    """

    # Compute the percentage distribution of the categories
    df = pd.DataFrame(data[feature].value_counts() / len(data) * 100).reset_index()
    df.columns = [feature, "%"]  # Rename columns

    # Define figure and axis for the plot
    fig, ax = plt.subplots(figsize=(16, 8))

    # Define x-ticks and rotate them for better visibility
    plt.xticks(df.index, df[feature], rotation=90)

    # Create a list of colors based on the condition
    bar_colors = ["green" if pct > level else "lightgrey" for pct in df["%"]]

    # Plot the distribution of categories with desired colors
    ax.bar(df.index, df["%"], color=bar_colors)

    ax.axhline(y=level, color="red")
    ax.set_ylabel(f"Category share, %")
    ax.set_xlabel(f"Category {feature}")
    ax.grid(False)  # Turn off the grid lines
    plt.show()


def plot_categories_and_targer(data, feature, TARGET_FEATURE, func="mean", level=5):
    """
    Столбчатая диаграмма для категориального признака со средним (медиана, СКО)
    значением целевого признака для каждой категории

    :param data: Датасет
    :param feature: Имя категориального признака
    :param TARGET_FEATURE: Имя целевого признака
    :param func: Функция по которой проводить аггрегацию. mean, median, std
    :param level: Уровень отсечения доли признака. По умолчанию 5%.
    Отображается на графике горизонтальной красной линией
    :return: Выводит график на экран
    """

    df = calculate_target_feature_per_category(data, feature, TARGET_FEATURE, func)

    col = df.columns

    fig, ax = plt.subplots(figsize=(16, 8))
    plt.xticks(df.index, df[feature], rotation=90)

    ax2 = ax.twinx()
    ax.bar(df.index, df[col[1]], color="lightgrey")
    ax2.plot(df.index, df[col[2]], color="green", label="Seconds")
    ax.axhline(y=level, color="red")
    ax.set_ylabel(f"Category share, {col[1]}")
    ax.set_xlabel(f"Category {feature}")
    ax2.set_ylabel(f"{func.capitalize()} {TARGET_FEATURE} per category")
    ax.grid(False)
    ax2.grid(False)
    plt.show()


def group_rare_labels(full_data, category_feature, level=5):
    """
    Функция для снижения количества категорий в указанном признаке

    :param full_data: Датасет
    :param category_feature: Имя категориального признака
    :param level: Уровень отсечения доли признака. По умолчанию 5%.
    Все что ниже заменяется на RARE.
    :return: Возвращает измененный датасет
    """

    # Получаю частоту повторения категорий в датасете в %
    category_series = full_data[category_feature].value_counts(True) * 100

    # Оставляю лишь те категории, частота которых выше level
    category_series = category_series[category_series > level]
    # Получаю список категорий
    category_list = list(category_series.index)

    full_data[category_feature] = full_data[category_feature].apply(
        lambda x: x if x in category_list else "RARE"
    )

    return full_data


def target_feature_boxplot_per_category(full_data, target_feature, category_feature):
    """
    Коробка с усами для целевого признкака по указанному категориальному признаку

    :param full_data: Объединенный датасет
    :param target_feature: Целевой признак
    :param category_feature: Категориальный признак
    :return:
    """

    current_train, _ = split_full_to_train_and_test(full_data, target_feature)

    fig = plt.figure(figsize=(18, 14))
    sns.boxplot(data=current_train, y=target_feature, x=category_feature)
    plt.title(target_feature + " per " + category_feature)

    plt.tight_layout()
    plt.show()


def plot_errors_boxplot(y_train, y_train_predict, y_test, y_test_predict):
    """
    Функция для визуализации ошибки на тренировочной и тестовой выборке.
    Ошибки в виде вектора разницы между правильными ответами и предсказаниями.

    :param y_train:
    :param y_train_predict:
    :param y_test:
    :param y_test_predict:
    :return:
    """

    # Визуализируем ошибки
    fig, ax = plt.subplots(figsize=(16, 6))  # фигура+координатная плоскость

    # Ошибки модели на тренировочной и тестовой выборке
    y_train_errors = y_train - y_train_predict
    y_test_errors = y_test - y_test_predict

    # Для удобства визуализации составим DataFrame из ошибок
    errors_df = pd.DataFrame(
        {"Train errors": y_train_errors, "Test errors": y_test_errors}
    )

    # Строим boxplot для ошибок
    sns.boxplot(data=errors_df, ax=ax, orient="h")
    ax.set_xlabel("Model errors")  # название оси абсцисс
    ax.set_ylabel("Sampling")  # название оси ординат

    ax.set_title("Prediction Errors")

    plt.tight_layout()
    plt.show()


def print_regression_metrics(
    y_train,
    y_train_predict,
    y_test,
    y_test_predict,
    show_R2=True,
    show_MAE=True,
    show_MAPE=True,
    show_MSE=False,
    show_RMSE=False,
):
    print("*** TRAIN ***")
    if show_R2:
        print(f"R^2: {metrics.r2_score(y_train, y_train_predict):.3f}")
    if show_MAE:
        print(f"MAE: {metrics.mean_absolute_error(y_train, y_train_predict):.3f}")
    if show_MAPE:
        print(
            f"MAPE: {metrics.mean_absolute_percentage_error(y_train, y_train_predict) * 100:.3f}"
        )
    if show_MSE:
        print(f"MSE: {metrics.mean_squared_error(y_train, y_train_predict):.3f}")
    if show_RMSE:
        print(
            f"RMSE: {np.sqrt(metrics.mean_squared_error(y_train, y_train_predict)):.3f}"
        )

    print()

    print("*** TEST ***")
    if show_R2:
        print(f"R^2: {metrics.r2_score(y_test, y_test_predict):.3f}")
    if show_MAE:
        print(f"MAE: {metrics.mean_absolute_error(y_test, y_test_predict):.3f}")
    if show_MAPE:
        print(
            f"MAPE: {metrics.mean_absolute_percentage_error(y_test, y_test_predict) * 100:.3f}"
        )
    if show_MSE:
        print(f"MSE: {metrics.mean_squared_error(y_test, y_test_predict):.3f}")
    if show_RMSE:
        print(
            f"RMSE: {np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)):.3f}"
        )


def print_classification_metrics(
    y_train,
    y_train_predict,
    y_test,
    y_test_predict,
    show_accuracy=True,
    show_precision=True,
    show_recall=True,
    show_f1=True,
    show_roc_auc=True,
    average="binary",
):
    """
    Print classification metrics for both training and test data.

    Parameters:
    - y_train: Ground truth labels for training data.
    - y_train_predict: Predicted labels for training data.
    - y_test: Ground truth labels for test data.
    - y_test_predict: Predicted labels for test data.
    - show_accuracy, show_precision, show_recall, show_f1: Booleans to toggle printing of respective metrics.
    - average: Method to compute multiclass classification metrics. Options include 'binary', 'micro', 'macro', 'weighted', 'samples'.

    Returns:
    None. This function only prints metrics.
    """

    def print_metrics(y_true, y_pred):
        """
        Helper function to print classification metrics for given true and predicted labels.
        """
        if show_accuracy:
            print(f"Accuracy: {metrics.accuracy_score(y_true, y_pred):.3f}")
        if show_precision:
            print(
                f"Precision: {metrics.precision_score(y_true, y_pred, average=average):.3f}"
            )
        if show_recall:
            print(
                f"Recall: {metrics.recall_score(y_true, y_pred, average=average):.3f}"
            )
        if show_f1:
            print(f"F1: {metrics.f1_score(y_true, y_pred, average=average):.3f}")

        if show_roc_auc:
            print(f"ROC AUC: {metrics.roc_auc_score(y_true, y_pred):.3f}")

    print("*** TRAIN ***")
    print_metrics(y_train, y_train_predict)
    print("\n*** TEST ***")
    print_metrics(y_test, y_test_predict)


def plot_learning_curve(model, X, y, cv, scoring="f1", ax=None, title=""):
    """
    Фунгкция для построения графика кривой обучения

    :param model: Модель машинного обучения
    :param X: Матрица признаков
    :param y: Вектор целевого признака
    :param cv: Объект кросс-валидатора
    :param scoring: Используемая метрика
    :param ax: Координатная плоскость, на которой отобразить график
    :param title: Подпись графика

    :return: Выводит график
    """

    # Вычисляем координаты для построения кривой обучения
    train_sizes, train_scores, valid_scores = model_selection.learning_curve(
        estimator=model,  # модель
        X=X,  # матрица наблюдений X
        y=y,  # вектор ответов y
        cv=cv,  # кросс-валидатор
        scoring=scoring,  # метрика
        train_sizes=np.arange(0.1, 1.1, 0.1),  # каждые 10%
    )

    # Вычисляем среднее значение по фолдам для каждого набора данных
    train_scores_mean = np.mean(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)

    # Если координатной плоскости не было передано, создаём новую
    if ax is None:
        fig, ax = plt.subplots(figsize=(18, 12))

    # Строим кривую обучения по метрикам на тренировочных фолдах
    ax.plot(train_sizes, train_scores_mean, label="Train")
    # Строим кривую обучения по метрикам на валидационных фолдах
    ax.plot(train_sizes, valid_scores_mean, label="Valid")

    # Даём название графику и подписи осям
    ax.set_title("Learning curve: {}".format(title))
    ax.set_xlabel("Train data size")
    ax.set_ylabel("Score")

    # Устанавливаем отметки по оси абсцисс
    ax.xaxis.set_ticks(train_sizes)
    # Устанавливаем диапазон оси ординат
    ax.set_ylim(0, 1)
    # Отображаем легенду
    ax.legend()


def plot_confusion_matrix(cm, title, in_pct=False):
    """
    Plots a confusion matrix using Seaborn's heatmap.

    Parameters:
    - cm: A confusion matrix (2x2 for binary classification).
    - title: Title for the plot.
    - in_pct: Boolean indicating if values should be displayed as percentages.

    Returns:
    None. Displays the plot.
    """
    # Convert matrix values to percentage if required
    if in_pct:
        cm_percentage = (cm / cm.sum() * 100).round(2)
    else:
        cm_percentage = cm

    # Prepare annotation labels for the matrix
    label_names = np.array([["TN = ", "FP = "], ["FN = ", "TP = "]])
    annotations = np.core.defchararray.add(label_names, cm_percentage.astype(str))

    # Create a heatmap for the confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm_percentage, annot=annotations, fmt="", cmap="Blues", cbar=False, robust=True
    )

    # Adjust tick settings for better visualization
    plt.tick_params(
        axis="x", bottom=False, top=True, labelbottom=False, labeltop=True, length=0
    )

    # Add title and labels
    plt.title(title, fontsize=16)
    plt.gca().xaxis.set_label_position("top")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)

    # Display the plot
    plt.show()


def fB_score(y_true, y_pred, B):
    """F-мера с возможностью указать значение беты.

    Args:
        y_true (_type_): Вектор правильных ответов
        y_pred (_type_): Вектор предсказаний
        B (float): Бета - это вес precision в метрике, чем бета больше, тем precision важнее

    Returns:
        float: Возвращает значение метрики
    """
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)

    return (1 + B**2) * ((precision * recall) / ((B**2 * precision) + recall))


def plot_PR_curve(y_train, y_cv_proba_pred):
    """
    Функция для построения PR-кривой

    Args:
        y_train: Вектор правильных ответов
        y_cv_proba_pred: Вектор с вероятностями предсказаний для первого класса

    Returns:
    """

    precision, recall, thresholds = metrics.precision_recall_curve(
        y_train, y_cv_proba_pred
    )

    # Вычисляем F1-score при различных threshold
    f1_scores = (2 * precision * recall) / (precision + recall)

    # Определяем индекс максимума F1
    idx = np.argmax(f1_scores)
    # И значение порога вероятности при котором F1 максимален
    threshold_opt = thresholds[idx]

    # Строим PR-кривую
    fig, ax = plt.subplots(figsize=(16, 8))  # фигура + координатная плоскость

    # Строим линейный график зависимости precision от recall
    ax.plot(precision, recall, label="model PR-curve")

    # Отмечаем точку максимума F1
    ax.scatter(
        precision[idx], recall[idx], marker="o", color="black", label="Best F1 score"
    )

    # Даём графику название и подписываем оси
    title_text = (
        "Precision-recall curve. "
        + f"Best probably threshold = {threshold_opt:.2f}, F1-Score = {f1_scores[idx]:.2f}. "
        + f"PR AUC = {metrics.auc(recall, precision):.2f}"
    )

    ax.set_title(title_text, fontsize=16)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    # Отображаем легенду
    ax.legend()
    plt.tight_layout()
    plt.show()


def adfuller_test(x, threshold=0.05, verbose=False):
    """
    Perform the Augmented Dickey-Fuller test to check the stationarity of a time series.

    Args:
        x (array-like): The time series data to test for stationarity.
        threshold (float, optional): Significance level for the test, default is 0.05.
        verbose (bool, optional): If True, prints the test's statistic, p-value, and critical values. Default is False.

    Returns:
        float: The p-value of the test.
    """

    # Perform the Augmented Dickey-Fuller test
    result = adfuller(x)
    statistic, p_value, _, _, critical_values = result

    # Verbose output
    if verbose:
        print("ADF Statistic:", statistic)
        print("P-Value:", p_value)
        print("Critical Values:")
        for key, value in critical_values.items():
            print(f"\t{key}: {value}")

    # Determine stationarity based on the p-value and threshold
    if p_value <= threshold:
        print("The time series is stationary.") if verbose else None
    else:
        print("The time series is not stationary.") if verbose else None

    return p_value


def pirson_normal_test(x, threshold=0.05):
    """
    Тест на нормальное распределение Д’Агостино-Пирсона

    Args:
        x: Вектор значений переданный для проверки на нормальное распределение
        threshold: Уровень достоверности, по умолчанию 5%

    Returns:
    """

    _, pvalue = normaltest(x)

    print("Test on normal distribution:", _)
    print("P-Value:", pvalue)

    # The null hypothesis is that the sample comes from a normal distribution
    H0 = "Normal distribution"
    Ha = "NOT a normal distribution!"

    if pvalue <= threshold / 2:  # Reject the null hypothesis
        print(Ha)
    else:
        print(H0)


def shapiro_normal_test(x, threshold=0.05):
    """
    Тест на нормальное распределение Шапиро — Уилка

    Args:
        x: Вектор значений переданный для проверки на нормальное распределение
        threshold: Уровень достоверности, по умолчанию 5%

    Returns:
    """

    _, p = shapiro(x)

    print("Test on normal distribution:", _)
    print("P-Value:", p)

    # The null hypothesis is that the sample comes from a normal distribution
    H0 = "Normal distribution"
    Ha = "NOT a normal distribution!"

    if p <= threshold:  # Reject the null hypothesis
        print(Ha)
    else:
        print(H0)


def profit_margin_for_zero_mo(profit_factor):
    """
    Calculate the profit margin for zero months operation.

    Parameters:
    - profit_factor (float): The profit factor to calculate the margin.

    Returns:
    - float: Profit margin, rounded to two decimal places.
    """

    # Calculate the profit margin using the provided formula and round it to 2 decimal places
    return round(1 / (profit_factor + 1), 2)


def create_X_y_from_timeseries(df_timeseries, target, T, flatten_features=True):
    """
    Transforms time series data into a supervised learning format.

    Args:
    - df_timeseries (pd.DataFrame): DataFrame containing time series data.
    - target (str): Name of the column to predict.
    - T (int): Number of time lags to use as features.
    - flatten_features (bool): If True, flattens the features for traditional ML models.
                              If False, retains 3D shape suitable for RNNs.

    Returns:
    - X (np.array): Input features.
    - y (np.array): Target values.
    - N (int): Number of samples.
    - D (int): Number of feature dimensions.
    """

    # Ensure target is a column in the DataFrame
    if target not in df_timeseries.columns:
        raise ValueError(f"'{target}' not found in the DataFrame columns.")

    # Ensure T is a valid lag value
    if not (1 <= T < len(df_timeseries)):
        raise ValueError(f"T must be between 1 and {len(df_timeseries) - 1}.")

    # Extract features and target values
    features = df_timeseries.drop(columns=target).values
    targets = df_timeseries[target].values

    # Determine dimensions
    num_samples = len(df_timeseries) - T
    num_features = features.shape[1]

    # Initialize X and y matrices
    X = np.zeros((num_samples, T, num_features))
    y = np.zeros(num_samples)

    # Populate X and y
    for t in range(num_samples):
        X[t, :, :] = features[t : t + T]
        y[t] = targets[t + T]

    # Flatten X for ML models if required
    if flatten_features:
        X = X.reshape(num_samples, T * num_features)

    return X, y, num_samples, num_features


class StateLessTransformer(BaseEstimator, TransformerMixin):
    """
    Apply a list of stateless transformation functions to the input data.
    Stateless means that the transformations don't learn any parameters
    from the training data and don't depend on it.

    Parameters:
    -----------
    transform_funcs : callable or list of callables
        Function or list of functions to apply to the input data.
        Each function should take a dataframe as its sole argument and
        return a transformed dataframe.

    Attributes:
    -----------
    transform_funcs : list of callables
        List of transformation functions to apply.
    """

    def __init__(self, transform_funcs):
        # Ensure transform_funcs is a list, even if a single function is passed
        if not isinstance(transform_funcs, list):
            transform_funcs = [transform_funcs]
        self.transform_funcs = transform_funcs

    def fit(self, X, y=None):
        """
        No fitting necessary for stateless transformations.
        Just return self.

        Parameters:
        -----------
        X : DataFrame
            Input data.
        y : Series or DataFrame, default=None
            Target variable. Not used in this transformer.

        Returns:
        --------
        self : object
            Returns the instance itself.
        """
        return self

    def transform(self, X, y=None):
        """
        Apply the provided transformation functions to the input data.

        Parameters:
        -----------
        X : DataFrame
            Input data to transform.
        y : Series or DataFrame, default=None
            Target variable. Not used in this transformer.

        Returns:
        --------
        DataFrame
            Transformed data.
        """
        X_transformed = X
        for func in self.transform_funcs:
            X_transformed = func(X_transformed)
        return X_transformed


class PeriodicTransformer(BaseEstimator, TransformerMixin):
    """
    Transform a periodic feature into its sine and cosine transformations for
    linear models to better understand periodic relationships.

    Parameters:
    -----------
    feature : str
        Name of the column in the DataFrame to be transformed.
    drop_origin : bool, default=True
        Whether to drop the original column after transformation.

    Attributes:
    -----------
    sin_name : str
        Name of the sine transformed feature column.
    cos_name : str
        Name of the cosine transformed feature column.
    max_value : float
        Maximum value of the feature column. Used for normalization.
    """

    def __init__(self, feature, drop_origin=True):
        self.feature = feature
        self.drop_origin = drop_origin
        self.sin_name = None
        self.cos_name = None
        self.max_value = None

    def fit(self, X, y=None):
        """
        Compute necessary attributes for transformation.

        Parameters:
        -----------
        X : DataFrame
            Input data.
        y : Series, default=None
            Target variable. Not used in this transformer.

        Returns:
        --------
        self : object
            Returns the instance itself.
        """
        # Generate the names for the new sin and cos features based on the original feature name
        self.sin_name = self.feature + "_sin"
        self.cos_name = self.feature + "_cos"

        # Find the maximum value of the feature to use for normalization in the transform step
        self.max_value = X[self.feature].max()
        return self

    def transform(self, X, y=None):
        """
        Apply the sine and cosine transformation to the feature.

        Parameters:
        -----------
        X : DataFrame
            Input data.
        y : Series, default=None
            Target variable. Not used in this transformer.

        Returns:
        --------
        DataFrame
            Transformed data.
        """
        # Create a copy of the input DataFrame to avoid modifying the original data
        X = X.copy()

        # Create the sin and cos transformations of the feature
        # The feature values are normalized by the max value and then multiplied by 2π to get the full circle
        X[self.sin_name] = np.sin(X[self.feature] / self.max_value * 2 * np.pi)
        X[self.cos_name] = np.cos(X[self.feature] / self.max_value * 2 * np.pi)

        # Drop the original feature as it's replaced by its sin and cos transformations
        if self.drop_origin:
            X.drop(columns=self.feature, inplace=True)
        return X


class MeanNormalizationScaler(BaseEstimator, TransformerMixin):
    """
    Perform mean normalization on data.
    The transformer scales features based on their mean and range (max-min).

    Attributes:
    -----------
    means : Series
        Means of each feature computed during fitting.
    ranges : Series
        Ranges (max-min) of each feature computed during fitting.
    """

    def __init__(self):
        self.means = None
        self.ranges = None

    def fit(self, X, y=None):
        """
        Compute means and ranges of features.

        Parameters:
        -----------
        X : DataFrame
            Input data.
        y : Series or DataFrame, default=None
            Target variable. Not used in this transformer.

        Returns:
        --------
        self : object
            Returns the instance itself.
        """
        self.means = X.mean()
        self.ranges = X.max() - X.min()
        return self

    def transform(self, X, y=None):
        """
        Apply mean normalization scaling on data.

        Parameters:
        -----------
        X : DataFrame
            Input data to transform.
        y : Series or DataFrame, default=None
            Target variable. Not used in this transformer.

        Returns:
        --------
        DataFrame
            Scaled data.
        """
        X_transform = X.copy()
        X_transform = (X_transform - self.means) / self.ranges
        return X_transform


class SelectScaler(BaseEstimator, TransformerMixin):
    """
    A custom transformer that allows for the selection of different scalers.

    Parameters:
    -----------
    name : str, default="no_scaler"
        The name of the scaler to use.
        Options are: "standard", "minmax", "robust", "mean_normalization", and "no_scaler".

    Attributes:
    -----------
    SCALERS : dict
        A dictionary containing the available scalers.
    scaler : scaler object
        The selected scaler based on the provided name.

    Example:
    --------
    >>> transformer = SelectScaler(name="minmax")
    >>> transformed_data = transformer.fit_transform(data)
    """

    SCALERS = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "mean_normalization": MeanNormalizationScaler(),
        "no_scaler": None,
    }

    def __init__(self, name="no_scaler"):
        """
        Initialize the SelectScaler with the desired scaler name.
        """
        self.name = name
        self.scaler = self.SCALERS.get(name, None)

    def fit(self, X, y=None):
        """
        Fit the selected scaler to the data if a scaler is selected.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The data to fit.
        y : array-like, shape (n_samples,), optional
            Target values. Not used.

        Returns:
        --------
        self : object
        """
        if self.scaler:
            self.scaler.fit(X, y)
        return self

    def transform(self, X, y=None):
        """
        Transform the data using the selected scaler if a scaler is selected.
        Otherwise, return the original data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The data to transform.
        y : array-like, shape (n_samples,), optional
            Target values. Not used.

        Returns:
        --------
        X_new : array-like, shape (n_samples, n_features)
            The transformed data or the original data if no scaler is selected.
        """
        if self.scaler:
            return self.scaler.transform(X)
        return X


def alert(sec=3):
    """
    Produce a beep sound as an alert.

    Parameters:
    - sec (int): Duration of the beep in seconds. Default is 3.

    Returns:
    - None
    """

    # Set frequency for the beep sound in Hertz
    frequency = 1000

    # Convert duration from seconds to milliseconds
    duration = sec * 1000

    # Produce the beep sound
    winsound.Beep(frequency, duration)


def reduce_memory_usage(df, verbose=True):
    """
    Reduce the memory usage of a pandas DataFrame by downcasting numeric columns
    to more memory-efficient data types.

    Parameters:
    df (pd.DataFrame): The DataFrame whose memory usage is to be optimized.
    verbose (bool): If True, prints the memory reduction information.

    Returns:
    pd.DataFrame: A DataFrame with optimized memory usage.
    """

    # Calculate the initial memory usage of the DataFrame
    start_mem = df.memory_usage().sum() / 1024**2

    # Iterate over each column that is of a numeric data type
    for col in df.select_dtypes(include=["number"]).columns:
        # Get the data type of the current column
        col_type = df[col].dtype

        # Check if the column is of integer type
        if pd.api.types.is_integer_dtype(col_type):
            # Calculate the minimum and maximum values in the column
            c_min, c_max = df[col].min(), df[col].max()
            # Downcast the column to the smallest possible integer type
            # that can accommodate all the values in the column
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)
        # If the column is not an integer type, it is a float
        else:
            # Similarly, downcast the column to the smallest possible
            # floating point type
            c_min, c_max = df[col].min(), df[col].max()
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    # Calculate the final memory usage after optimization
    end_mem = df.memory_usage().sum() / 1024**2

    # If verbose is True, print the amount of memory saved
    if verbose:
        print(
            f"Mem. usage decreased to {end_mem:.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)"
        )

    # Return the DataFrame with reduced memory usage
    return df
