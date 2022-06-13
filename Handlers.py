import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # библиотека визуализации
from scipy import stats # библиотека для расчетов


def get_columns_unique_info_df(df):
    """
    Функция для получиения количества уникальных значнений, количества пусты значений и типа данных в каждом из столбцов

    :param df: Датафрейм для анализа
    :return: Итоговый датафрейм с разультатами анализа
    """

    # создаём пустой список
    unique_list = []
    # пробегаемся по именам столбцов в таблице
    for col in df.columns:
        # создаём кортеж (имя столбца, число уникальных значений)
        item = (col, df[col].nunique(), df[col].isnull().sum(), df[col].dtype)
        # добавляем кортеж в список
        unique_list.append(item)
        # создаю датафрейм который будет возвращаться
    unique_counts = pd.DataFrame(
        unique_list,
        columns=['Column Name', 'Num Unique', 'Num Null', 'Type']
    )
    #.sort_values(by='Num Unique', ignore_index=True)

    return unique_counts


def get_columns_isnull_info_df(df, in_percent=True):
    """
    Функия для получения количества пропущенных элементов в % по каждому из столбцов

    :param df: Датафрейм для анализа
    :return: Итоговый датафрейм с разультатами анализа
    """

    if in_percent:
        cols_null = df.isnull().mean() * 100
    else:
        cols_null = df.isnull().sum()
        
    cols_with_null = cols_null[cols_null > 0].sort_values(ascending=False)
    
    if len(cols_with_null) > 0:
        return cols_with_null
    else:
        return 'No one null values!'


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

    quartile_1, quartile_3 = x.quantile(0.25), x.quantile(0.75),
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * left)
    upper_bound = quartile_3 + (iqr * right)
    outliers = df[(x < lower_bound) | (x > upper_bound)]
    cleaned = df[(x > lower_bound) & (x < upper_bound)]
    return outliers, cleaned


def outliers_z_score(df, feature, log_scale=False, left=3, right=3):
    """
    Функция для определения выбросов по методу 3х сигм

    :param df: Исходный датафрейм
    :param feature: Фитча датафрейма для определения выбросов
    :param log_scale: Нужно ли логарифмировать рассмативаемый признак
    :return: Функция возвращает датафрейм с выбросами и отчищенный от выбросов датафрейм
    """

    current_series = df[feature]

    if log_scale:
        # Если в серии минимальное значение 0,
        # то небходимо добавить 1 во всю серии, т.к. логарифм от 0 невозможен
        current_series = np.log(current_series + 1)

    mu = current_series.mean()
    sigma = current_series.std()

    lower_bound = mu - left * sigma
    upper_bound = mu + right * sigma

    outliers = df[(current_series < lower_bound) | (current_series > upper_bound)]
    cleaned = df[(current_series > lower_bound) & (current_series < upper_bound)]

    return outliers, cleaned


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
            print(f'{col}: {round(top_freq * 100, 2)}% одинаковых значений')
        # сравниваем долю уникальных значений с порогом
        elif nunique_ratio > level:
            low_inform_features.append(col)
            print(f'{col}: {round(nunique_ratio * 100, 2)}% уникальных значений')

    return low_inform_features


def Q_Q_plot(df, column_name):
    """
    Метод для получения графиков Q-Q Plots для проверки нормальности распределения фитчи

    :param df: Датафрейм для анализа признаков
    :param column_name: Имя столбца по которому построить графики
    """
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1) # задаем сетку рисунка количество строк и столбцов
    stats.probplot(df[column_name], plot=plt) # qq plot

    plt.subplot(1, 2, 2) # располагаем второй рисунок рядом
    plt.hist(df[column_name]) # гистограмма распределения признака
    plt.title(column_name)

    plt.tight_layout() # чтобы графики не наезжали другу на друга, используем tight_layout

    plt.show() # просмотр графика