def if_number_contein_five(number):
    """
    Дано двузначное число. Определите, входит ли в него цифра 5.
    Попробуйте решить задачу с использованием целочисленного деления
    и деления с остатком.
    :param number:
    :return:
    """
    number_list = map(int, list(str(number)))
    return 5 in number_list


print(if_number_contein_five(55))


def number_not_in_intervel(number):
    """
    Запишите логическое выражение, которое определяет,
    что число А не принадлежит интервалу от -10 до -1 или интервалу от 2 до 15.
    :param number:
    :return:
    """
    interval_list1 = list(range(-10, 0))
    interval_list2 = list(range(2, 16))

    return not (number in interval_list1 or number in interval_list2)


print(number_not_in_intervel(-5))


def is_list_uniq_elements(lst):
    """
    Проверьте, все ли элементы в списке являются уникальными.

    :param lst:
    :return:
    """
    list_to_set = set(lst)

    return len(lst) == len(list_to_set)


print(is_list_uniq_elements([1, 2, 15, 1]))


def is_num_palindrom(num):
    """
    Дано натуральное восьмизначное число.
    Выясните, является ли оно палиндромом
    (читается одинаково слева направо и справа налево).
    :param num:
    :return:
    """

    num_str = str(num)
    revers_num_str = num_str[::-1]

    return num_str == revers_num_str


print(is_num_palindrom(123321))


def print_asterix(n):
    counter = 1
    for i in range(n):
        print('*' * counter)

        counter += 1


print_asterix(5)


def sort_list_by_mass(lst):
    """
    Представим, что мы пишем программу для компании, занимающейся грузоперевозками.
    В нашем распоряжении есть список масс товаров, предназначенных для перевозки (в килограммах).
    Согласно нашим данным, товары, которые весят больше 100 килограммов, необходимо перевозить на грузовой машине,
    а товары меньшей массы можно перевести на легковой.
    Необходимо написать алгоритм, который распределял бы товары по типам машин.

    :param lst: Исходный лист с массами всех товаров
    :return: кортеж содержащий список легких и тяжелых товаров отдельно
    """

    light_list = []
    heavy_list = []

    for i in lst:
        if i > 100:
            heavy_list.append(i)
        else:
            light_list.append(i)

    return light_list, heavy_list


weight_of_products = [10, 42.4, 240.1, 101.5, 98, 0.4, 0.3, 15]
print(sort_list_by_mass(weight_of_products))

places = [
    'Red Square',
    'Swallow Nest',
    'Niagara Falls',
    'Grand Canyon',
    'Louvre',
    'Hermitage'
]

location = {
    'Red Square': 'Russia',
    'Swallow Nest': 'Russia',
    'Niagara Falls': 'USA',
    'Grand Canyon': 'USA',
    'Louvre': 'France',
    'Hermitage': 'Russia'
}

i = 0

for i in range(len(places)):
    place = places[i]
    country = location[place]
    if country != 'Russia':
        places[i] = 'Unavailable'

print(places)

# Напишите цикл, который складывает все строки из списка в одно предложение,
# и выведите это предложение на экран.
word_list = ['My', 'name', 'is', 'Artem']

result_str = ""
for i in word_list:
    result_str = result_str + i + " "

print(result_str.strip())

# Создайте список num_list = [1, 10, 3, -5].
# Отсортируйте его с помощью метода sort() для списков.
# Последовательно выведите на экран элементы этого списка с помощью цикла for.
num_list = [1, 10, 3, -5]
num_list.sort()

for i in num_list:
    print(i)

print("*********")
# Создайте список my_list = list(range(0, 100, 3)).
# С помощью цикла for посчитайте количество чётных элементов в списке.
# В ответ запишите получившееся число.
my_list = list(range(0, 100, 3))
even_number = 0

for i in my_list:
    if i % 2 == 0:
        even_number += 1

print(even_number)

print("*********")
# Создайте список my_list = [True, 1, -10, 'hello', False, 'string_1', 123, 2.5, [1, 2], 'another'].
# С помощью цикла for посчитайте количество элементов типа str в списке. В ответ запишите получившееся число.
my_list = [True, 1, -10, 'hello', False, 'string_1', 123, 2.5, [1, 2], 'another']

count_str = 0
for i in my_list:
    if isinstance(i, str):
        count_str += 1

print(count_str)

print("*********")
# Пусть в нашем распоряжении есть два числа x = 21 и y = 55.
# Мы хотим посчитать, сколько раз надо прибавить к x число 2,
# чтобы x стало больше или равно y.

x = 21
y = 55
counter = 0

while x < y:
    x += 2
    counter += 1

print(counter)

print("*********")
# Условие задачи. Пассажирский лифт имеет ограничение на перевозку — не более 400 кг единовременно.
# Пусть входящие в лифт люди имеют одинаковый вес (weight).
#
# Необходимо написать программу, которая должна следить за изменением нагрузки на лифт,
# сравнивать её с грузоподъёмностью и в случае перевеса выдавать предупреждение 'Overweight N kg').
# Например, при перевесе в 15 кг должно быть выведено сообщение 'Overweight 15 kg').
#
# Для данного примера вес заходящих людей зададим как weight = 67).

max_weight = 400
weight = 75
current_weight = 0

while current_weight < 400:
    current_weight += weight

overweight = current_weight - max_weight
print(f'Overweight {overweight} kg')

print("*********")
# Написать цикл, который будет складывать натуральные числа, пока их сумма не превысит 500.
# Натуральные числа — это числа 1, 2, 3, 4 и т. д.

sum_num = 0
current_num = 1

while sum_num < 500:
    sum_num += current_num
    current_num += 1

print(f"Сумма чисел: {sum_num}")

print("*********")
# Напишите цикл while, который находит максимальное натуральное число, квадрат которого меньше 1000.
max_num = 1

while max_num ** 2 < 500:
    max_num += 1

print(f"Max number: {max_num}")

print("*********")
#  Напишите бесконечный цикл while с условием выхода внутри цикла,
#  который находит максимальное натуральное число, квадрат которого меньше 1000.

max_num1 = 1

while True:
    if max_num1**2 > 1000:
        break

    max_num1 += 1

print(f'Макс натуральное число квадрат которого меньше 1000 это {max_num1}')


n = 10
i = 1
while i ** 2 < n:
    i += 1
print(i)


# Напишите цикл while, который вычисляет произведение натуральных чисел (1*2*3*4*...).
# Цикл должен выполняться до тех пор, пока значение произведения не превысит 1 000.
# В ответ запишите полученное после цикла произведение.

current_num = 1
final_num = 1

while final_num < 1000:
    current_num += 1
    final_num *= current_num

print(final_num)


# Напишите программу, которая возводит число 3 в n-ю степень (3, 9, 27 …), начиная с n = 1.
# На какой итерации цикла значение n превысит значение 1000? В ответ запишите номер этой итерации.
init_num = 1

while 3**init_num < 1000:
    init_num += 1

print(init_num)


# Олег положил тысячу рублей в банк под 8% годовых.
# Через сколько лет у него на счету будет не менее трёх тысяч рублей?
# Выведите на экран это число и запишите его в ответ.

init_deposit = 1000
rate = 8
years = 0

while init_deposit < 3000:
    init_deposit *= 1.08
    years += 1

print(years)