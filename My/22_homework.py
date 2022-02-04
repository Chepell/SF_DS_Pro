python_string = 'Hello! My name is Python. I will help you to analyze some data'

print(len(python_string) ** 3)

print(python_string[18:24])

python_string = python_string.replace("! ", " ")
python_string = python_string.replace(". ", " ")

lst = python_string.split()
print(len(lst))
print(lst)


# Напишите функцию change_password, которая должна возвращать отформатированную строку в следующем виде:
#
# User user_name changed password to new_password
#
# Мы уже сделали заготовку функции — вам осталось только задать строку.
#
# Переменные, которые надо использовать,
# указаны в круглых скобках после имени функции (user_name — имя пользователя, new_password — новый пароль).
#
# Запишите форматированную строку вместо знаков вопроса.

def change_password(user_name, new_password):
    return "User {} changed password to {}".format(user_name, new_password)


print(change_password("Lena", "qwerty"))


# Напишите функцию get_unique_words(), которая избавляется от знаков препинания в тексте и возвращает
# упорядоченный список (слова расположены по алфавиту) из уникальных (неповторяющихся) слов.
# Учтите, что слова, написанные в разных регистрах считаются одним и тем же словом.
def get_unique_words(text):
    punctuation_list = ['.', ',', ';', ':', '...', '!', '?', '-', '"', '(', ')']

    for symbol in punctuation_list:
        text = text.replace(symbol, "")

    text = text.lower()

    split_list = text.split()

    to_set = set(split_list)

    back_to_list = list(to_set)
    back_to_list.sort()

    return back_to_list


def get_most_frequent_word(text):
    punctuation_list = ['.', ',', ';', ':', '...', '!', '?', '-', '"', '(', ')']

    for symbol in punctuation_list:
        text = text.replace(symbol, "")

    text = text.lower()
    split_list = text.split()

    print(split_list)

    word_dict = dict()

    for word in split_list:
        if word not in word_dict:
            word_dict[word] = 1
        else:
            word_dict[word] += 1

    freq = 0
    word = ""

    for key, value in word_dict.items():
        if value > freq:
            freq = value
            word = key

    return word


text_example = "A beginning is the time for taking the most delicate care that the balances are correct. This every sister of the Bene Gesserit knows. To begin your study of the life of Muad'Dib, then take care that you first place him in his time: born in the 57th year of the Padishah Emperor, Shaddam IV. And take the most special care that you locate Muad'Dib in his place: the planet Arrakis. Do not be deceived by the fact that he was born on Caladan and lived his first fifteen years there. Arrakis, the planet known as Dune, is forever his place."

print(get_unique_words(text_example))

print(get_most_frequent_word(text_example))


# Разработайте функцию holes_count(),
# которая подсчитывает количество отверстий в заданном числе.
# Например, в цифре 8 два отверстия, в цифре 9 — одно.
# В числе 146 два отверстия.


def holes_count(number):
    holes_dict = {
        '0': 1,
        '1': 0,
        '2': 0,
        '3': 0,
        '4': 1,
        '5': 0,
        '6': 1,
        '7': 0,
        '8': 2,
        '9': 1
    }

    num_str = str(number)

    counter = 0

    for number in num_str:
        counter += holes_dict[number]

    return counter


print(holes_count(88))

# Напишите программу, которая запрашивает у пользователя
# следующие данные : username, age, email о нескольких пользователях и собирает эти данные в структуру:
# [(1, {'username': user1, 'age': age1, 'email': email1}),
# (2, {'username': user2, 'age': age2, 'email': email2}), ... ]
# Первый элемент каждого кортежа — порядковый номер пользователя, второй элемент — словарь с данными.
#
# В итоге должен получиться список с кортежами.
#
# Далее необходимо провести аналитику (собрать данные о пользователях в словарь)
#
# {'username': [user1, user2, ...],
# 'age': [age1, age2, ...],
# 'email': [email1, email2, ...]}
# и вывести эту аналитику на экран.


print("***********")


# Напишите функцию lucky_ticket(), которая проверяет, является ли билет счастливым.
# Примечание: билет счастливый, если сумма первых трёх цифр равна сумме последних трёх цифр.

def lucky_ticket(ticket_number):
    num_str = str(ticket_number)

    first_sum = 0
    for i in num_str[0:3]:
        first_sum += int(i)

    second_sum = 0
    for i in num_str[-3:]:
        second_sum += int(i)

    return first_sum == second_sum


print(lucky_ticket(111111))
print(lucky_ticket(123456))

print("***********")


# Напишите функцию def fib_number(), которая получает на вход некоторое число n и выводит n-e число Фибоначчи.
#
# Задачу можно решить как с помощью цикла for, так и с помощью цикла while.

def fib_number(n):
    result_list = [0, 1, 1]

    if n > 2:
        for i in range(3, n + 1):
            result_list.append(result_list[i - 1] + result_list[i - 2])

    return result_list[n]


print(fib_number(6))
print(fib_number(2))
print(fib_number(3))

print("*********")


# Напишите функцию def even_numbers_in_matrix(),
# которая получает на вход матрицу (список из списков) и возвращает количество чётных чисел в ней.

def even_numbers_in_matrix(matrix):
    even_counter = 0

    for row in matrix:
        for i in row:
            if i % 2 == 0:
                even_counter += 1

    return even_counter


matrix_example = [
    [1, 5, 4],
    [4, 2, -2],
    [7, 65, 88]
]

print(even_numbers_in_matrix(matrix_example))

print("*********")


# Напишите функцию def matrix_sum(), которая получает на вход две матрицы и возвращает их сумму.
# Но перед этим необходимо проверить,
# что размеры матриц одинаковы (одинаковое количество столбцов и одинаковое количество строк).
# Если размеры матриц не совпадают,
# то надо вывести на экран сообщение 'Error! Matrices dimensions are different!' и выполнение функции должно прекратиться.


def matrix_sum(matrix1, matrix2):
    if len(matrix1) != len(matrix2):
        print('Error! Matrices dimensions are different!')
        return None

    sum_matrix = []

    for row in range(len(matrix1)):
        sum_row = []
        for elem in range(len(matrix1[row])):
            if len(matrix1[row]) != len(matrix2[row]):
                print('Error! Matrices dimensions are different!')
                return None
            m1_elem = matrix1[row][elem]
            m2_elem = matrix2[row][elem]
            sum_row.append(m1_elem + m2_elem)
        sum_matrix.append(sum_row)

    return sum_matrix


matrix_example = [
    [1, 5, 4],
    [4, 2, -2],
    [7, 65, 88]
]

print(matrix_sum(matrix_example, matrix_example))

m1 = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
m2 = [[1, 1], [1, 1]]

print(matrix_sum(m1, m2))


# Реализуйте программу, которая сжимает последовательность символов. На вход подаётся последовательность вида:
#
# aaabbccccdaa
# Необходимо вывести строку, состоящую из символов и количества повторений этого символа. Вывод должен выглядеть как:
#
# a3b2c4d1a2

def zip_str(initial_str):
    final_str = ""

    current_ch = initial_str[0]
    ch_counter = 0

    for ch in initial_str:
        if ch == current_ch:
            ch_counter += 1
        else:
            final_str += current_ch + str(ch_counter)
            ch_counter = 1
            current_ch = ch

    final_str += current_ch + str(ch_counter)

    return final_str


print(zip_str("aaabbccccdaa"))
print(zip_str("drfgyllq"))

print("*********")


# Напишите функцию def distance_between_dots().
# Функция должна получать на вход координаты двух точек (в виде четырёх чисел) и возвращать расстояние между ними.
# Чтобы посчитать расстояние между точками, нужно воспользоваться формулой:
def distance_between_dots(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1 / 2)


print(distance_between_dots(0, 0, 2, 2))

print("***********")


# Напишите функцию, которая вычисляет среднее арифметическое значений списка.
# Примечание: среднее арифметическое считается как сумма всех чисел, делённая на их количество.
# Не забудьте проверить значение полученного аргумента!

def avr_list(lst):
    sum_val = 0

    for i in lst:
        sum_val += i

    return sum_val / len(lst)


print(avr_list([1, 2, 3]))

# Перепишите функцию из предыдущей задачи в виде lambda-функции
avr_lst_lambda = lambda x: sum(x) / len(x)
print(avr_lst_lambda([1, 2, 3]))

print("*********")


# Напишите функцию, которая принимает на вход строку и подсчитывает
# в ней количество слов начинающихся на каждую букву алфавита.
# Возвращать функция должна словарь следующего вида: {'a': 10, 'b': 3, 'c': 0, ...}
# Для задания словаря используйте строку с алфавитом:
# alphabet_str = 'abcdefghijklmnopqrstuvwxyz'
# Словарь с буквами создайте с помощью генератора.
# Не забудьте, что слова в предложении могут начинаться с большой буквы!

def count_ch(text):
    alphabet_str = 'abcdefghijklmnopqrstuvwxyz'
    # генерирую начальный словарь вида {'a': 0, 'b': 0, 'c': 0, .....
    alpha_gen = {x: 0 for x in alphabet_str}

    text_list = text.split()

    for word in text_list:
        ch = word.lower()[0]
        alpha_gen[ch] += 1

    return alpha_gen


print(count_ch('Hello! Word! hi!'))

print("*********")


# Дан список пользователей: ['admin', 'ivan', 'ivan_ivan']
# Напишите декоратор, который запрашивает имя пользователя и проверяет, есть ли оно в списке пользователей.
# Если да, то мы можем выполнить следующую нашу функцию get_data_from_database()

# Если пользователя нет в списке, нужно вывести об этом сообщение и
# пропустить выполнение функции get_data_from_database().

def check_name(name):
    users_list = ['admin', 'ivan', 'ivan_ivan']

    def decorator(func):
        def decorated_func(*args, **kwargs):
            if name in users_list:
                result = func(*args, **kwargs)
                return result
            else:
                print("User not found!")

        return decorated_func

    return decorator


@check_name('admin')
def get_data_from_database():
    print("Super secure data from database")


get_data_from_database()
