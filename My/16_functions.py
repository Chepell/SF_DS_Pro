def root(value, n=2):
    # Как мы уже выяснили, чтобы посчитать
    # корень степени n из числа, можно возвести это число
    # в степень 1/n
    result = value ** (1 / n)
    return result


print(root(81))

print("Список покупок:", end=" ")
print("Хлеб", "Молоко", "Яйца", sep=", ")


def arg_count(*number):
    return len(number)


print(arg_count(7))


def root(value, n=2, verbose=False):
    result = value ** (1 / n)
    if verbose:
        # Аргументы в функции print,
        # перечисленные через запятую,
        # печатаются через пробел
        print('Root of power', n, 'from',
              value, 'equals', result)
    return result


print(root(verbose=True, value=25))

langs = ['Python', 'SQL', 'Machine Learning', 'Statistics']
print(*langs, sep=", ")


def mean_mark(name, *marks):
    result = sum(marks) / len(marks)
    # Не возвращаем результат, а печатаем его
    print(name + ':', result)


marks = [4, 5, 5, 5, 5, 3, 4, 4, 5, 4, 5]

mean_mark('Kuznetsov', *marks)


def print_lists(*lst, **kwords):
    if kwords:
        sep = kwords['sep']
        end = kwords['end']
        for i in lst:
            print(*i, sep=sep, end=end)
    else:
        for i in lst:
            print(i)


print_lists([1, 2, 3], [4, 5, 6], [7, 8, 9], sep=', ', end='; ')

print("*******")


# В качестве упражнения попробуйте переписать использованную lambda-функцию в классический вид (с def).
# Назовите её get_length.
# Затем воспользуйтесь полученной функцией для сортировки списка:

def get_length(word):
    return len(word)


new_list = ['bbb', 'ababa', 'aaa', 'aaaaa', 'cc']
new_list.sort(key=get_length)
print(new_list)

# Сортировка с помощью лямбда функции
new_list = ['bbb', 'ababa', 'aaa', 'aaaaa', 'cc']
new_list.sort(key=lambda word: len(word))
print(new_list)


# Напишите функцию sort_sides, которая сортирует переданный в неё список.
# Входной список состоит из кортежей с парами чисел — длинами катетов прямоугольных треугольников.
# Функция должна возвращать список, отсортированный по возрастанию длин гипотенуз треугольников.
def sort_sides(l_in):
    # В лямбдафункции обращаюсь к кортежу и уже внутри функции достаю катеты по индексу
    l_in.sort(key=lambda x: (x[0] ** 2 + x[1] ** 2) ** (1 / 2))
    return l_in


print(sort_sides([(3, 4), (1, 2), (10, 10)]))

marks = [4, 5, 5, 5, 5, 3, 4, 4, 5, 4, 5]
print(*marks, sep=", ")


def get_less(l, num):
    for i in l:
        if i < num:
            return i
    return None


l = [1, 5, 8, 10]
print(get_less(l, -1))


# Напишите функцию split_date(date), которая принимает на вход строку, задающую дату,
# в формате ддммгггг без разделителей.
# Функция должна вернуть кортеж из чисел (int): день, месяц, год.

def split_date(date):
    dd = int(date[:2])
    mm = int(date[2:4])
    yyyy = int(date[-4:])
    return dd, mm, yyyy


print_lists(split_date("27012022"))

print(split_date("31012019"))

print("********")


# Напишите функцию is_prime(num), которая проверяет, является ли число простым.
# Функция должна вернуть True, если число простое, иначе — False.
def is_prime(num):
    if num != 1 and num % 2 != 0 and num % 3 != 0:
        return True
    return False


print(is_prime(1))
print(is_prime(10))
print(is_prime(13))

print("********")


# Напишите функцию between_min_max(...), которая принимает на вход числа через запятую.
# Функция возвращает среднее арифметическое между максимальным и минимальным значением этих чисел, то есть (max + min)/2

def between_min_max(*numbers):
    max_val = max(list(numbers))
    min_val = min(list(numbers))
    return (max_val + min_val) / 2


print(between_min_max(10))
print(between_min_max(1, 2, 3, 4, 5))

print("********")


# Напишите функцию best_student(...), которая принимает на вход в виде
# именованных аргументов фамилии студентов и их номера в рейтинге (нагляднее в примере).
#
# Необходимо вернуть фамилию студента с минимальным номером по рейтингу.


def best_student(**name_and_rate):
    name = list(name_and_rate.keys())[0]
    rate = list(name_and_rate.values())[0]

    for n, r in name_and_rate.items():
        if r < rate:
            name = n
            rate = r

    return name


print(best_student(Tom=12, Mike=3))
print(best_student(Tom=12))
print(best_student(Tom=12, Jerry=1, Jane=2))

print("********")
# Напишите lambda-функцию is_palindrom, которая принимает на вход одну строку и проверяет,
# является ли она палиндромом, то есть читается ли она слева-направо и справа-налево одинаково.
#
# Функция возвращает 'yes', если строка является палиндромом, иначе — 'no'.

is_palindrom = lambda word: 'yes' if word == word[::-1] else 'no'

print(is_palindrom('1234'))
print(is_palindrom('12321'))

print("********")
# Напишите lambda-функцию area, которая принимает на вход
# два числа — стороны прямоугольника — через запятую и возвращает площадь прямоугольника.

area = lambda x, y: x * y
print(area(12, 5))

print("********")
# Перепишите функцию between_min_max из задания 7.12 в lambda-функцию.
# Функция принимает на вход числа через запятую и возвращает одно число — среднее между максимумом и минимумом этих чисел.
between_min_max = lambda *nums: (min(nums) + max(nums)) / 2
print(between_min_max(1, 2, 3, 4, 5))

print("********")
# Напишите функцию sort_ignore_case, которая принимает на вход список и сортирует его без учёта регистра по алфавиту.
# Функция возвращает отсортированный список.

def sort_ignore_case(lst):
    lst.sort(key=lambda x: x.lower())
    return lst

print(sort_ignore_case(['Acc', 'abc']))



# Напишите функцию exchange(usd, rub, rate), которая может принимать на вход сумму в долларах (usd),
# сумму в рублях (rub) и обменный курс (rate). Обменный курс показывает, сколько стоит один доллар.
# Например, курс 85.46 означает, что один доллар стоит 85 рублей и 46 копеек.
#
# В функцию должно одновременно передавать два аргумента. Если передано менее двух аргументов,
# должна возникнуть ошибка ValueError('Not enough arguments'). Если же передано три аргумента,
# должна возникнуть ошибка: ValueError('Too many arguments').
#
# Функция должна находить третий аргумент по двум переданным. Например, если переданы суммы в разных валютах,
# должен возвращаться обменный курс. Если известны сумма в рублях и курс,
# должна быть получена эквивалентная сумма в долларах, аналогично — если передана сумма в долларах и обменный курс.

def exchange(usd=None, rub=None, rate=None):
    if usd is not None and rub is not None and rate is not None:
        raise ValueError('Too many arguments')

    if usd is not None and rub is None and rate is None:
        raise ValueError('Not enough arguments')

    if usd is None and rub is not None and rate is None:
        raise ValueError('Not enough arguments')

    if usd is None and rub is None and rate is not None:
        raise ValueError('Not enough arguments')

    if usd is not None and rub is not None:
        return rub / usd

    if usd is not None and rate is not None:
        return usd * rate

    if rub is not None and rate is not None:
        return rub / rate


print(exchange(usd=100, rub=8500))
print(exchange(usd=100, rate=85))
print(exchange(rub=1000, rate=85))
# 85.0
# 8500
# 11.764705882352942
# print(exchange(rub=1000, rate=85, usd=90))
# ValueError: Too many arguments
# print(exchange(rub=1000))
# ValueError: Not enough arguments