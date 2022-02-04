# Напишите функцию is_leap(year), которая принимает на вход год и возвращает True, если год високосный, иначе — False.
# Условие для проверки на високосность: год делится на 400 или год делится на 4, но не на 100.
# def is_leap(year):
#     if year % 400 == 0 or (year % 4 == 0 and year % 100 != 0):
#         return True
#
#     return False

# print(is_leap(2000))
# print(is_leap(1900))
# print(is_leap(2020))

# Напишите функцию check_date(day, month, year), которая проверяет корректность даты рождения по следующим условиям:
#
# Все аргументы должны быть целыми числами (проверить с помощью type(day) is int).
# Годом рождения не может быть год до 1900 и год после 2022.
# Номер месяца не может быть больше 12 и меньше 1.
# Номер дня не может быть больше 31 и меньше 1.
# В сентябре, апреле, июне и ноябре 30 дней.
# Если год является високосным, то в феврале (второймесяц) должно быть 29 дней, в противном случае — 28.
# Если дата корректна, вернуть True, если же хотя бы одно из представленных условий не было выполнено — False.


# def check_date(day, month, year):
#     def is_leap(year):
#         if year % 400 == 0 or (year % 4 == 0 and year % 100 != 0):
#             return True
#
#         return False
#
#     if type(day) is not int:
#         return False
#
#     if type(month) is not int:
#         return False
#
#     if type(year) is not int:
#         return False
#
#     if year < 1900 or year > 2022:
#         return False
#
#     if month < 1 or month > 12:
#         return False
#
#     if day < 1 or day > 31:
#         return False
#
#     if month in [4, 6, 9, 11] and day > 30:
#         return False
#
#     if not is_leap(year) and month == 2 and day > 28:
#         return False
#
#     return True
#
#
# print(check_date(18, 9, 1999))
# print(check_date(29, 2, 2000))
# print(check_date(29, 2, 2021))
# print(check_date(13, 13, 2021))
# print(check_date(13.5, 12, 2021))


# Представьте, что вы пишете приложение, предусматривающее регистрацию пользователей.
# Давайте реализуем небольшой функционал регистрации. Не забудем также про «проверку на дурака».
#
# Напишите функцию register(surname, name, date, middle_name, registry).
#
# Функция имеет следующие аргументы:
#
# surname — фамилия;
# name — имя;
# date — дата рождения (в виде кортежа из трёх чисел — день, месяц, год);
# middle_name — отчество ;
# registry — список, в который необходимо добавить полученные аргументы
# в виде кортежа в следующем порядке: фамилия, имя, отчество, день, месяц, год рождения.
# Функция должна возвращать список, в который добавила запись.

def register(surname, name, date, middle_name=None, registry=None):
    def check_date(day, month, year):
        def is_leap(year):
            if year % 400 == 0 or (year % 4 == 0 and year % 100 != 0):
                return True

            return False

        if type(day) is not int:
            return False

        if type(month) is not int:
            return False

        if type(year) is not int:
            return False

        if year < 1900 or year > 2022:
            return False

        if month < 1 or month > 12:
            return False

        if day < 1 or day > 31:
            return False

        if month in [4, 6, 9, 11] and day > 30:
            return False

        if not is_leap(year) and month == 2 and day > 28:
            return False

        return True

    if not check_date(*date):
        raise ValueError("Invalid Date!")

    if registry is None:
        registry = []

    registry.append((surname, name, middle_name, date[0], date[1], date[2]))

    return registry


reg = register('Petrova', 'Maria', (13, 3, 2003), 'Ivanovna')
reg = register('Ivanov', 'Sergej', (24, 9, 1995), registry=reg)
reg = register('Smith', 'John', (13, 2, 2003), registry=reg)
print(reg)

# reg = register('Ivanov', 'Sergej', (24, 13, 1995))