counter = 0


def count():
    global counter  # обращение к глобальной переменной
    counter += 1


counter = 0

while counter < 10:
    count()
    print(counter)


# объявляем внешнюю функцию для регистрации сотрудников
def register_employee(name, surname):
    # объявляем функцию для промежуточных вычислений
    def create_full_name():
        # функция использует внешние переменные name и surname
        sep = ' '  # разделитель между именем и фамилией
        result = name + sep + surname  # вычисляем полное имя
        return result

    full_name = create_full_name()  # вызываем внутреннюю функцию
    # выводим результат на экран, используя внешнюю переменную company_name
    print('Employee {} is registered with the company {}'.format(full_name, company_name))


company_name = 'TheBlindMice'  # название компании
register_employee('John', 'Doe')


print("**************")
# Напишите функцию-копилку с названием saver(), которая не принимает никаких аргументов.
# Она должна возвращать внутреннюю функцию adder(),
# которая принимает на вход одно число и возвращает сумму в копилке после прибавления числа.
#
# Изначально в новой копилке хранится 0.

def saver():
    sum_value = 0
    def adder(add_num):
        nonlocal sum_value
        sum_value += add_num
        return sum_value
    return adder


pig = saver()
print(pig(25))
print(pig(100))
print(pig(19))