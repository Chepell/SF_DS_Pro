# Представьте, что пытаетесь выгрузить несколько новостей с сайта.
# У вас есть список путей до интересующих вас статей. Пример такого списка:
docs = [
    '//doc/5041434?query=data%20science',
    '//doc/5041567?query=data%20science',
    '//doc/4283670?query=data%20science',
    '//doc/3712659?query=data%20science',
    '//doc/4997267?query=data%20science',
    '//doc/4372673?query=data%20science',
    '//doc/3779060?query=data%20science',
    '//doc/3495410?query=data%20science',
    '//doc/4308832?query=data%20science',
    '//doc/4079881?query=data%20science'
]

begin_link = 'https://www.kommersant.ru'

full_links = map(lambda x: begin_link + x, docs)

print(list(full_links))

# Вы — сотрудник отдела разработки в МФЦ.
# МФЦ предоставляет некоторый спектр услуг многодетным семьям.
# Необходимо написать функционал, который позволяет отфильтровать среди всех
# запрашиваемых пользователем услуг (их количество произвольное) только те, которые предоставляются многодетным семьям.
family_list = [
    'certificate of a large family',
    'social card',
    'maternity capital',
    'parking permit',
    'tax benefit',
    'reimbursement of expenses',
    "compensation for the purchase of children's goods"
]

def family(*args):
    return list(filter(lambda x: x in family_list, args))

print(family('newborn registration', 'parking permit', 'maternity capital', 'tax benefit', 'medical policy'))


# Мы вновь занимаемся регистрацией пользователей. В нашем распоряжении имеется список кортежей reg.
# В каждом кортеже хранится информация о зарегистрированном пользователе
# и его дате рождения в формате (Фамилия, Имя, день, месяц, год):
# Выберите из списка reg только те записи, в которых год рождения пользователя 2000 и больше (2001, 2002 и т. д.).
# Из оставшихся записей составьте новый список кортежей, в котором фамилия и имя объединены
# в одну строку по следующему шаблону Фамилия И. (обратите внимание на точку после сокращения имени).


reg = [('Ivanov', 'Sergej', 24, 9, 1995),
      ('Smith', 'John', 13, 2, 2003),
      ('Petrova', 'Maria', 13, 3, 2003)]

filter_by_year = filter(lambda x: x[4] >= 2000, reg)
new_format = map(lambda x: (x[0] + " " + x[1][0].upper() + ".", x[2], x[3], x[4]), filter_by_year)


surnames = ['Ivanov', 'Smirnov', 'Kuznetsova', 'Nikitina']
names = ['Sergej', 'Ivan', 'Maria', 'Elena']

fio_list = []

for name, surname in zip(names, surnames):
    fio_list.append(f"{name} {surname}")

print(fio_list)



# Перед вами стоит задача разбить пользователей на три группы, чтобы в дальнейшем проводить А/B/C-тестирование.
# Например, первой группе вы выдаёте первый вариант интерфейса вашего приложения,
# второй группе — второй, третьей группе — третий.
# Затем вы будете сравнивать реакцию пользователей и делать вывод, какой интерфейс лучше.
#
# Для генерации групп дан генератор group_gen.
# Аргументом данного генератора является число n — число групп (у нас n = 3).
# При каждом вызове  генератора он возвращает число от 1 до n.
# После достижения n генератор снова возвращает номера групп, начиная с 1.

def group_gen(n=3):
    while True:
        for i in range(1, n+1):
            yield i

users = ['Smith J.', 'Petrova M.', 'Lubimov M.', 'Holov J.']

# Напишите функцию print_groups, которая принимает список с именами пользователей.
# Используя генератор групп group_gen, печатайте на экран:
#
# <Фамилия И.> in group <номер группы по порядку>.

def print_groups(users):
    for name, number in zip(users, group_gen()):
        text = f"{name} in group {number}"
        print(text)

print_groups(users)